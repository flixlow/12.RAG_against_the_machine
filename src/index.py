from langchain_text_splitters import (RecursiveCharacterTextSplitter as RCTS,
                                      Language)
from chromadb.utils.embedding_functions import (OllamaEmbeddingFunction,
                                                EmbeddingFunction)
from src.models import ChunkData, MinimalSource
from pydantic import BaseModel, Field
from tqdm import tqdm  # type: ignore
from src.errors import RagIndexError
from chromadb import ClientAPI, Collection
from src.config import Config
import bm25s  # type: ignore
from pathlib import Path
from typing import Any, cast
import chromadb
import shutil
import hashlib
import json


class Index(BaseModel):
    dir: str
    chunk_size: int = Field(gt=0, le=2000)
    embedding_flag: bool = Field(default=False)
    incremental_flag: bool = Field(default=True)

    def model_post_init(self, _: Any) -> None:
        self._deleted: list[str] = []
        self._manifest: dict[str, str] = Index._load_manifest()
        self._chunks: list[ChunkData] = self._load_existing_chunks()
        self._files: list[Path] = self.listing()

        if self.embedding_flag:
            if Path(Config.CHROMA_PATH).exists():
                shutil.rmtree(Config.CHROMA_PATH)
            oef = OllamaEmbeddingFunction(model_name=Config.EMBEDDING_MODEL)
            ef = cast(EmbeddingFunction, oef)
            cli: ClientAPI = chromadb.PersistentClient(path=Config.CHROMA_PATH)
            self._collection: Collection = cli.get_or_create_collection(
                "collection", embedding_function=ef)

    @staticmethod
    def _hash(file: Path) -> str:
        return hashlib.sha256(file.read_bytes()).hexdigest()

    @staticmethod
    def _load_manifest() -> dict[str, str]:
        manifest = Path(Config.MANIFEST)
        return json.loads(manifest.read_text()) if manifest.exists() else {}

    def _load_existing_chunks(self) -> list[ChunkData]:
        path = Path(Config.CHUNKS_PATH)
        if not self.incremental_flag or not path.exists():
            if path.exists():
                shutil.rmtree(Config.CHUNKS_PATH)
            return []
        content = path.read_text(encoding='utf-8')
        return [ChunkData(**chunk) for chunk in json.loads(content)]

    def _save_manifest(self) -> None:
        path = Path(Config.MANIFEST)
        manifest = json.dumps(self._manifest, ensure_ascii=False, indent=4)
        path.write_text(manifest, encoding='utf-8')

    def listing(self) -> list[Path]:
        dir = Path(self.dir)
        if not dir.exists():
            raise RagIndexError("The given path does not exist.")

        if self.incremental_flag is False or self.embedding_flag is True:
            return [f for f in dir.rglob('*') if f.is_file()]

        seen: list[str] = []
        new: list[Path] = []
        modified: list[Path] = []
        deleted: list[str] = []
        manifest = self._manifest

        for f in dir.rglob('*'):
            if not f.is_file() or f.suffix not in ['.py', '.txt', '.md']:
                continue
            key = f.as_posix()
            seen.append(key)
            if manifest.get(key) is None:
                new.append(f)
            elif manifest[key] != self._hash(f):
                modified.append(f)

        deleted = [k for k in manifest.keys() if k not in set(seen)]

        for d in deleted:
            del self._manifest[d]

        self.remove_obsolete([m.as_posix() for m in modified] + deleted)

        return new + modified

    def remove_obsolete(self, obsolete: list[str]) -> None:
        self._chunks = [
            c for c in self._chunks if c.metadata.file_path not in obsolete]

    def open(self) -> None:
        overlap: int = int(self.chunk_size * 0.2)
        txt_splitter = RCTS(chunk_size=self.chunk_size,
                            chunk_overlap=overlap,
                            add_start_index=True)
        py_splitter = RCTS.from_language(chunk_size=self.chunk_size,
                                         chunk_overlap=overlap,
                                         language=Language.PYTHON,
                                         add_start_index=True)
        md_splitter = RCTS.from_language(chunk_size=self.chunk_size,
                                         chunk_overlap=overlap,
                                         language=Language.MARKDOWN,
                                         add_start_index=True)
        splitters = {".py": py_splitter, ".md": md_splitter}

        for file in tqdm(self._files, desc="chunking"):
            try:
                with open(file, encoding='utf-8') as f:
                    self.chunking(splitters.get(file.suffix, txt_splitter),
                                  file, f.read())
                self._manifest[file.as_posix()] = self._hash(file)
            except OSError:
                print(f"\033[1;38;5;208m[WARNING]\033[0m Can't open {file}.")
                continue

        self._save_manifest()

    def chunking(self, splitter: RCTS, file: Path, content: str) -> None:
        chunks = splitter.create_documents([content])

        for chunk in chunks:
            start = chunk.metadata['start_index']
            end = start + len(chunk.page_content)
            source = MinimalSource(file_path=file.as_posix(),
                                   first_character_index=start,
                                   last_character_index=end)
            self._chunks.append(
                ChunkData(content=chunk.page_content, metadata=source))

    def save(self) -> None:
        if self._chunks == []:
            raise RagIndexError("No data has been processed: "
                                "please, ensure raw data is available.")
        file = Path(Config.CHUNKS_PATH)
        file.parent.mkdir(exist_ok=True, parents=True)

        try:
            with open(file, 'w', encoding='utf-8') as f:
                chunks = [chunk.model_dump() for chunk in self._chunks]
                json.dump(chunks, f, ensure_ascii=False, indent=4)
        except OSError as e:
            raise RagIndexError(f"Can't save chunk to file {file}.") from e

    def embedding(self, size: int = 42) -> None:
        chunks = self._chunks
        ids = [str(i) for i in range(len(chunks))]
        docs = [c.content for c in chunks]

        for start in tqdm(range(0, len(chunks), size), desc="Embedding: "):
            end = start + size
            self._collection.add(ids=ids[start:end],
                                 documents=docs[start:end])

    def index(self) -> None:
        corpus = [c.content for c in self._chunks]

        corpus_tokens = bm25s.tokenize(corpus)
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens, leave_progress=True)
        if Path(Config.BM25_PATH).exists():
            shutil.rmtree(Config.BM25_PATH)
        retriever.save(Config.BM25_PATH)
