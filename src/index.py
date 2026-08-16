from langchain_text_splitters import (RecursiveCharacterTextSplitter as RCTS,
                                      Language)
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from src.models import ChunkData, MinimalSource
from pydantic import BaseModel, Field
from tqdm import tqdm  # type: ignore
from src.errors import RagIndexError
from chromadb import ClientAPI, Collection
from src.config import Config
import bm25s  # type: ignore
from pathlib import Path
from typing import Any
import chromadb
import shutil
import hashlib
import json


class Index(BaseModel):
    dir: str
    chunk_size: int = Field(gt=0, le=2000)
    emb_flag: bool = Field(default=False)
    incremental_flag: bool = Field(default=True)

    def model_post_init(self, _: Any) -> None:
        self._chunks: list[ChunkData] = []
        self._side_chunks: list[str] = []
        self._deleted: list[str] = []
        self._manifest: dict[str, str] = Index._load_manifest()
        self._files: list[Path] = self.listing(self.dir)

        if self.emb_flag:
            if Path(Config.CHROMA_PATH).exists():
                shutil.rmtree(Config.CHROMA_PATH)
            ef = OllamaEmbeddingFunction(model_name=Config.EMBEDDING_MODEL)
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

    def _save_manifest(self) -> None:
        Path(Config.MANIFEST).write_text(
            json.dumps(self._manifest, indent=2), encoding='utf-8')

    def load_existing_chunks(self) -> list[ChunkData]:
        return [ChunkData(**json.loads(Config.CHUNKS_PATH))]

    def listing(self, dir: str) -> list[Path]:
        seen: set[str] = set()
        file_to_index: list[Path] = []

        if not Path(dir).exists():
            raise RagIndexError("The given path does not exist.")

        if self.incremental_flag is False:
            return [f for f in Path(dir).rglob('*') if f.is_file()]

        self._chunks = self.load_existing_chunks()

        for file in Path(dir).rglob('*'):
            if file.is_file():
                key = file.as_posix()
                seen.add(key)
                file_hash = Index._hash(file)
                if self._manifest.get(key) != file_hash:
                    file_to_index.append(file)
                    self._manifest[key] = file_hash

        self._deleted = [k for k in self._manifest if k not in seen]
        for k in self._deleted:
            del self._manifest[k]

        return file_to_index

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
                if file.suffix in ['.py', '.txt', '.md']:
                    with open(file, encoding='utf-8') as f:
                        self.chunking(splitters.get(file.suffix, txt_splitter),
                                      file, f.read())
            except OSError:
                print(f"\033[1;38;5;208m[WARNING]\033[0m Can't open {file}.")
                continue

        self._save_manifest()

    def chunking(self, splitter: RCTS, file: Path, content: str) -> None:
        path = str(file.parent).replace('/', ' ') + '\n'
        chunks = splitter.create_documents([content])
        content = f"FILE={file.name}\n" * 5 + f"PATH={path}"

        for chunk in chunks:
            start = chunk.metadata['start_index']
            end = start + len(chunk.page_content)
            source = MinimalSource(file_path=file.as_posix(),
                                   first_character_index=start,
                                   last_character_index=end)
            self._side_chunks.append(content + chunk.page_content)
            self._chunks.append(
                ChunkData(content=chunk.page_content, metadata=source))

    def save(self) -> None:
        if self._chunks == []:
            raise RagIndexError("No data has been processed: "
                                "please, ensure raw data is available.")
        file = Path(Config.CHUNKS_PATH)
        side = Path(Config.SIDE_CHUNKS_PATH)
        file.parent.mkdir(exist_ok=True, parents=True)

        try:
            with open(file, 'w', encoding='utf-8') as f:
                chunks = [chunk.model_dump() for chunk in self._chunks]
                json.dump(chunks, f, ensure_ascii=False, indent=4)
            with open(side, 'w', encoding='utf-8') as f:
                json.dump(self._side_chunks, f, ensure_ascii=False, indent=4)
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
        corpus = self._side_chunks

        corpus_tokens = bm25s.tokenize(corpus)
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens, leave_progress=True)
        if Path(Config.BM25_PATH).exists():
            shutil.rmtree(Config.BM25_PATH)
        retriever.save(Config.BM25_PATH)
