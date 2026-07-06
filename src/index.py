from langchain_text_splitters import (RecursiveCharacterTextSplitter as RCTS,
                                      Language)
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
import json


class Index(BaseModel):
    dir: str
    chunk_size: int = Field(gt=0, le=2000)
    emb_flag: bool = Field(default=False)

    def model_post_init(self, _: Any) -> None:
        self._chunks: list[ChunkData] = []
        self._files: list[Path] = self.listing(self.dir)
        if self.emb_flag:
            if Path(Config.CHROMA_PATH).exists():
                shutil.rmtree(Config.CHROMA_PATH)
            cli: ClientAPI = chromadb.PersistentClient(path=Config.CHROMA_PATH)
            self._collection: Collection = cli.get_or_create_collection("coll")

    @staticmethod
    def listing(dir: str) -> list[Path]:
        if not Path(dir).exists():
            raise RagIndexError("The given path does not exist.")
        return [f for f in Path(dir).rglob('*') if f.is_file()]

    def open(self) -> None:
        overlap: int = int(self.chunk_size * 0.02)
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
                    with open(file) as f:
                        self.chunking(splitters.get(file.suffix, txt_splitter),
                                      file.as_posix(), f.read())
            except OSError:
                print(f"\033[1;38;5;208m[WARNING]\033[0m Can't open {file}.")
                continue

    def chunking(self, splitter: RCTS, file: str, content: str) -> None:
        # clean_file = file.replace('/', ' ').replace('_', ' ')
        # chunks = splitter.create_documents([clean_file*5 + '\n' + content])
        chunks = splitter.create_documents([content])

        for chunk in chunks:
            start = chunk.metadata['start_index']
            source = MinimalSource(
                file_path=file,
                first_character_index=start,
                last_character_index=start + len(chunk.page_content)
            )
            self._chunks.append(ChunkData(
                content=chunk.page_content,
                metadata=source
            ))

    def save(self) -> None:
        if self._chunks == []:
            raise RagIndexError("No data has been processed: "
                                "please, ensure raw data is available.")
        file = Path(Config.CHUNKS_PATH)
        file.parent.mkdir(exist_ok=True, parents=True)

        try:
            with open(file, 'w') as f:
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
        corpus = [chunk.content for chunk in self._chunks]

        corpus_tokens = bm25s.tokenize(corpus)
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens, leave_progress=True)
        if Path(Config.BM25_PATH).exists():
            shutil.rmtree(Config.BM25_PATH)
        retriever.save(Config.BM25_PATH)
        print(len(self._chunks))
