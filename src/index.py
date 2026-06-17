from langchain_text_splitters import (RecursiveCharacterTextSplitter as RCTS,
                                      Language)
from src.models import ChunkData, MinimalSource
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any
import bm25s
import json

class Index(BaseModel):
    dir: str
    chunk_size: int = Field(gt=0)

    def model_post_init(self, _: Any) -> None:
        self._chunks: list[ChunkData] = []
        self._files: list[Path] = self.listing(self.dir)
        self.open(self.chunk_size)
        self.split()
        self.index()

    @staticmethod
    def listing(dir: str) -> list[Path]:
        return [f for f in Path(dir).rglob('*') if f.is_file()]

    def chunking(self, splitter: RCTS, file: str, content: str) -> None:
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

    def open(self, chunk_size: int) -> None:
        overlap: int = int(chunk_size * 0.05)
        txt_splitter = RCTS(chunk_size=chunk_size,
                            chunk_overlap=overlap,
                            add_start_index=True)
        py_splitter = RCTS.from_language(chunk_size=chunk_size,
                                         chunk_overlap=overlap,
                                         language=Language.PYTHON,
                                         add_start_index=True)
        md_splitter = RCTS.from_language(chunk_size=chunk_size,
                                         chunk_overlap=overlap,
                                         language=Language.MARKDOWN,
                                         add_start_index=True)
        splitters = {".py": py_splitter, ".md": md_splitter}

        for file in self._files:
            try:
                if file.suffix in ['.py', '.txt', '.md']:
                    with open(file) as f:
                        self.chunking(splitters.get(file.suffix, txt_splitter),
                                      file.as_posix(), f.read())
            except (PermissionError, FileNotFoundError, IsADirectoryError):
                print(f"\033[1;38;5;208m[WARNING]\033[0m Can't index {file}.")
                continue

    def split(self) -> None:
        file = Path("data/processed/chunks/splitted.json")
        file.parent.mkdir(exist_ok=True, parents=True)
        try:
            with open(file, 'w') as f:
                chunks = [chunk.model_dump() for chunk in self._chunks]
                json.dump(chunks, f, ensure_ascii=True, indent=4)
        except (PermissionError, FileNotFoundError, IsADirectoryError):
            print(f"\033[1;38;5;208m[ERROR]\033[0m Can't open file {file}.")
        print("Ingestion complete! Indices saved under data/processed/")

    def index(self) -> None:
        corpus = [chunk['content'] for chunk in self._chunks]
        folder = Path("data/processed/bm25_index/")
        folder.mkdir(exist_ok=True, parents=True)
        try:
