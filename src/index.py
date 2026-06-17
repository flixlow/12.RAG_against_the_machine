from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter as RCTS
import ast


class Index:
    def __init__(self, dir: str, chunk_size: int) -> None:
        self.files: list[Path] = self.listing(dir)
        self.dc_splitters: RCTS = RCTS(chunk_overlap=int(chunk_size * 0.05),
                                       chunk_size=chunk_size)
        self.chunks: list[dict[str, str | int]] = []
        self.open()
        self.output()

    @staticmethod
    def listing(dir: str) -> list[Path]:
        return [f for f in Path(dir).rglob('*') if f.is_file()]

    def chunk_dc(self, document: str) -> None:
        self.dc_splitters.split_text(document)

    def chunk_py(self, d: str) -> None:
        pass

    def open(self) -> None:
        for file in self.files:
            try:
                if file.suffix in ['.py', '.txt', '.md']:
                    with open(file) as f:
                        data = f.read()
                        if file.suffix == '.py':
                            self.chunk_py(data)
                        if file.suffix == '.txt' or file.suffix == '.md':
                            self.chunk_dc(data)
            except (PermissionError, FileNotFoundError, IsADirectoryError):
                continue

    def output(self) -> None:
        print("Ingestion complete! Indices saved under data/processed/")
