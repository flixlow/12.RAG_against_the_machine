from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter as RCTS
from langchain_text_splitters import Language
from langchain_core.documents import Document


class Index:
    def __init__(self, dir: str, chunk_size: int) -> None:
        self.files: list[Path] = self.listing(dir)
        self.chunks: list[list[Document]] = []
        self.open(chunk_size)
        self.output()

    @staticmethod
    def listing(dir: str) -> list[Path]:
        return [f for f in Path(dir).rglob('*') if f.is_file()]

    def chunking(self, splitter: RCTS, file: str, content: str) -> None:
        documents = splitter.create_documents([content], [{"file_path": file}])
        print(documents)
        self.chunks.append(splitter.split_documents(documents))

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

        for file in self.files:
            try:
                if file.suffix in ['.py', '.txt', '.md']:
                    with open(file) as f:
                        self.chunking(splitters.get(file.suffix, txt_splitter),
                                      file.as_posix(), f.read())
            except (PermissionError, FileNotFoundError, IsADirectoryError):
                print(f"\033[1;38;5;208m[WARNING]\033[0m Can't index {file}.")
                continue

    def output(self) -> None:
        output_file = Path("data/processed/chunks/splitted.json")
        output_file.parent.mkdir(exist_ok=True, parents=True)
        try:
            with open(output_file, 'w') as f:
                pass
        except (PermissionError, FileNotFoundError, IsADirectoryError):
            print("\033[1;38;5;208m[ERROR]\033[0m",
                  "Can't open file to store chunks.")
        print("Ingestion complete! Indices saved under data/processed/")
