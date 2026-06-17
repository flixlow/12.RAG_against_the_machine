from src.index import Index


class Rag:
    def index(self, dir: str = "vllm-0.10.1", chunk_size: int = 2000) -> None:
        Index(dir, chunk_size)
        # open files
        # chuncking strategy for markdown or python files

    def search(self) -> None:
        print("search")

    def search_dataset(self) -> None:
        print("search_data_set")

    def answer(self, k: int) -> None:
        print("answer")

    def answer_dataset(self) -> None:
        print("answer_dataset")

    def evaluate(self) -> None:
        print("evaluate")
