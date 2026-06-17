from src.index import Index


class Rag:
    def index(self, dir: str = "data/raw/vllm-0.10.1",
              chunk_size: int = 2000) -> None:
        Index(dir=dir, chunk_size=chunk_size)

    def search(self, query: str, k: int = 5) -> None:
        print("search")

    def search_dataset(self) -> None:
        print("search_data_set")

    def answer(self, query: str, k: int = 5) -> None:
        print("answer")

    def answer_dataset(self) -> None:
        print("answer_dataset")

    def evaluate(self) -> None:
        print("evaluate")
