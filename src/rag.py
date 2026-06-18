import time
from src.index import Index
from src.errors import SearchError
from src.search import SingleQuery


class Rag:
    def index(self, dir: str = "data/raw/vllm-0.10.1",
              chunk_size: int = 2000) -> None:
        start = time.time()
        index = Index(dir=dir, chunk_size=chunk_size)
        index.open()
        index.save()
        index.index()
        print(f"\n\033[34mIngestion complete in {time.time() - start:.3f}s!")
        print("\033[0;1mIndices saved under data/processed/")

    def search(self, query: str | None = None, k: int = 5) -> None:
        if query is None:
            raise SearchError("Please, enter a query: "
                              "make search ARG=\"'--query' '--k'\"")
        search = SingleQuery(query=query, k=k)
        search.search()

    def search_dataset(self) -> None:
        print("search_data_set")

    def answer(self, query: str, k: int = 5) -> None:
        print("answer")

    def answer_dataset(self) -> None:
        print("answer_dataset")

    def evaluate(self) -> None:
        print("evaluate")
