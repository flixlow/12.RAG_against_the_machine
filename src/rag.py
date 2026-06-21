from src.models import UnansweredQuestion
from src.errors import SearchError, InputSingleQueryError
from src.search import Search
from src.index import Index
from pathlib import Path
import time
import json
import os


class Rag:
    def index(self, dir: str = "data/raw/vllm-0.10.1",
              max_chunk_size: int = 2000) -> None:
        start = time.time()
        index = Index(dir=dir, chunk_size=max_chunk_size)
        index.open()
        index.save()
        index.index()
        print(f"\n\033[34mIngestion complete in {time.time() - start:.3f}s!")
        print("\033[0;1mIndices saved under data/processed/")

    def search(self, query: str | None = None, k: int = 5,
               save_directory: str = "data/output/search_results"
               ) -> None:
        if query is not None:
            single = UnansweredQuestion(question=query)
        else:
            raise InputSingleQueryError

        searcher = Search(
            rag_questions=[single],
            k=k,
            save_dir=save_directory,
            file="single_query.json"
            )
        searcher.search_dataset()

    def search_dataset(self,
                       dataset_path: str,
                       k: int = 5,
                       save_directory: str = "data/output/search_results"
                       ) -> None:
        if not Path(dataset_path).exists():
            raise SearchError(f"invalid dataset_path: {dataset_path}")

        try:
            with open(dataset_path) as f:
                content = f.read()
                questions = json.loads(content)
        except OSError:
            raise SearchError(f"can't loading content from {dataset_path}")
        except json.JSONDecodeError as e:
            raise SearchError from e

        searcher = Search(
            **questions,
            k=k,
            save_dir=save_directory,
            file=os.path.basename(dataset_path)
            )
        searcher.search_dataset()

    def answer(self, query: str, k: int = 5) -> None:
        pass

    def answer_dataset(
            self, search_results_path: str,
            save_directory: str = "data/output/search_results_and_answer"
            ) -> None:
        print("answer_dataset")

    def evaluate(self) -> None:
        print("evaluate")
