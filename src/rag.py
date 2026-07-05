from src.models import UnansweredQuestion
from src.errors import SearchError, InputSingleQueryError
from src.search import Search
from src.config import Config
from src.answer import Answer
from src.index import Index
from pathlib import Path
import time
import json
import os


class Rag:
    def index(self, dir: str = Config.RAW,
              max_chunk_size: int = 2000) -> None:
        start = time.time()
        index = Index(dir=dir, chunk_size=max_chunk_size)
        index.open()
        index.save()
        index.index()
        print(f"\n\033[34mIngestion complete in {time.time() - start:.3f}s!")
        print(f"\033[0;1mIndices saved under {Config.PROCESSED}")

    def search(self, query: str | None = None, k: int = 5,
               save_directory: str = Config.SEARCH_PATH) -> None:
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
                       save_directory: str = Config.SEARCH_PATH
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

    def answer(self, query: str | None = None, k: int = 5,
               save_directory: str = Config.ANSWER_PATH) -> None:
        pass

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str = Config.ANSWER_PATH
            ) -> None:
        provider = Answer(results_path=student_search_results_path,
                          save_directory=save_directory)
        provider.answer()

    def evaluate(self) -> None:
        print("evaluate")
