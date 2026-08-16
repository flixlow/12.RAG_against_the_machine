from src.models import UnansweredQuestion
from src.errors import SearchError, InputSingleQueryError
from src.search import Search
from src.config import Config
from src.answer import Answer
from src.index import Index
from src.evaluate import Evaluator
from pathlib import Path
import time
import json
import os


class Rag:
    def index(self, dir: str = Config.RAW, max_chunk_size: int = 2000,
              embedding: bool = False, incremental: bool = True) -> None:
        start = time.time()
        index = Index(dir=dir, chunk_size=max_chunk_size,
                      embedding_flag=embedding, incremental_flag=incremental)
        index.open()
        index.save()
        index.index()
        if embedding:
            index.embedding()
        print(f"\n\033[34mIngestion complete in {time.time() - start:.3f}s!")
        print(f"\033[0;1mIndices saved under {Config.PROCESSED}\033[0m")

    def search(self, query: str | None = None, k: int = 5,
               save_directory: str = Config.SEARCH_PATH,
               hybrid: bool = False, expansion: bool = False) -> None:
        if query is not None:
            single = UnansweredQuestion(question=query)
        else:
            raise InputSingleQueryError
        searcher = Search(rag_questions=[single], k=k, save_dir=save_directory,
                          file=Config.SIGLE_QUERY,
                          hybrid=hybrid, expansion=expansion)
        searcher.search_dataset()

    def search_dataset(self, dataset_path: str, k: int = 5,
                       save_directory: str = Config.SEARCH_PATH,
                       hybrid: bool = False, expansion: bool = False) -> None:
        if not Path(dataset_path).exists():
            raise SearchError(f"invalid dataset_path: {dataset_path}")
        try:
            with open(dataset_path, encoding='utf-8') as f:
                questions = json.load(f)
        except OSError:
            raise SearchError(f"can't loading content from {dataset_path}")
        except json.JSONDecodeError as e:
            raise SearchError from e
        searcher = Search(**questions, k=k, save_dir=save_directory,
                          file=os.path.basename(dataset_path),
                          hybrid=hybrid, expansion=expansion)
        searcher.search_dataset()

    def answer(self, query: str | None = None, k: int = 5,
               save_directory: str = Config.ANSWER_PATH) -> None:
        pass

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str = Config.ANSWER_PATH) -> None:
        provider = Answer(results_path=student_search_results_path,
                          save_directory=save_directory)
        provider.answer()

    def evaluate(self, student_search_results_path: str,
                 dataset_path: str) -> None:
        print("Recall@k Calculation")

        evaluator = Evaluator(
            student_search_results_path=student_search_results_path,
            dataset_path=dataset_path)
        evaluator.evaluate()
