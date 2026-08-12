import json
from src.errors import EvaluateError
from pydantic import BaseModel
from typing import Any
from src.models import StudentSearchResults, RagDataset


class Evaluator(BaseModel):
    student_search_results_path: str
    dataset_path: str

    def model_post_init(self, _: Any = None) -> None:
        self._search_results: StudentSearchResults = self._load_search_results(
            self.student_search_results_path)
        self._dataset: RagDataset = self._load_dataset(self.dataset_path)

    @staticmethod
    def _load_search_results(path: str) -> StudentSearchResults:
        try:
            with open(path) as f:
                return StudentSearchResults(**json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            raise EvaluateError from e

    @staticmethod
    def _load_dataset(path: str) -> RagDataset:
        try:
            with open(path) as f:
                return RagDataset(**json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            raise EvaluateError from e

    def evaluate(self) -> None:
        pass
