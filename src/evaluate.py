import json
from typing import Any
from pydantic import BaseModel
from src.errors import EvaluateError
from src.models import AnsweredQuestion, RagDataset, StudentSearchResults


class Evaluator(BaseModel):
    student_search_results_path: str
    dataset_path: str

    def model_post_init(self, _: Any = None) -> None:
        self._student: StudentSearchResults = self.load_model(
            self.student_search_results_path, StudentSearchResults)
        self._dataset: RagDataset = self.load_model(
            self.dataset_path, RagDataset)
        self._reference = self.create_reference_set()

    @staticmethod
    def load_model(path: str, model_cls: type) -> Any:
        try:
            with open(path, encoding='utf-8') as f:
                return model_cls(**json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            raise EvaluateError from e

    @staticmethod
    def iou(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
        if start_a >= end_a or start_b >= end_b:
            return False

        intersection = max(0, min(end_a, end_b) - max(start_a, start_b))
        union = max(end_a, end_b) - min(start_a, start_b)

        return True if union > 0 and intersection / union >= 0.05 else False

    def create_reference_set(self) -> dict[str, tuple[str, int, int]]:
        reference: dict[str, tuple[str, int, int]] = {}
        for question in self._dataset.rag_questions:
            if isinstance(question, AnsweredQuestion):
                src = question.sources[0]
                reference[question.question_id] = (
                    src.file_path,
                    src.first_character_index,
                    src.last_character_index
                )
        return reference

    def recall_at_k(self, question_id: str, k: int) -> bool:
        reference = self._reference.get(question_id, None)

        for result in self._student.search_results:
            if result.question_id == question_id:
                student_result = result
                break

        if reference is None or student_result is None:
            return False

        ref_path, reference_start_index, reference_end_index = reference
        top_k_results = student_result.retrieved_sources[:k]

        for chunk in top_k_results:
            if chunk.file_path == ref_path and self.iou(
             reference_start_index, reference_end_index,
             chunk.first_character_index, chunk.last_character_index):
                return True

        return False

    def evaluate(self) -> None:
        reference = self._reference
        summary: dict[int, float] = {}
        ks = [1, 3, 5, 10]

        if not reference:
            print("Empty set of questions.")
            return

        for k in ks:
            scores = []
            for question_id in reference:
                score = self.recall_at_k(question_id, k)
                scores.append(score)
            summary[k] = sum(scores) / len(scores) if scores else 0.0

        for k in ks:
            value = summary.get(k, 0.0)
            percentage = value * 100
            print(f"Recall@{k}: {value:.2f} ({percentage:.1f}%)")
