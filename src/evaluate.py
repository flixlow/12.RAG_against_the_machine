import json
from typing import Any
from pydantic import BaseModel
from src.errors import EvaluateError
from src.models import AnsweredQuestion, RagDataset, StudentSearchResults


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
            with open(path, encoding='utf-8') as f:
                return StudentSearchResults(**json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            raise EvaluateError from e

    @staticmethod
    def _load_dataset(path: str) -> RagDataset:
        try:
            with open(path, encoding='utf-8') as f:
                return RagDataset(**json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            raise EvaluateError from e

    @staticmethod
    def _iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
        if start_a >= end_a or start_b >= end_b:
            return 0.0

        intersection = max(0, min(end_a, end_b) - max(start_a, start_b))
        union = max(end_a, end_b) - min(start_a, start_b)
        if union <= 0:
            return 0.0
        return intersection / union

    def create_reference_set(self) -> dict[str, list[tuple[str, int, int]]]:
        ref: dict[str, list[tuple[str, int, int]]] = {}
        for question in self._dataset.rag_questions:
            if isinstance(question, AnsweredQuestion):
                ref[question.question_id] = [
                    (source.file_path, source.first_character_index,
                     source.last_character_index)
                    for source in question.sources
                ]
        return ref

    def recall_at_k(self, question_id: str, k: int) -> float:
        reference = self.create_reference_set().get(question_id, [])
        if not reference:
            return 0.0

        student_result = next(
            (result for result in self._search_results.search_results
             if result.question_id == question_id),
            None,
        )
        if student_result is None:
            return 0.0

        top_k_results = student_result.retrieved_sources[:k]
        found: set[tuple[str, int, int]] = set()

        for ref_path, ref_start, ref_end in reference:
            for candidate in top_k_results:
                if candidate.file_path != ref_path:
                    continue

                iou = self._iou(ref_start, ref_end,
                                candidate.first_character_index,
                                candidate.last_character_index)
                if iou >= 0.05:
                    found.add((ref_path, ref_start, ref_end))
                    break

        return len(found) / len(reference)

    def evaluate(self) -> None:
        reference = self.create_reference_set()
        if not reference:
            print("Empty set of questions.")
            return

        ks = sorted({1, 3, 5, 10, self._search_results.k})
        summary: dict[int, float] = {}
        per_question: dict[str, dict[int, float]] = {}

        for k in ks:
            scores = []
            for question_id in reference:
                score = self.recall_at_k(question_id, k)
                scores.append(score)
                per_question.setdefault(question_id, {})[k] = score

            summary[k] = sum(scores) / len(scores) if scores else 0.0

        print("\nRecall Results")
        print("========================================")
        print(f"Questions evaluated: {len(reference)}")

        for k in ks:
            value = summary.get(k, 0.0)
            percentage = value * 100
            print(f"Recall@{k}: {value:.2f} ({percentage:.1f}%)")

        output = {f"recall@{k}": round(summary.get(k, 0.0), 2) for k in ks}
        print(output)
