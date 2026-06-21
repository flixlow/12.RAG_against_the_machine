from src.models import (UnansweredQuestion, MinimalSearchResults,
                        MinimalSource, StudentSearchResults)
from src.errors import SearchError
from pydantic import BaseModel
import bm25s  # type: ignore
from pathlib import Path
from bm25s import BM25
from typing import Any
import json


class Search(BaseModel):
    rag_questions: list[UnansweredQuestion]
    k: int
    save_dir: str
    file: str

    def model_post_init(self, _: Any) -> None:
        self._results: list[MinimalSearchResults] = []

        try:
            self._retriever: BM25 = bm25s.BM25.load(
                "data/processed/bm25_index",
                load_corpus=True
                )
        except OSError as e:
            raise SearchError("BM25 index not found or corrupted. "
                              "Please build the index first: make index"
                              ) from e

        try:
            with open("data/processed/chunks/splitted.json") as f:
                self._splitted: list[dict[str, Any]] = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise SearchError from e

    def search_dataset(self) -> None:
        for query in self.rag_questions:
            search_result = self.search(query)
            self._results.append(search_result)

        self.save()

    def search(self, query: UnansweredQuestion) -> MinimalSearchResults:
        query_tokens = bm25s.tokenize(query.question)
        sources: list[MinimalSource] = []

        docs = self._retriever.retrieve(
            query_tokens,
            k=self.k,
            sorted=True,
            return_as="documents"
            )

        for doc in docs[0]:
            id = doc["id"]
            metadata = self._splitted[id]["metadata"]
            sources.append(MinimalSource(**metadata))

        return MinimalSearchResults(
            question_id=query.question_id,
            question_str=query.question,
            retrieved_sources=sources
            )

    def save(self) -> None:
        output = StudentSearchResults(search_results=self._results, k=self.k)

        try:
            file_str = f"{self.save_dir}/{self.file}"
            file = Path(file_str)
            file.parent.mkdir(exist_ok=True, parents=True)
            with open(file, 'w') as f:
                json.dump(output.model_dump(), f, ensure_ascii=False, indent=4)

        except (OSError, json.JSONDecodeError) as e:
            raise SearchError from e

        print(f"Saved student_search_results to {file_str}")

    def answer(self) -> None:
        pass
