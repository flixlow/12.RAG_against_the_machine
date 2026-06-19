from src.models import UnansweredQuestion
from src.errors import SearchError
from pydantic import BaseModel
import bm25s  # type: ignore
from typing import Any


class Search(BaseModel):
    rag_questions: list[UnansweredQuestion]
    k: int
    save_dir: str

    def model_post_init(self, _: Any) -> None:
        try:
            self._retriever = bm25s.BM25.load(
                "data/processed/bm25_index",
                load_corpus=True
                )
        except OSError as e:
            raise SearchError("BM25 index not found or corrupted. "
                              "Please build the index first: make index"
                              ) from e

    def search_dataset(self) -> None:
        for query in self.rag_questions:
            self.search(query.question)

    def search(self, query: str) -> None:
        query_tokens = bm25s.tokenize(query)

        docs, scores = self._retriever.retrieve(
            query_tokens, k=self.k, sorted=True)
        print(f"scores: {scores[0][0]:.2f}: {docs[0][0]}")

    def answer(self) -> None:
        pass
