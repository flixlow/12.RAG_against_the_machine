from pydantic import BaseModel, Field
from src.errors import SearchError
import bm25s  # type: ignore


class SingleQuery(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(gt=0)

    def search(self) -> None:
        query_tokens = bm25s.tokenize(self.query)

        try:
            retriever = bm25s.BM25.load("data/processed/bm25_index",
                                        load_corpus=True)
        except (FileNotFoundError, PermissionError) as e:
            raise SearchError("BM25 index not found or corrupted. "
                              "Please build the index first: make index"
                              ) from e

        docs, scores = retriever.retrieve(query_tokens, k=self.k, sorted=True)
        print(f"scores: {scores[0][0]:.2f}: {docs[0][0]}")

    def answer(self) -> None:
        pass
