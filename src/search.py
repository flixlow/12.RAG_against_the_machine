from src.models import UnansweredQuestion, MinimalSearchResults
from src.models import MinimalSource, StudentSearchResults
from tqdm import tqdm  # type: ignore
from src.errors import SearchError
from chromadb import Collection
from pydantic import BaseModel
from src.config import Config
import bm25s  # type: ignore
from pathlib import Path
from bm25s import BM25
from typing import Any
import chromadb
import json


class Search(BaseModel):
    rag_questions: list[UnansweredQuestion]
    k: int
    save_dir: str
    file: str
    hybrid: bool

    def model_post_init(self, _: Any) -> None:
        self._results: list[MinimalSearchResults] = []
        try:
            self._retriever: BM25 = bm25s.BM25.load(
                Config.BM25_PATH, load_corpus=True)
        except OSError as e:
            raise SearchError("BM25 index not found or corrupted. "
                              "Please build the index first: make index"
                              ) from e
        try:
            with open(Config.CHUNKS_PATH) as f:
                self._chunks: list[dict[str, Any]] = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise SearchError from e
        try:
            if self.hybrid:
                client = chromadb.PersistentClient(Config.CHROMA_PATH)
                self._collection: Collection = client.get_collection("coll")
        except ValueError:
            raise SearchError("collection from embedding does not exist, "
                              "please run index --embedding first.")

    def search_dataset(self) -> None:
        for query in tqdm(self.rag_questions):
            search_result = self.search(query)
            self._results.append(search_result)

        self.save()

    def search_chroma(self, query: str) -> list[int]:
        return []

    def hybrid_retrieve(self, bm25_results: list[int],
                        chroma_results: list[int]) -> None:
        pass

    def search_bm25(self, query: UnansweredQuestion) -> list[int]:
        query_tokens = bm25s.tokenize(query.question)
        docs = self._retriever.retrieve(query_tokens, k=self.k,
                                        sorted=True, return_as="documents")
        return [doc['id'] for doc in docs[0]]

    def search(self, query: UnansweredQuestion) -> MinimalSearchResults:
        sources: list[MinimalSource] = []

        bm25_results = self.search_bm25(query)
        if self.hybrid:
            chroma_results = self.search_chroma(query.question)
            self.hybrid_retrieve(bm25_results, chroma_results)
        # metadata = self._chunks[id]["metadata"]
        # sources.append(MinimalSource(**metadata))

        return MinimalSearchResults(question_id=query.question_id,
                                    question_str=query.question,
                                    retrieved_sources=sources)

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
