from src.models import UnansweredQuestion, MinimalSearchResults
from src.models import MinimalSource, StudentSearchResults
from tqdm import tqdm  # type: ignore
from collections import defaultdict
from src.errors import SearchError
from chromadb import Collection
from pydantic import BaseModel
from src.config import Config
import bm25s  # type: ignore
import dspy  # type: ignore
from pathlib import Path
from bm25s import BM25
from typing import Any
import chromadb
import json


class QA(dspy.Signature):
    query: str = dspy.InputField()
    expanded_query: str = dspy.OutputField(
        desc="Space-separated keywords for BM25 retrieval, in [LANGUE]. "
        "Include the key terms from the original query, then add relevant "
        "synonyms, related terms, and acronym expansions. "
        "Keep multi-word named entities or technical terms intact. "
        "No full sentences, no stopwords, no punctuation, no special symbols. "
        "Aim for 8-15 keywords total, avoid near-duplicate variants.")


class Search(BaseModel):
    rag_questions: list[UnansweredQuestion]
    k: int
    save_dir: str
    file: str
    hybrid: bool
    expansion: bool

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
            with open(Config.CHUNKS_PATH, encoding='utf-8') as f:
                self._chunks: list[dict[str, Any]] = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise SearchError from e
        try:
            if self.hybrid:
                client = chromadb.PersistentClient(Config.CHROMA_PATH)
                self._collection: Collection = client.get_collection(
                    "collection")
        except chromadb.errors.NotFoundError:
            raise SearchError("collection from embedding does not exist, "
                              "please run index --embedding first.")
        if self.expansion:
            lm = dspy.LM(Config.MODEL, api_base=Config.API_BASE)
            dspy.configure(lm=lm)
            self._predict = dspy.Predict(QA)

    def expand_query(self, query: str) -> str:
        # print("before", query)
        result = self._predict(query=query)
        # print("after", result.expanded_query)
        return result.expanded_query

    def reciprocal_rank_fusion(self, bm25_results: list[int],
                               chroma_results: list[int]) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        k = 60

        for rank, id in enumerate(bm25_results):
            scores[id] += 1 / (k + rank)
        for rank, id in enumerate(chroma_results):
            scores[id] += 1 / (k + rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [id for id, _ in ranked[:self.k]]

    def search_chroma(self, query: str, k: int) -> list[int]:
        results = self._collection.query(query_texts=[query], n_results=k)
        return [int(id) for id in results['ids'][0]]

    def search_bm25(self, query: str, k: int) -> list[int]:
        query_tokens = bm25s.tokenize(query)
        docs = self._retriever.retrieve(query_tokens, k=k,
                                        sorted=True, return_as="documents")
        return [doc['id'] for doc in docs[0]]

    def search(self, unanswerer: UnansweredQuestion) -> MinimalSearchResults:
        ids: list[int] = []
        sources: list[MinimalSource] = []

        query = self.expand_query(unanswerer.question) if self.expansion \
            else unanswerer.question

        if not self.hybrid:
            ids = self.search_bm25(query, self.k)
        else:
            bm25_results = self.search_bm25(query, 20)
            chroma_results = self.search_chroma(query, 20)
            ids = self.reciprocal_rank_fusion(bm25_results, chroma_results)

        for id in ids:
            metadata = self._chunks[id]["metadata"]
            sources.append(MinimalSource(**metadata))

        return MinimalSearchResults(question_id=unanswerer.question_id,
                                    question_str=unanswerer.question,
                                    retrieved_sources=sources)

    def search_dataset(self) -> None:
        for query in tqdm(self.rag_questions):
            search_result = self.search(query)
            self._results.append(search_result)

        self.save()

    def save(self) -> None:
        output = StudentSearchResults(search_results=self._results, k=self.k)

        try:
            file_str = f"{self.save_dir}/{self.file}"
            file = Path(file_str)
            file.parent.mkdir(exist_ok=True, parents=True)
            with open(file, 'w', encoding='utf-8') as f:
                json.dump(output.model_dump(), f, ensure_ascii=False, indent=4)

        except (OSError, json.JSONDecodeError) as e:
            raise SearchError from e

        print(f"Saved student_search_results to {file_str}")
