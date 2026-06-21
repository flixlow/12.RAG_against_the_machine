# from llm_sdk import Small_LLM_Model  # type: ignore
from src.models import StudentSearchResults, MinimalAnswer, ChunkData
from src.models import StudentSearchResultsAndAnswer, MinimalSearchResults
from pydantic import BaseModel
from typing import Any
from src.config import Config
from src.errors import AnswerError
import json


class Answer(BaseModel):
    results_path: str
    save_directory: str

    def model_post_init(self, _: Any) -> None:
        self._results: StudentSearchResults = self.open()
        self._chunks: list[ChunkData] = self.load()
        # self._llm = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")

    def open(self) -> StudentSearchResults:
        try:
            with open(self.results_path) as f:
                return StudentSearchResults(**json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            raise AnswerError from e

    def load(self) -> list[ChunkData]:
        try:
            with open(Config.CHUNKS) as f:
                return [ChunkData(**c) for c in json.load(f)]
        except (OSError, json.JSONDecodeError) as e:
            raise AnswerError from e

    def create_context(self, result: MinimalSearchResults) -> str:
        pass
        # context = ""
        # for chunk in result.retrieved_sources:
            # id = chunk['id']
            # self._chunks

    def answer(self) -> None:
        for result in self._results.search_results:
            pass
            # context = self.create_context(result)
            # self._llm.encode()
