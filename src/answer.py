from llm_sdk import Small_LLM_Model  # type: ignore
from src.models import MinimalSearchResults, StudentSearchResults, ChunkData
from src.errors import AnswerError
from pydantic import BaseModel
from src.config import Config
from pathlib import Path
from typing import Any
import json


class Answer(BaseModel):
    results_path: str
    save_directory: str

    def model_post_init(self, _: Any) -> None:
        self._results: StudentSearchResults = self.open()
        self._chunks: list[ChunkData] = self.load()
        self._llm = Small_LLM_Model(model_name="Qwen/Qwen3-0.6B")

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
        context = ""
        for i, source in enumerate(result.retrieved_sources[:3], start=1):
            path = source.file_path
            first = source.first_character_index
            last = source.last_character_index
            try:
                with open(path) as f:
                    content = f.read()
                    context += f"source {i}: {Path(path).as_posix()}\n"
                    context += f"{content[first:last]}\n\n"
            except OSError as e:
                raise AnswerError from e
        return context

    def answer(self) -> None:
        for result in self._results.search_results:
            context = self.create_context(result)
            context += f"""
You are a Retrival Augmented Generator
According to the corpus of texts given before answer this question
{result.question_str}
"""
            # self._llm.encode()
