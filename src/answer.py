from src.models import (MinimalSearchResults, StudentSearchResults,
                        ChunkData, AnsweredQuestion)
from src.errors import AnswerError
from pydantic import BaseModel
from src.config import Config
from pathlib import Path
from typing import Any
import dspy  # type: ignore
import json
from tqdm import tqdm


class QA(dspy.Signature):
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()


class Answer(BaseModel):
    results_path: str
    save_directory: str

    def model_post_init(self, _: Any) -> None:
        self._results: StudentSearchResults = self.open()
        self._chunks: list[ChunkData] = self.load()
        self._lm = dspy.LM("ollama/qwen3:0.6b",
                           api_base="http://localhost:11434")
        dspy.configure(lm=self._lm)
        self._predict = dspy.Predict(QA)

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
        context = ""
        for i, source in enumerate(result.retrieved_sources[:3], start=1):
            path = source.file_path
            first = source.first_character_index
            last = source.last_character_index
            try:
                with open(path) as f:
                    content = f.read()
                    context += f"[Source {i}]\n"
                    context += f"Path: {Path(path).as_posix()}\n"
                    context += "Content:\n"
                    context += f"{content[max(0, first):max(0, last)]}\n"
                    context += "---\n\n"
            except OSError as e:
                raise AnswerError from e
        return context

    def answer(self) -> None:
        answered: list[AnsweredQuestion] = []
        for result in tqdm(self._results.search_results):
            context = self.create_context(result)
            ret = self._predict(
                context=context,
                question=result.question_str
            )
            answered.append(AnsweredQuestion(
                question_id=result.question_id,
                question=result.question_str,
                sources=result.retrieved_sources,
                answer=ret.answer
            ))
        self.save(answered)

    def save(self, answered: list[AnsweredQuestion]) -> None:
        try:
            file_str = f"{self.save_directory}/{Path(self.results_path).name}"
            file = Path(file_str)
            file.parent.mkdir(exist_ok=True, parents=True)
            with open(file, 'w') as f:
                json.dump([a.model_dump() for a in answered], f, indent=4)
        except OSError as e:
            raise AnswerError from e
