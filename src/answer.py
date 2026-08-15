from src.models import (MinimalSearchResults, StudentSearchResults,
                        ChunkData, MinimalAnswer)
from src.models import StudentSearchResultsAndAnswer as SSRAA
from tqdm import tqdm  # type: ignore
from src.errors import AnswerError
from pydantic import BaseModel
from src.config import Config
import dspy  # type: ignore
from pathlib import Path
from typing import Any
import json


class QA(dspy.Signature):
    context = dspy.InputField(desc="retieved sources used to answer question")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="""
    Answer using only the provided context

    Requirements:
    - Grounded in sources (no hallucinations)
    - Clear and self-contained
    - Directly answers the question
    - Include source references when possible

    If context is insufficient:
    say 'not enough information in the provided context'""")


class Answer(BaseModel):
    results_path: str
    save_directory: str

    def model_post_init(self, _: Any) -> None:
        self._results: StudentSearchResults = self.load_search_results(
            self.results_path)
        self._chunks: list[ChunkData] = self.load_chunks()
        lm = dspy.LM(Config.MODEL, api_base=Config.API_BASE)
        dspy.configure(lm=lm)
        self._predict = dspy.Predict(QA)

    @staticmethod
    def load_search_results(path: str) -> StudentSearchResults:
        try:
            with open(path, encoding='utf-8') as f:
                return StudentSearchResults(**json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            raise AnswerError from e

    @staticmethod
    def load_chunks() -> list[ChunkData]:
        try:
            with open(Config.CHUNKS_PATH, encoding='utf-8') as f:
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
                with open(path, encoding='utf-8') as f:
                    content = f.read()
                    context += f"{content[max(0, first):max(0, last)]}\n"
                    context += "---\n\n"
            except OSError as e:
                raise AnswerError from e
        return context

    def answer(self) -> None:
        answer: list[MinimalAnswer] = []
        for result in tqdm(self._results.search_results, desc="answering"):
            context = self.create_context(result)

            ret = self._predict(context=context, question=result.question_str)

            answer.append(
                MinimalAnswer(**result.model_dump(), answer=ret.answer))
        self.save(SSRAA(search_results=answer, k=self._results.k))

    def save(self, answer: SSRAA) -> None:
        try:
            file_str = f"{self.save_directory}/{Path(self.results_path).name}"
            file = Path(file_str)
            file.parent.mkdir(exist_ok=True, parents=True)
            with open(file, 'w', encoding='utf-8') as f:
                json.dump(answer.model_dump(), f, indent=4)
        except OSError as e:
            raise AnswerError from e
