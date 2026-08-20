from src.models import (MinimalSearchResults, StudentSearchResults,
                        ChunkData, MinimalAnswer)
from src.models import StudentSearchResultsAndAnswer as SSRAA
from pydantic import BaseModel, Field
from src.errors import AnswerError
from src.config import Config
from pathlib import Path
from typing import Any
from tqdm import tqdm
import dspy
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
    cache_flag: bool = Field(default=True)

    def model_post_init(self, _: Any) -> None:
        self._chunks: list[ChunkData] = Answer._load_chunks()
        lm = dspy.LM(Config.MODEL, api_base=Config.API_BASE)
        dspy.configure(lm=lm)
        self._predict = dspy.Predict(QA)
        if self.cache_flag is True:
            self._cache: dict[str, str] = Answer._load_cache()

    @staticmethod
    def _load_cache() -> dict[str, str]:
        try:
            with open(Path(Config.CACHE)) as f:
                return dict(json.load(f))
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _load_search_results(path: str) -> StudentSearchResults:
        try:
            with open(path, encoding='utf-8') as f:
                return StudentSearchResults(**json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            raise AnswerError("can't load search_results.") from e

    @staticmethod
    def _load_chunks() -> list[ChunkData]:
        try:
            with open(Config.CHUNKS_PATH, encoding='utf-8') as f:
                return [ChunkData(**c) for c in json.load(f)]
        except (OSError, json.JSONDecodeError) as e:
            raise AnswerError("can't load chunks.") from e

    @staticmethod
    def create_context(result: MinimalSearchResults) -> str:
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

    def save_cache(self) -> None:
        try:
            with open(Config.CACHE, 'w') as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=4)
        except (OSError, json.JSONDecodeError):
            raise AnswerError("Error occurs when saving cache.")

    def save(self, answers: SSRAA) -> None:
        try:
            file_str = f"{self.save_directory}/{Path(self.results_path).name}"
            file = Path(file_str)
            file.parent.mkdir(exist_ok=True, parents=True)
            with open(file, 'w', encoding='utf-8') as f:
                json.dump(answers.model_dump(), f, indent=4)
        except OSError as e:
            raise AnswerError from e

    def answer(self, result: MinimalSearchResults) -> MinimalAnswer:
        if self.cache_flag is True:
            try:
                cache_answer = self._cache[result.question]
                return MinimalAnswer(
                    **result.model_dump(), answer=cache_answer)
            except KeyError:
                pass
        c = Answer.create_context(result)
        new = self._predict(context=c, question=result.question)
        if self.cache_flag:
            self._cache[result.question] = new.answer
            self.save_cache()
        return MinimalAnswer(**result.model_dump(), answer=new.answer)

    def answer_dataset(self) -> None:
        results = Answer._load_search_results(self.results_path)
        answers: list[MinimalAnswer] = []

        for result in tqdm(results.search_results, desc="answering"):
            formated_answer = self.answer(result)

            answers.append(formated_answer)
        self.save(SSRAA(search_results=answers, k=results.k))
