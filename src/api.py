from pydantic import BaseModel
from fastapi import FastAPI
from src.rag import Rag
from typing import Any
import json


class SingleQuery(BaseModel):
    question: str


app = FastAPI()
rag = Rag()


@app.post("/answer")
def answer(question: SingleQuery) -> Any:
    return json.loads(rag.answer(question.question))


@app.post("/search")
def search(question: SingleQuery) -> Any:
    return json.loads(rag.search(question.question))
