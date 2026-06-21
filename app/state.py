from typing import TypedDict

class State(TypedDict, total=False):
    question: str
    documents: list[dict]
    answer: str
    retries: int