from dataclasses import dataclass
from typing import Any


@dataclass
class EvalCase:

    question: str

    expected_answer: str

    expected_chunk_ids: list[str]

    metadata: dict[str, Any]  # optional (type, difficulty, source chunk, etc.)