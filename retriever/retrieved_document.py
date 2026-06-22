from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievedDocument:
    content: str
    metadata: dict[str, Any]
    source: str
    score: float | None = None