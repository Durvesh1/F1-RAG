from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievedDocument:
    content: str
    metadata: dict[str, Any]
    source: str
    score: float | None = None
    rerank_score: float | None = None

    @property
    def chunk_id(self):
        return self.metadata.get(
            "chunk_id"
        )

    @property
    def chunk_index(self):
        return self.metadata.get(
            "chunk_index"
        )