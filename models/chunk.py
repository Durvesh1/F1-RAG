import uuid

from datetime import datetime

from sqlalchemy import (
    String,
    Text,
    DateTime,
    ForeignKey
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column
)

from pgvector.sqlalchemy import Vector

from models.base import Base


class Chunk(Base):

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id"),
        nullable=False
    )

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id"),
        nullable=False
    )

    chunk_id: Mapped[str] = mapped_column(
        String,
        unique=True,
        nullable=False
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(768)
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now()
    )