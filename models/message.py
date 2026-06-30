import uuid

from sqlalchemy import (
    String,
    Text,
    DateTime,
    ForeignKey
)

from sqlalchemy.dialects.postgresql import JSONB

from sqlalchemy.orm import (
    Mapped,
    mapped_column
)

from datetime import datetime

from models.base import Base


class Message(Base):

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id"),
        nullable=False
    )

    role: Mapped[str] = mapped_column(
        String,
        nullable=False
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now()
    )