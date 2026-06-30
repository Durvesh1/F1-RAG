import uuid

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

from datetime import datetime

from models.base import Base


class ConversationSummary(Base):

    __tablename__ = "conversation_summaries"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id"),
        unique=True,
        nullable=False
    )

    summary: Mapped[str] = mapped_column(
        Text
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now()
    )