import uuid

from sqlalchemy import (
    String,
    DateTime,
    ForeignKey
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column
)

from datetime import datetime

from models.base import Base


class Conversation(Base):

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id"),
        nullable=False
    )

    title: Mapped[str] = mapped_column(
        String
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.now()
    )