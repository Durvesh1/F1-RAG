import uuid

from langchain_core.messages import (HumanMessage,AIMessage)

from models import (Conversation,Message,ConversationSummary)


class ConversationService:

    def __init__(self,db_session,embedding_model):

        self.db = db_session
        self.embedding_model = embedding_model

    # --------------------------------
    # Conversations
    # --------------------------------

    def create_conversation(self,user_id,title):

        conversation = Conversation(id=str(uuid.uuid4()),user_id=user_id,title=title)

        self.db.add(conversation)
        self.db.commit()

        return conversation.id

    # --------------------------------
    # Messages
    # --------------------------------

    def add_user_message(self,conversation_id,content):

        message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role="user",
            content=content
        )

        self.db.add(message)
        self.db.commit()

    def add_assistant_message(self,conversation_id,content,source_chunk_ids=None):

        metadata = {
            "source_chunk_ids":
                source_chunk_ids or []
        }

        message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            metadata=metadata
        )

        self.db.add(message)
        self.db.commit()

    # --------------------------------
    # History
    # --------------------------------

    def get_recent_messages(self,conversation_id,limit=20):

        return (
            self.db.query(Message)
            .filter(
                Message.conversation_id
                == conversation_id
            )
            .order_by(
                Message.created_at.desc()
            )
            .limit(limit)
            .all()
        )

    def get_langchain_messages(self,conversation_id,limit=20):

        messages = (
            self.get_recent_messages(
                conversation_id,
                limit
            )
        )

        messages.reverse()

        result = []

        for msg in messages:

            if msg.role == "user":

                result.append(
                    HumanMessage(
                        content=msg.content
                    )
                )

            elif msg.role == "assistant":

                result.append(
                    AIMessage(
                        content=msg.content
                    )
                )

        return result

    # --------------------------------
    # Summary
    # --------------------------------

    def get_summary(self,conversation_id):

        summary = (
            self.db.query(
                ConversationSummary
            )
            .filter(
                ConversationSummary.conversation_id
                == conversation_id
            )
            .first()
        )

        if summary:
            return summary.summary

        return ""

    # --------------------------------
    # Prompt Context Builder
    # --------------------------------

    def build_memory_context(
            self,
            user_id,
            conversation_id,
            query
    ):

        summary = (
            self.get_summary(
                conversation_id
            )
        )

        return {
            "summary": summary
        }

    def get_or_create_conversation(self,user_id: str,title: str):

        conversation = (
            self.db.query(Conversation)
            .filter(
                Conversation.user_id == user_id,
                Conversation.title == title
            )
            .first()
        )

        if conversation:
            return conversation.id

        conversation = Conversation(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=title
        )

        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)

        return conversation.id