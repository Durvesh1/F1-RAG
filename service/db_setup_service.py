from database_service.db_connection import SessionLocal
from service.conversation_service import ConversationService
from service.user_service import UserService

db = SessionLocal()


def setup_db_services(embedding_model, username):
    conversation_service = ConversationService(db, embedding_model)
    user_service = UserService(db)
    user = user_service.get_or_create_user(username)

    return conversation_service, user, db