import sys

from agents_setup import setup_agents
from service.db_setup_service import setup_db_services

retriever_agent, rag_agent, embedding_model = setup_agents()

username = input("Please enter your username email Id: ")

conversation_service, user, db = setup_db_services(embedding_model, username)

conversation_id = conversation_service.create_conversation(user_id=user.id,title="F1 Chat")

query = ""

while True:
    query = input("Please enter a query: ")
    if query.lower() == "exit":
        db.close()
        sys.exit()
    conversation_service.add_user_message(
        conversation_id,
        query
    )
    top_chunks = retriever_agent.retrieve(query)
    response = rag_agent.get_response(query, top_chunks)
    print(response)
    conversation_service.add_assistant_message(
        conversation_id,
        response,
        [
            doc.chunk_id
            for doc in top_chunks
        ]
    )
