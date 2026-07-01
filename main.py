import sys

from agents_setup import setup_agents
from guardrails.input_guardrails.prompt_injection_guard import PromptInjectionGuard
from guardrails.input_guardrails.query_normaliser import QueryNormalizer
from guardrails.input_guardrails.query_validator import QueryValidator
from service.db_setup_service import setup_db_services

retriever_agent, rag_agent, embedding_model = setup_agents()

username = input("Please enter your username email Id: ")

conversation_service, user, db = setup_db_services(embedding_model, username)

conversation_id = conversation_service.get_or_create_conversation(user_id=user.id,title="F1 Chat")

history = conversation_service.get_langchain_messages(conversation_id,limit=20)

query_validator = QueryValidator()
query_normaliser = QueryNormalizer()
prompt_injection_check = PromptInjectionGuard()

query = ""

while True:
    query = input("Please enter a query: ")
    if query.lower() == "exit":
        db.close()
        sys.exit()
    try:
        query = query_validator.run(query)
        query = query_normaliser.run(query)
        query = prompt_injection_check.run(query)
    except Exception as error:
        print("Error: ", error)
        continue
    conversation_service.add_user_message(
        conversation_id,
        query
    )
    top_chunks = retriever_agent.retrieve(query)
    response = rag_agent.get_response(query, top_chunks, history)
    print(response)
    conversation_service.add_assistant_message(
        conversation_id,
        response,
        [
            doc.chunk_id
            for doc in top_chunks
        ]
    )
