import sys

from agents_setup import setup_agents

retriever_agent, rag_agent = setup_agents()


query = ""

while True:
    query = input("Please enter a query: ")
    if query.lower() == "exit":
        sys.exit()
    top_chunks = retriever_agent.retrieve(query)
    response = rag_agent.get_response(query, top_chunks)
    print(response)
