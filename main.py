from data_loader import load_data
from retriever_agent import get_retriever_agent
from data_vectorize import vectorize_data, vectorize_f1_data
from rag_agent import RagAgent

# data = load_data()
#
# print(data)


# vector_store = vectorize_data("")
# retriever_agent = get_retriever_agent(vector_store)
# query = "Tell me more about the Aircraft operator report"
# top_chunks = retriever_agent.get_chunks(query)
#
# print(top_chunks)


vector_store = vectorize_f1_data()
retriever_agent = get_retriever_agent(vector_store)

while True:

    # query = "What are the rules regarding two wheelers?"
    query = input("Please enter a query: ")
    top_chunks = retriever_agent.get_chunks(query)

    rag = RagAgent(query, top_chunks)

    response = rag.get_response()

    print(response)