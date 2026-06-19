from data_loader import load_data
from llm_ollama_service import LLM_Ollama_Service
from retriever_agent import RetrieverAgent
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


llm_service = LLM_Ollama_Service(model="llama3.2:1b", temperature=0)

vector_store = vectorize_f1_data()
retriever_agent = RetrieverAgent(vector_store, llm_service)
rag = RagAgent(llm_service)

while True:

    # query = "What are the rules regarding two wheelers?"
    query = input("Please enter a query: ")
    top_chunks = retriever_agent.get_chunks(query)
    response = rag.get_response(query, top_chunks)

    print(response)