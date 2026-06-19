from data_loader import load_f1_data
from llm_services.llm_ollama_service import LLM_Ollama_Service
from agents.retriever_agent import RetrieverAgent
from data_vectorize import vectorize_f1_data
from agents.rag_agent import RagAgent
from vectors_service.faiss_vector_store import FaissVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings

# vector_store = vectorize_data("")
# retriever_agent = get_retriever_agent(vector_store)
# query = "Tell me more about the Aircraft operator report"
# top_chunks = retriever_agent.get_chunks(query)
#
# print(top_chunks)


llm_service = LLM_Ollama_Service(model="llama3.2:1b", temperature=0)


data = load_f1_data()
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vector_store_type = FaissVectorStore(embedding_model=embedding_model)
vector_store = vectorize_f1_data(vector_store_type, data)
retriever_agent = RetrieverAgent(vector_store, llm_service, data)
rag = RagAgent(llm_service)

while True:

    # query = "What are the rules regarding two wheelers?"
    query = input("Please enter a query: ")
    top_chunks = retriever_agent.get_chunks(query)
    response = rag.get_response(query, top_chunks)

    print(response)