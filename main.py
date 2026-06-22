from agents.rag_agent import RagAgent
from agents.retriever_agent import RetrieverAgent
from chunking_service.data_to_chunks_service import data_to_chunks
from chunking_service.recursive_character_text_splitter import RecursiveChunkingStrategy
from data_loader import load_f1_data
from llm_services.llm_ollama_service import LLM_Ollama_Service
from data_vectorize import vectorize_data_from_chunks
from retriever.fusion_rrf import RRFFusion
from retriever.reranker_bge import BGEReranker
from retriever.retriever_bm25 import BM25ChunkRetriever
from retriever.retriever_similarity import VectorSimilarityRetriever
from vectors_service.faiss_vector_store import FaissVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings


llm_service = LLM_Ollama_Service(model="llama3.2:1b", temperature=0)


data = load_f1_data()
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vector_store_type = FaissVectorStore(embedding_model=embedding_model)
chunking_strategy = RecursiveChunkingStrategy(chunk_size=1000, chunk_overlap=200)
chunks = data_to_chunks(data, chunking_strategy)
vector_store = vectorize_data_from_chunks(vector_store_type, chunks)
similarity_search_retriever = VectorSimilarityRetriever(vector_store)
bm25_retriever = BM25ChunkRetriever(chunks)
fusion = RRFFusion()
reranker = BGEReranker()
retriever_agent = RetrieverAgent(retrievers=[similarity_search_retriever, bm25_retriever],
    fusion_strategy=fusion,
    reranker=reranker
)
rag = RagAgent(llm_service)

while True:

    query = input("Please enter a query: ")
    top_chunks = retriever_agent.retrieve(query)
    response = rag.get_response(query, top_chunks)

    print(response)
