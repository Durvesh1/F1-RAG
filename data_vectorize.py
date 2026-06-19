from langchain_community.retrievers import BM25Retriever

from data_loader import load_data, load_f1_data
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


def vectorize_data(path):
    data = load_data(path)

    if not data:
        raise ValueError("No data")

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    text_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_amount=90,
        breakpoint_threshold_type="percentile",
    )

    docs = text_splitter.create_documents([data])

    vector_store = FAISS.from_documents(docs, embedding_model)

    return vector_store

def retrieve_chunks_from_vector(vector_store):
    chunks_retriever = vector_store.as_retriever(kwargs = {"k":3})
    query = "Aircraft operator report"
    chunks = chunks_retriever.invoke(query)

    return chunks


def vectorize_f1_data(data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunks = splitter.split_documents(data)

    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = FAISS.from_documents(chunks, embedding_model)

    return vector_store

def bm25_retrieve(vector_store):
    bm25_retriever = BM25Retriever.from_documents(vector_store)
    bm25_retriever.k = 5

    results = bm25_retriever.invoke("How many team members are allowed in the signalling area?")

    print(results)

# data = load_f1_data()
# bm25_retrieve(data)