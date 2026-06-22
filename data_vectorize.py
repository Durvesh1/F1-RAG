from langchain_community.retrievers import BM25Retriever

from data_loader import load_data, load_f1_data
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

from vectors_service.base_vector_store import BaseVectorStore


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


def vectorize_data_from_chunks(vector_store:BaseVectorStore ,chunks):
    vector_store.add_documents(chunks)
    return vector_store


