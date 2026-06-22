
from vectors_service.base_vector_store import BaseVectorStore

def vectorize_data_from_chunks(vector_store:BaseVectorStore ,chunks):
    vector_store.add_documents(chunks)
    return vector_store


