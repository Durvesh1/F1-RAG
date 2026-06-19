from langchain_core.documents import Document
from langchain_postgres import PGVector

from vectors_service.base_vector_store import BaseVectorStore


class PGVectorStore(BaseVectorStore):

    def __init__(
        self,
        embedding_model,
        connection_string,
        collection_name
    ):
        self.store = PGVector(
            embeddings=embedding_model,
            connection=connection_string,
            collection_name=collection_name
        )

    def add_documents(self, document: list[Document]):
        self.store.add_documents(document)

    def similarity_search(self, query, k=5):
        return self.store.similarity_search(
            query,
            k=k
        )

    def similarity_search_with_score(
        self,
        query,
        k=5
    ):
        return self.store.similarity_search_with_score(
            query,
            k=k
        )

    def as_retriever(self, **kwargs):
        return self.store.as_retriever(**kwargs)