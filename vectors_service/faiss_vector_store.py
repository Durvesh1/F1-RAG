from langchain_community.vectorstores import FAISS

from vectors_service.base_vector_store import BaseVectorStore


class FaissVectorStore(BaseVectorStore):

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.store = None

    def add_documents(self, documents):
        if self.store is None:
            self.store = FAISS.from_documents(
                documents,
                self.embedding_model
            )
        else:
            self.store.add_documents(documents)

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