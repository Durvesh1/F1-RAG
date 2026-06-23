from retriever.base_retriever import BaseRetrieverService
from retriever.retrieved_document import RetrievedDocument


class VectorSimilarityRetriever(BaseRetrieverService):

    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int):
        docs = self.vector_store.similarity_search_with_score(
            query,
            k=k
        )

        return [
            RetrievedDocument(
                content=d.page_content,
                metadata=d.metadata,
                source="vector",
                score = float(score)
            )
            for d,score in docs
        ]