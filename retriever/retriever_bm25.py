from langchain_community.retrievers import BM25Retriever

from retriever.base_retriever import BaseRetrieverService
from retriever.retrieved_document import RetrievedDocument


class BM25ChunkRetriever(BaseRetrieverService):

    def __init__(self, chunks):

        self.retriever = BM25Retriever.from_documents(
            chunks
        )

    def retrieve(
        self,
        query: str,
        k: int = 10
    ) -> list[RetrievedDocument]:

        self.retriever.k = k

        docs = self.retriever.invoke(query)

        return [
            RetrievedDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                source="bm25"
            )
            for doc in docs
        ]