from abc import ABC, abstractmethod

from retriever.retrieved_document import RetrievedDocument


class BaseRetrieverService(ABC):

    @abstractmethod
    def retrieve(self,query: str,k: int) -> list[RetrievedDocument]:
        pass