from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from chunking_service.base_chunking_service import BaseChunkingService


class RecursiveChunkingStrategy(BaseChunkingService):

    def __init__(self,chunk_size=1000,chunk_overlap=200):
        self.splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        )

    def split_documents(self, documents):
        return self.splitter.split_documents(
            documents
        )