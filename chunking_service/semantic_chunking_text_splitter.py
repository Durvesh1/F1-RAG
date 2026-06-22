from langchain_experimental.text_splitter import (
    SemanticChunker
)

from chunking_service.base_chunking_service import BaseChunkingService


class SemanticChunkingStrategy(BaseChunkingService):

    def __init__(self,embeddings):
        self.splitter = SemanticChunker(
            embeddings
        )

    def split_documents(
        self,
        documents
    ):
        return self.splitter.split_documents(
            documents
        )