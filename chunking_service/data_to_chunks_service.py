from chunking_service.base_chunking_service import BaseChunkingService


def data_to_chunks(data, chunking_strategy: BaseChunkingService):
    return chunking_strategy.split_documents(data)
