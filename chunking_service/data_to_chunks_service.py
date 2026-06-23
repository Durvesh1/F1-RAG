from chunking_service.base_chunking_service import BaseChunkingService


# def data_to_chunks(data, chunking_strategy: BaseChunkingService):
#     return chunking_strategy.split_documents(data)


import hashlib


def data_to_chunks(data, chunking_strategy: BaseChunkingService):

    chunks = chunking_strategy.split_documents(
        data
    )

    for idx, chunk in enumerate(chunks):

        chunk.metadata["chunk_index"] = idx

        chunk.metadata["chunk_id"] = hashlib.md5(
            chunk.page_content.encode("utf-8")
        ).hexdigest()

    return chunks