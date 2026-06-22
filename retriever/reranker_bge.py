from sentence_transformers import CrossEncoder

from retriever.retrieved_document import RetrievedDocument


class BGEReranker:

    def __init__(
        self,
        model_name="BAAI/bge-reranker-base"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        docs: list[RetrievedDocument],
        top_k: int = 5
    ):

        pairs = [
            (query, doc.content)
            for doc in docs
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            doc
            for doc, _ in ranked[:top_k]
        ]