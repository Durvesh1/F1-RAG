from collections import defaultdict

from retriever.retrieved_document import RetrievedDocument


class RRFFusion:

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self,
        retrieval_results: list[list[RetrievedDocument]]
    ) -> list[RetrievedDocument]:

        scores = defaultdict(float)
        doc_lookup = {}

        for result_list in retrieval_results:

            for rank, doc in enumerate(result_list):

                doc_id = hash(doc.content)

                doc_lookup[doc_id] = doc

                scores[doc_id] += (
                    1 / (self.k + rank + 1)
                )

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        fused_docs = []

        for doc_id, score in ranked:

            doc = doc_lookup[doc_id]

            doc.score = score

            fused_docs.append(doc)

        return fused_docs