class NewRetrieverAgent:

    def __init__( self, retrievers, fusion_strategy, reranker):

        self.retrievers = retrievers

        self.fusion_strategy = fusion_strategy

        self.reranker = reranker

    def retrieve(self,query: str,retrieval_k: int = 10,rerank_k: int = 5):

        retrieval_results = []

        for retriever in self.retrievers:

            docs = retriever.retrieve(
                query,
                retrieval_k
            )

            retrieval_results.append(docs)

        fused_docs = self.fusion_strategy.fuse(
            retrieval_results
        )

        final_docs = self.reranker.rerank(
            query,
            fused_docs,
            top_k=rerank_k
        )

        return final_docs