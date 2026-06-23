from agents.retriever_agent import RetrieverAgent
from data_loader import load_f1_data
from evaluation.eval_class import EvalCase


from agents.rag_agent import RagAgent
from agents.retriever_agent import RetrieverAgent
from chunking_service.data_to_chunks_service import data_to_chunks
from chunking_service.recursive_character_text_splitter import RecursiveChunkingStrategy
from data_loader import load_f1_data
from llm_services.llm_ollama_service import LLM_Ollama_Service
from data_vectorize import vectorize_data_from_chunks
from retriever.fusion_rrf import RRFFusion
from retriever.reranker_bge import BGEReranker
from retriever.retriever_bm25 import BM25ChunkRetriever
from retriever.retriever_similarity import VectorSimilarityRetriever
from vectors_service.faiss_vector_store import FaissVectorStore
from langchain_ollama.embeddings import OllamaEmbeddings


class RetrievalEvaluator:

    @staticmethod
    def recall_at_k(retrieved_docs,expected_chunk_ids):

        retrieved_ids = {
            doc.chunk_id
            for doc in retrieved_docs
        }

        expected_ids = set(
            expected_chunk_ids
        )

        hits = len(
            retrieved_ids.intersection(
                expected_ids
            )
        )

        return hits / len(expected_ids)

    @staticmethod
    def mrr(retrieved_docs,expected_chunk_ids):

        expected_ids = set(
            expected_chunk_ids
        )

        for rank, doc in enumerate(
                retrieved_docs,
                start=1
        ):

            if doc.chunk_id in expected_ids:
                return 1.0 / rank

        return 0.0


    def run(self,retriever_agent,eval_cases,retrieval_k=10,rerank_k=5):

        recalls = []
        mrr_scores = []

        for case in eval_cases:
            docs = retriever_agent.retrieve(
                case.question,
                retrieval_k=retrieval_k,
                rerank_k=rerank_k
            )

            recalls.append(
                self.recall_at_k(
                    docs,
                    case.expected_chunk_ids
                )
            )

            mrr_scores.append(
                self.mrr(
                    docs,
                    case.expected_chunk_ids
                )
            )

        return {

            "Recall":
                sum(recalls)
                / len(recalls),

            "MRR":
                sum(mrr_scores)
                / len(mrr_scores)
        }


from typing import List
import json

def load_eval_dataset(file_path: str) -> List[EvalCase]:

    with open(file_path, "r", encoding="utf-8") as f:

        data = json.load(f)

    return [
        EvalCase(**item)
        for item in data
    ]

eval_cases = load_eval_dataset("eval_dataset.json")

retrieval_evaluator = RetrievalEvaluator()

data = load_f1_data()
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vector_store_type = FaissVectorStore(embedding_model=embedding_model)
chunking_strategy = RecursiveChunkingStrategy(chunk_size=1000, chunk_overlap=200)
chunks = data_to_chunks(data, chunking_strategy)
vector_store = vectorize_data_from_chunks(vector_store_type, chunks)
similarity_search_retriever = VectorSimilarityRetriever(vector_store)
bm25_retriever = BM25ChunkRetriever(chunks)
fusion = RRFFusion()
# reranker = BGEReranker()
reranker = None
retriever_agent = RetrieverAgent(retrievers=[similarity_search_retriever, bm25_retriever],
    fusion_strategy=fusion,
    reranker=reranker)

res = retrieval_evaluator.run(
    retriever_agent,
    eval_cases,
    retrieval_k=10
)

print(res)