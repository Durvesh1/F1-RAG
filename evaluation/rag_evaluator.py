import json
from collections import defaultdict

from langchain_community.embeddings import OllamaEmbeddings

from agents.rag_agent import RagAgent
from agents.retriever_agent import RetrieverAgent
from chunking_service.data_to_chunks_service import data_to_chunks
from chunking_service.recursive_character_text_splitter import RecursiveChunkingStrategy
from data_loader import load_f1_data
from data_vectorize import vectorize_data_from_chunks
from evaluation.eval_class import EvalCase
from evaluation.retrieval_evaluator import RetrievalEvaluator
from evaluation.semantic_evaluator import SemanticAnswerEvaluator
from llm_services.llm_ollama_service import LLM_Ollama_Service
from retriever.fusion_rrf import RRFFusion
from retriever.retriever_bm25 import BM25ChunkRetriever
from retriever.retriever_similarity import VectorSimilarityRetriever
from vectors_service.faiss_vector_store import FaissVectorStore
from retriever.reranker_bge import BGEReranker



class RagEvaluator:

    def __init__(self,retriever_agent,rag_agent,answer_evaluator):

        self.retriever_agent = retriever_agent

        self.rag_agent = rag_agent

        self.answer_evaluator = answer_evaluator

        self.retrieval_evaluator = RetrievalEvaluator()

        # SINGLE CASE EVALUATION
        # -------------------------

    def evaluate_case(self, case, retrieval_k=10, rerank_k=5):

        # 1. RETRIEVAL
        docs = self.retriever_agent.retrieve(
            case.question,
            retrieval_k=retrieval_k,
            rerank_k=rerank_k
        )

        # 2. RETRIEVAL METRICS (delegated)
        recall = self.retrieval_evaluator.recall_at_k(
            docs,
            case.expected_chunk_ids
        )

        mrr = self.retrieval_evaluator.mrr(
            docs,
            case.expected_chunk_ids
        )

        # 3. GENERATION
        answer = self.rag_agent.get_response(
            case.question,
            docs
        )

        # 4. ANSWER EVALUATION
        answer_score = self.answer_evaluator.evaluate(
            answer,
            case.expected_answer
        )

        # 5. RETURN FULL TRACE
        return {
            "question": case.question,

            "retrieval_metrics": {
                "recall@k": recall,
                "mrr": mrr
            },

            "answer_score": answer_score,

            "generated_answer": answer,

            "expected_answer": case.expected_answer,

            "retrieved_chunk_ids": [
                doc.chunk_id for doc in docs
            ]
        }

        # -------------------------
        # FULL DATASET RUN
        # -------------------------

    def run(self, eval_cases, retrieval_k=10, rerank_k=5):

        results = []

        aggregate = defaultdict(list)

        for case in eval_cases:
            result = self.evaluate_case(
                case,
                retrieval_k=retrieval_k,
                rerank_k=rerank_k
            )

            results.append(result)

            aggregate["recall"].append(
                result["retrieval_metrics"]["recall@k"]
            )

            aggregate["mrr"].append(
                result["retrieval_metrics"]["mrr"]
            )

            aggregate["answer_score"].append(
                result["answer_score"]
            )

        summary = {
            "avg_recall@k":
                sum(aggregate["recall"]) / len(results),

            "avg_mrr":
                sum(aggregate["mrr"]) / len(results),

            "avg_answer_score":
                sum(aggregate["answer_score"]) / len(results)
        }

        return {
            "summary": summary,
            "results": results
        }

        # -------------------------
        # SAVE RESULTS
        # -------------------------

    # def save(self, output, path="rag_eval_results.json"):
    #
    #     with open(path, "w", encoding="utf-8") as f:
    #         json.dump(output, f, indent=2, ensure_ascii=False)



from typing import List

def load_eval_dataset(file_path: str) -> List[EvalCase]:

    with open(file_path, "r", encoding="utf-8") as f:

        data = json.load(f)

    return [
        EvalCase(**item)
        for item in data
    ]



data = load_f1_data()
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vector_store_type = FaissVectorStore(embedding_model=embedding_model)
chunking_strategy = RecursiveChunkingStrategy(chunk_size=1000, chunk_overlap=200)
chunks = data_to_chunks(data, chunking_strategy)
vector_store = vectorize_data_from_chunks(vector_store_type, chunks)
similarity_search_retriever = VectorSimilarityRetriever(vector_store)
bm25_retriever = BM25ChunkRetriever(chunks)
fusion = RRFFusion()
reranker = BGEReranker()
retriever_agent = RetrieverAgent(retrievers=[similarity_search_retriever, bm25_retriever],
    fusion_strategy=fusion,
    reranker=reranker)
llm_service = LLM_Ollama_Service(model="llama3.2:1b", temperature=0)
rag_agent = RagAgent(llm_service)

semantic_answer_evaluator = SemanticAnswerEvaluator(embedding_model=embedding_model)


eval_cases = load_eval_dataset("eval_dataset.json")

evaluator = RagEvaluator(
    retriever_agent=retriever_agent,
    rag_agent=rag_agent,
    answer_evaluator=semantic_answer_evaluator
)

results = evaluator.run(
    eval_cases=eval_cases,
    retrieval_k=10,
    rerank_k=5
)

