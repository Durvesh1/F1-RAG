📚 RAG System

A modular Retrieval-Augmented Generation (RAG) system with hybrid retrieval, reranking, and a full evaluation pipeline.

🚀 Features
Hybrid Retrieval: BM25 + Vector Search (FAISS)
Fusion: Reciprocal Rank Fusion (RRF)
Reranking: Pluggable rerankers (e.g., BGE / LLM-based)
LLM Generation: Ollama-based RAG response generation
Evaluation:
Retrieval: Recall@K, MRR
Answer quality: Semantic similarity

🏗️ Pipeline
Query → Retrievers (BM25 + Vector) → RRF Fusion → Reranker → LLM → Answer

📊 Evaluation

Each test case includes:

Question
Expected answer
Expected chunk IDs

Metrics:

Retrieval quality: Recall@K, MRR
Answer quality: cosine similarity (embeddings)

📌 Goal

To enable systematic comparison of RAG configurations (BM25, vector, hybrid, RRF, reranking) using measurable evaluation metrics.
