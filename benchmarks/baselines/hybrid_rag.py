"""
benchmarks/baselines/hybrid_rag.py — Hybrid RAG Baseline System

Dense + BM25 with basic RRF fusion and reranking. No agents, no debate, no memory.
"""

import time
from src.vector_db.qdrant_manager import QdrantManager
from src.agents.llm import call_llm

def run_hybrid_rag(query: str, top_k: int = 5) -> dict:
    qdrant = QdrantManager()
    
    # 1. Hybrid search (dense + BM25 + RRF + Cross-Encoder if enabled)
    results = qdrant.hybrid_search(query, top_k=top_k)
    
    context_chunks = []
    citations = []
    for r in results:
        text = r.get("text", "")
        src = r.get("source", "unknown")
        if text:
            context_chunks.append(f"Source: {src} | {text}")
            if src not in citations:
                citations.append(src)
                
    # 2. Call LLM with simple non-agentic prompt
    system_prompt = (
        "You are a helpful assistant. Answer the user query based ONLY on the provided context. "
        "Keep your answer factual and cite the source filename like [filename.pdf] where appropriate. "
        "If the context does not contain the answer, say that you cannot find the answer."
    )
    context_str = "\n\n".join(context_chunks)
    user_content = f"Context:\n{context_str}\n\nQuery: {query}"
    
    t0 = time.perf_counter()
    answer = call_llm(system_prompt, user_content)
    latency = time.perf_counter() - t0
    
    confidence = 0.85 if context_chunks else 0.15
    
    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "latency_s": latency,
        "uncertainty": len(context_chunks) == 0
    }
