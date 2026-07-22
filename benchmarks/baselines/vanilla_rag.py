"""
benchmarks/baselines/vanilla_rag.py — Vanilla RAG Baseline System

Simple dense retrieval + single LLM call. No agents, no debate, no surprise gate, no memory.
"""

import time
from src.vector_db.qdrant_manager import QdrantManager
from src.agents.llm import call_llm

def run_vanilla_rag(query: str, top_k: int = 5) -> dict:
    qdrant = QdrantManager()
    
    # 1. Embed query
    q_emb = qdrant.embedder.encode(query)
    
    # 2. Dense search (no hybrid, no BM25, no RRF, no Cross-Encoder reranking)
    hits = qdrant.client.search(
        collection_name=qdrant.COLLECTION,
        query_vector=q_emb.tolist(),
        limit=top_k,
    )
    
    context_chunks = []
    citations = []
    for hit in hits:
        payload = hit.payload or {}
        text = payload.get("text", "")
        src = payload.get("source", "unknown")
        score = getattr(hit, "score", 0.0)
        if text and score >= 0.65:
            context_chunks.append(f"Source: {src} | {text}")
            if src not in citations:
                citations.append(src)

                
    # 3. Call LLM with simple non-agentic prompt
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
    
    confidence = 0.8 if context_chunks else 0.2
    
    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "latency_s": latency,
        "uncertainty": len(context_chunks) == 0
    }
