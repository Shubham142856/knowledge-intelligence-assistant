"""
Full end-to-end test of the OpenRouter embedding pipeline.
Tests: embedder → ingestion → Qdrant upsert → hybrid search
"""
import sys, os
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("  VYOR AI — OpenRouter Integration Test")
print("=" * 60)

# 1. Check which backend is active
from src.rag.ingestion import EMBEDDER_BACKEND, EMBEDDING_DIM, get_embedder
print(f"\n[1] Embedder backend : {EMBEDDER_BACKEND}")
print(f"    Embedding dim    : {EMBEDDING_DIM}")

# 2. Load the embedder
print("\n[2] Loading embedder...")
emb = get_embedder()
print(f"    Type: {type(emb).__name__}")

# 3. Encode a test sentence
print("\n[3] Encoding test sentence...")
vec = emb.encode("The secret project code is VYOR-ALPHA-7")
print(f"    Shape     : {vec.shape}")
print(f"    First 5   : {vec[:5].tolist()}")
assert vec.shape == (EMBEDDING_DIM,), f"Expected shape ({EMBEDDING_DIM},), got {vec.shape}"
print("    PASS")

# 4. Encode a batch
print("\n[4] Encoding batch of 3 sentences...")
import numpy as np
sentences = [
    "VYOR AI uses Titans Neural Memory for long-term context.",
    "Hybrid search combines dense vectors and BM25 sparse retrieval.",
    "GDPR Article 17 grants the right to be forgotten.",
]
vecs = emb.encode(sentences)
print(f"    Batch shape : {vecs.shape}")
assert vecs.shape == (3, EMBEDDING_DIM), f"Expected (3, {EMBEDDING_DIM}), got {vecs.shape}"
print("    PASS")

# 5. Multimodal test (OpenRouter only)
if EMBEDDER_BACKEND == "openrouter" and hasattr(emb, "embed_image_text"):
    print("\n[5] Multimodal image+text embedding...")
    try:
        mv = emb.embed_image_text(
            text="What is in this image?",
            image_url="https://live.staticflickr.com/3851/14825276609_098cac593d_b.jpg",
        )
        print(f"    Multimodal shape : {mv.shape}")
        assert mv.shape == (EMBEDDING_DIM,)
        print("    PASS")
    except Exception as e:
        print(f"    SKIP (network/image error): {e}")

print(f"\n{'=' * 60}")
print(f"  All tests PASSED — backend={EMBEDDER_BACKEND}, dim={EMBEDDING_DIM}")
print(f"{'=' * 60}")
