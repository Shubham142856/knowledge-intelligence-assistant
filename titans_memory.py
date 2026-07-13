"""
titans_memory.py — VYORNeuralBrain (Titans-Inspired Long-Term Memory wrapper)

Bridges calls to src/core/titans_memory_core.py to support advanced LTM, STM,
and the three architectural variants (MAC, MAG, MAL).
"""

import os
import logging
import torch
import torch.nn as nn
from src.core.titans_memory_core import TitansMemoryCore

log = logging.getLogger("vyor_ai.titans_memory")

# Auto-match embedding dimension from the active backend
try:
    from src.rag.ingestion import EMBEDDING_DIM as _EMBED_DIM
except ImportError:
    _EMBED_DIM = 384   # fallback if imported standalone


class VYORNeuralBrain(nn.Module):
    """
    Wrapper around TitansMemoryCore for VYOR AI.
    Loads the variant configuration from the TITANS_VARIANT environment variable.
    """

    DIM = _EMBED_DIM
    CHUNK_SIZE = 64

    def __init__(self, dim: int = DIM, chunk_size: int = CHUNK_SIZE):
        super().__init__()
        self.dim = dim
        self.chunk_size = chunk_size
        
        # Load variant configuration from environment
        variant = os.getenv("TITANS_VARIANT", "MAC").upper().strip()
        num_p_tokens = int(os.getenv("TITANS_PERSISTENT_TOKENS", "8"))
        
        self.core = TitansMemoryCore(
            dim=dim,
            variant=variant,
            num_persistent_tokens=num_p_tokens,
            chunk_size=chunk_size
        )
        log.info(f"VYORNeuralBrain bridge initialized utilizing core variant: {variant}")

    def learn_new_info(self, retrieved_vector: torch.Tensor, alpha_t: float = 0.0) -> torch.Tensor:
        """
        Write high-surprise content into LTM via a test-time gradient update.
        """
        # Learn and return updated memory representation
        self.core.learn_new_info(retrieved_vector, alpha_t=alpha_t)
        return self.recall_info(retrieved_vector)

    def recall_info(self, query_vector: torch.Tensor) -> torch.Tensor:
        """
        Retrieve information from the Titans core for a query vector.
        """
        # Wrap core forward pass
        return self.core(query_vector)