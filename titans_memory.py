"""
titans_memory.py — VYORNeuralBrain (Titans-Inspired Long-Term Memory)

Fixes applied vs v1:
  1. dim auto-matches active embedder backend (384 local | 2048 openrouter).
  2. NaN guard: detect NaN in output and log a warning + return zeros (safe fallback).
  3. Gradient clipping applied on every learn_new_info() call to prevent explosions.
  4. Tuple output from NeuralMemory handled safely (extracts [0] element).
  5. Safe fallback when titans_pytorch is not installed (stub mode).
"""

import logging
import torch
import torch.nn as nn

log = logging.getLogger("vyor_ai.titans_memory")

# Auto-match embedding dimension from the active backend
try:
    from src.rag.ingestion import EMBEDDING_DIM as _EMBED_DIM
except ImportError:
    _EMBED_DIM = 384   # fallback if imported standalone

# ── Try importing titans-pytorch; fall back to a stub if unavailable ──────────
try:
    from titans_pytorch import NeuralMemory
    _TITANS_AVAILABLE = True
except ImportError:
    _TITANS_AVAILABLE = False
    log.warning(
        "titans-pytorch not installed. VYORNeuralBrain will run in STUB mode "
        "(pass-through only, no LTM learning). Install via: pip install titans-pytorch"
    )


class _StubMemory(nn.Module):
    """No-op replacement when titans-pytorch is unavailable."""
    def __init__(self, dim: int, chunk_size: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class VYORNeuralBrain(nn.Module):
    """
    Titans Neural Long-Term Memory module for VYOR AI.

    Args:
        dim:        Embedding dimension. Must match the sentence-transformer
                    output dim (384 for all-MiniLM-L6-v2).
        chunk_size: Sequence is partitioned into sub-sequences of this length
                    before being processed by NeuralMemory.
    """

    DIM        = _EMBED_DIM  # auto: 384 (local) | 2048 (openrouter)
    CHUNK_SIZE = 64
    MAX_GRAD_NORM = 1.0   # gradient clipping threshold

    def __init__(self, dim: int = DIM, chunk_size: int = CHUNK_SIZE):
        super().__init__()
        self.dim = dim
        self.stub_mode = not _TITANS_AVAILABLE

        if self.stub_mode:
            self.memory = _StubMemory(dim=dim, chunk_size=chunk_size)
            log.info("VYORNeuralBrain initialised in STUB mode (no LTM learning).")
        else:
            self.memory = NeuralMemory(dim=dim, chunk_size=chunk_size)
            log.info(
                f"VYORNeuralBrain initialised — Titans LTM "
                f"(dim={dim}, chunk_size={chunk_size})."
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _is_valid(tensor: torch.Tensor) -> bool:
        """Return False if tensor contains NaN or Inf values."""
        return not (torch.isnan(tensor).any() or torch.isinf(tensor).any())

    def _safe_zeros(self, shape) -> torch.Tensor:
        """Return a zero tensor as a safe fallback."""
        return torch.zeros(shape, dtype=torch.float32)

    # ── Public API ────────────────────────────────────────────────────────────

    def learn_new_info(self, retrieved_vector: torch.Tensor) -> torch.Tensor:
        """
        Write high-surprise content into LTM via a test-time gradient update.

        Args:
            retrieved_vector: Shape (batch, seq_len, dim).

        Returns:
            Updated memory output tensor, or zero tensor on NaN failure.
        """
        if self.stub_mode:
            return retrieved_vector

        # Input NaN guard
        if not self._is_valid(retrieved_vector):
            log.warning("LTM learn_new_info: input contains NaN/Inf — skipping update.")
            return self._safe_zeros(retrieved_vector.shape)


        try:
            # Enable gradient computation for test-time memory update
            with torch.enable_grad():
                result = self.memory(retrieved_vector)
                # titans-pytorch may return (output, state) tuple — extract tensor
                updated_memory = result[0] if isinstance(result, (tuple, list)) else result

                # Gradient clipping to prevent explosion
                nn.utils.clip_grad_norm_(
                    self.memory.parameters(),
                    max_norm=self.MAX_GRAD_NORM
                )

            # Output NaN guard
            if not self._is_valid(updated_memory):
                log.warning(
                    "LTM learn_new_info: output contains NaN/Inf — "
                    "returning zero fallback."
                )
                return self._safe_zeros(updated_memory.shape)

            log.info("Titans LTM: Neural weights updated with new high-surprise facts.")
            return updated_memory

        except Exception as exc:
            log.error(f"LTM learn_new_info failed: {exc}. Returning zero fallback.")
            return self._safe_zeros(retrieved_vector.shape)

    def recall_info(self, query_vector: torch.Tensor) -> torch.Tensor:
        """
        Retrieve information from LTM for a query vector.

        Args:
            query_vector: Shape (batch, seq_len, dim).

        Returns:
            Recalled context tensor, or zero tensor on NaN failure.
        """
        if self.stub_mode:
            return query_vector

        if not self._is_valid(query_vector):
            log.warning("LTM recall_info: query contains NaN/Inf — returning zeros.")
            return self._safe_zeros(query_vector.shape)

        try:
            with torch.no_grad():
                result  = self.memory(query_vector)
                # titans-pytorch may return (output, state) tuple — extract tensor
                recalled = result[0] if isinstance(result, (tuple, list)) else result

            if not self._is_valid(recalled):
                log.warning(
                    "LTM recall_info: output NaN/Inf detected — returning zeros."
                )
                return self._safe_zeros(recalled.shape)

            return recalled

        except Exception as exc:
            log.error(f"LTM recall_info failed: {exc}. Returning zero fallback.")
            return self._safe_zeros(query_vector.shape)