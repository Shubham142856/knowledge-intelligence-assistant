"""
tests/test_core.py — pytest suite for VYOR AI Core Components

Runs entirely offline using USE_MOCK_LLM=1 and a local Qdrant path.
No API keys required.
"""

import os
import sys
import pytest
import torch

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force mock LLM for all tests
os.environ.setdefault("USE_MOCK_LLM", "1")
os.environ.setdefault("QDRANT_PATH", "/tmp/qdrant_test_db")


# ─────────────────────────────────────────────────────────────────────
# 1. Titans Neural LTM + VYORNeuralBrain
# ─────────────────────────────────────────────────────────────────────


class TestTitansMemory:
    def test_brain_initialises(self):
        """VYORNeuralBrain should construct without errors."""
        from titans_memory import VYORNeuralBrain

        brain = VYORNeuralBrain(dim=256)
        assert brain is not None

    def test_learn_and_recall(self):
        """learn_new_info + recall_info should return a tensor of correct shape."""
        from titans_memory import VYORNeuralBrain

        brain = VYORNeuralBrain(dim=256)
        vec = torch.randn(1, 1, 256)
        brain.learn_new_info(vec)
        recalled = brain.recall_info(vec)
        assert recalled.shape[-1] == 256, "Recalled tensor dim must match input dim"

    def test_nan_guard(self):
        """NaN input to VYORNeuralBrain must not raise and must return a valid tensor."""
        from titans_memory import VYORNeuralBrain

        brain = VYORNeuralBrain(dim=256)
        nan_vec = torch.full((1, 1, 256), float("nan"))
        # Should return gracefully without exception
        try:
            result = brain.recall_info(nan_vec)
            assert result is not None
        except Exception as exc:
            pytest.fail(f"NaN input raised unexpected exception: {exc}")

    def test_variants_mac_mag_mal(self):
        """All three Titans variants (MAC, MAG, MAL) must produce output tensors."""
        from src.core.titans_memory_core import TitansMemoryCore

        for variant in ["MAC", "MAG", "MAL"]:
            core = TitansMemoryCore(dim=64, variant=variant, num_persistent_tokens=0)
            x = torch.randn(1, 4, 64)
            out = core(x)
            assert out.shape[-1] == 64, f"Variant {variant}: output dim mismatch"


# ─────────────────────────────────────────────────────────────────────
# 2. Surprise Gate
# ─────────────────────────────────────────────────────────────────────


class TestSurpriseGate:
    def test_gate_initialises(self):
        from surprise_gate import VYORSurpriseGate

        gate = VYORSurpriseGate()
        assert gate is not None

    def test_huber_loss_zero_for_identical_inputs(self):
        """Identical vectors should yield near-zero Huber loss."""
        from surprise_gate import VYORSurpriseGate

        gate = VYORSurpriseGate()
        v = torch.ones(1, 1, 64)
        loss = gate.compute_huber_loss(v, v)
        assert loss < 1e-6, f"Expected ~0 loss, got {loss}"

    def test_high_surprise_routes_to_memory(self):
        """Very high-loss chunks should be routed to memory after warmup."""
        from surprise_gate import VYORSurpriseGate

        gate = VYORSurpriseGate()
        base = torch.zeros(1, 1, 64)
        # Warmup with 12 low-surprise chunks
        for _ in range(12):
            chunk = torch.randn(1, 1, 64) * 0.05
            loss = gate.compute_huber_loss(chunk, base)
            gate.update_and_route(loss)
        # Now send a very high-surprise chunk
        big_chunk = torch.randn(1, 1, 64) * 20.0
        high_loss = gate.compute_huber_loss(big_chunk, base)
        decision, _ = gate.update_and_route(high_loss)
        assert decision == "memory_updated", f"Expected memory_updated, got {decision}"

    def test_low_surprise_routes_to_qdrant(self):
        """A chunk with zero loss should drop accumulator below threshold and route to qdrant."""
        from surprise_gate import VYORSurpriseGate

        gate = VYORSurpriseGate()
        base = torch.zeros(1, 1, 64)

        # Warmup with high-surprise chunks to elevate the threshold.
        for _ in range(50):
            chunk = torch.randn(1, 1, 64) * 3.0
            loss = gate.compute_huber_loss(chunk, base)
            gate.update_and_route(loss)

        # Now send a zero-loss chunk, which drops the accumulator below the 80th-pct threshold
        tiny = torch.zeros(1, 1, 64)
        loss = gate.compute_huber_loss(tiny, base)
        decision, threshold = gate.update_and_route(loss)
        assert decision == "save_to_qdrant", (
            f"Expected save_to_qdrant after accumulator drain, "
            f"got {decision} (threshold={threshold:.6f}, acc={gate.surprise_accumulator:.6f})"
        )


# ─────────────────────────────────────────────────────────────────────
# 3. Orchestrator (mock LLM — no real API needed)
# ─────────────────────────────────────────────────────────────────────


class TestOrchestrator:
    def test_execute_query_returns_contract(self):
        """execute_query must return answer, citations, confidence."""
        from orchestrator import VYOROrchestrator

        orch = VYOROrchestrator()
        result = orch.execute_query("What is the medical leave policy?")
        assert "answer" in result, "Missing 'answer' key"
        assert "citations" in result, "Missing 'citations' key"
        assert isinstance(result["confidence"], float), "confidence must be float"

    def test_run_alias_works(self):
        """run() must behave identically to execute_query()."""
        from orchestrator import VYOROrchestrator

        orch = VYOROrchestrator()
        result = orch.run("Hello")
        assert "answer" in result

    def test_ltm_route_greeting(self):
        """Greetings should route to ltm and still return valid contract."""
        from orchestrator import VYOROrchestrator

        orch = VYOROrchestrator()
        result = orch.execute_query("Hello, how are you?")
        assert "answer" in result

    def test_confidence_is_bounded(self):
        """Confidence score must be between 0.0 and 1.0."""
        from orchestrator import VYOROrchestrator

        orch = VYOROrchestrator()
        result = orch.execute_query("Explain the leave approval process.")
        assert 0.0 <= result["confidence"] <= 1.0, f"Confidence out of range: {result['confidence']}"


# ─────────────────────────────────────────────────────────────────────
# 4. Integration Interface
# ─────────────────────────────────────────────────────────────────────


class TestIntegrationInterface:
    def test_process_chunk_returns_route(self):
        """process_incoming_chunk must return a known routing destination string."""
        from integration_interface import VYORIntegrationInterface

        bridge = VYORIntegrationInterface(dim=256)
        vec = torch.randn(1, 1, 256) * 3.0
        route = bridge.process_incoming_chunk(vec, raw_text_meta="Test doc")
        # The interface maps gate decisions to human-readable destination labels:
        #   "memory_updated" -> "TITANS_NEURAL_MEMORY"
        #   "save_to_qdrant" -> "PARTNER_QDRANT_DB"
        valid_routes = ("TITANS_NEURAL_MEMORY", "PARTNER_QDRANT_DB")
        assert route in valid_routes, f"Unexpected route: {route!r}. Expected one of {valid_routes}"

    def test_run_orchestrator_returns_answer(self):
        from integration_interface import VYORIntegrationInterface

        bridge = VYORIntegrationInterface(dim=256)
        result = bridge.run_orchestrator("What is the leave policy?")
        assert "answer" in result
        assert "confidence" in result
