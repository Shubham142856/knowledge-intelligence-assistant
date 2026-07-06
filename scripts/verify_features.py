"""Verification script for new VYOR AI features."""
import sys
sys.path.insert(0, ".")

import torch
import tempfile
import os

print("=== VYOR AI Feature Verification ===\n")

# 1. NaN protection in VYORNeuralBrain
print("[TEST 1] NaN guard in VYORNeuralBrain...")
from titans_memory import VYORNeuralBrain
brain = VYORNeuralBrain(dim=384)
bad_tensor = torch.full((1, 1, 384), float("nan"))
out = brain.learn_new_info(bad_tensor)
nan_guarded = torch.all(out == 0).item()
print(f"  Input NaN -> output zeros: {nan_guarded}")
assert nan_guarded, "NaN guard FAILED"
print("  PASS\n")

# 2. Dimension fix (dim=384, was 256)
print("[TEST 2] VYORNeuralBrain dim=384 alignment...")
assert brain.dim == 384, f"dim={brain.dim}, expected 384"
good_tensor = torch.rand(1, 1, 384)
out2 = brain.recall_info(good_tensor)
print(f"  dim={brain.dim}, recall output shape: {out2.shape}")
print("  PASS\n")

# 3. VYORSurpriseGate Huber Loss + adaptive threshold
print("[TEST 3] VYORSurpriseGate Huber Loss...")
from surprise_gate import VYORSurpriseGate
gate = VYORSurpriseGate()
v1 = torch.rand(384)
v2 = torch.rand(384)
loss = gate.compute_huber_loss(v1, v2)
decision, threshold = gate.update_and_route(loss)
print(f"  loss={loss:.4f}, threshold={threshold}, decision={decision}")
assert decision in ("memory_updated", "save_to_qdrant"), "Bad routing decision"
print("  PASS\n")

# 4. ExactFactStore SQLite
print("[TEST 4] ExactFactStore exact lookup + GDPR delete...")
from src.vector_db.qdrant_manager import ExactFactStore
db_path = tempfile.mktemp(suffix=".db")
store = ExactFactStore(db_path)
store.insert("abc123", "The project code is VYOR-ALPHA-7", "test.pdf", "doc1", "2026-01-01T00:00:00")
results = store.exact_search("VYOR-ALPHA-7", limit=3)
print(f"  Exact search hits: {len(results)}")
assert len(results) == 1, "Expected 1 result"
print(f"  Text: {results[0]['text']}")
deleted = store.delete_by_source("test.pdf")
print(f"  GDPR delete: {deleted} rows removed")
after = store.exact_search("VYOR-ALPHA-7", limit=3)
assert len(after) == 0, "Expected 0 results after delete"
os.unlink(db_path)
print("  PASS\n")

print("=" * 40)
print("All 4 feature tests PASSED.")
print("=" * 40)
