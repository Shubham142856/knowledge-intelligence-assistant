"""Smoke test for all new VYOR AI v2 API endpoints."""
import requests
import json

BASE = "http://localhost:8000"
SEP  = "=" * 50

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

all_pass = True

# 1. GET /health
section("GET /health")
r = requests.get(f"{BASE}/health", timeout=10)
print(f"  Status: {r.status_code}")
print(json.dumps(r.json(), indent=2))
assert r.status_code == 200, "Health check failed"
print("  PASS")

# 2. GET /sources
section("GET /sources  (document index)")
r = requests.get(f"{BASE}/sources", timeout=10)
print(f"  Status: {r.status_code}")
print(json.dumps(r.json(), indent=2))
assert r.status_code == 200, "/sources failed"
print("  PASS")

# 3. POST /exact-search
section("POST /exact-search  (verbatim SQLite lookup)")
r = requests.post(
    f"{BASE}/exact-search",
    json={"keyword": "VYOR-ALPHA-7", "limit": 3},
    timeout=10,
)
print(f"  Status: {r.status_code}")
print(json.dumps(r.json(), indent=2))
assert r.status_code == 200, "/exact-search failed"
print("  PASS")

# 4. POST /forget
section("POST /forget  (GDPR Article 17)")
r = requests.post(
    f"{BASE}/forget",
    json={"source": "test_document.pdf"},
    timeout=10,
)
print(f"  Status: {r.status_code}")
print(json.dumps(r.json(), indent=2))
assert r.status_code == 200, "/forget failed"
assert r.json().get("status") == "forgotten", "Unexpected status"
print("  PASS")

# 5. Bad /forget (empty source)
section("POST /forget  (validation: empty source)")
r = requests.post(f"{BASE}/forget", json={"source": ""}, timeout=10)
print(f"  Status: {r.status_code} (expected 400)")
assert r.status_code == 400, "Expected 400 for empty source"
print("  PASS")

print(f"\n{SEP}")
print("  All 5 endpoint smoke tests PASSED")
print(SEP)
