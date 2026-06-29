# VYOR AI — Knowledge Intelligence Assistant

> Powered by **Google Titans Architecture** · Neural Long-Term Memory · Cited Answers

## Team Member 2 — Infrastructure Engineer

This repository contains the complete **Team 2 infrastructure** scope:

| Layer | What | File |
|-------|------|------|
| 🐳 Containers | Qdrant · Redis · PostgreSQL | `docker-compose.yml` |
| ⚡ API | FastAPI REST + WebSocket | `src/api/app.py` |
| 📄 Parsers | PDF · DOCX · PPTX · CSV · TXT | `src/rag/ingestion.py` |
| 🔍 Vector DB | Qdrant + Hybrid Search (RRF) | `src/vector_db/qdrant_manager.py` |
| 🤝 Integration | Team 1 boundary (stubs/real) | `src/integration_interface.py` |
| 🖥️ Dashboard | Streamlit chat UI | `dashboard.py` |
| 📊 Benchmarks | Retrieval · Needle · Hallucination | `benchmarks/` |

---

## Quick Start

### 1. Install dependencies

```bash
cd vyor-ai
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Start infrastructure containers

```bash
docker-compose up -d
docker ps   # All 3 containers should show "healthy"
```

### 3. Copy environment variables

```bash
cp .env.example .env
# Edit .env if your URLs differ from defaults
```

### 4. Start FastAPI server

```bash
uvicorn src.api.app:app --reload --port 8000
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 5. Start Streamlit dashboard (separate terminal)

```bash
streamlit run dashboard.py --server.port 8501
# → http://localhost:8501
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service status map |
| `POST` | `/upload` | Upload a document (PDF/DOCX/PPTX/CSV/TXT) |
| `POST` | `/query` | Ask a natural-language question |

### Example: Upload

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@report.pdf"
```

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?"}'
```

---

## Download Test Datasets (Day 3–7)

```bash
# EnterpriseRAG-Bench — primary retrieval test corpus
git clone https://github.com/EnterpriseRAG-Bench/enterprise-rag-bench data/enterprise_rag_bench

# BABILong — long-context Titans memory test
git clone https://github.com/babilong/babilong data/babilong

# Generate needle-in-haystack test files
python scripts/generate_needle_haystack.py --tokens 100000 \
  --output data/needle_tests/needle_100k.json
```

---

## Run Benchmarks (Day 7)

```bash
# Retrieval accuracy — target > 80%
python benchmarks/test_retrieval.py

# Needle-in-haystack — target > 90%
python benchmarks/test_needle_haystack.py

# Hallucination rate — target < 5%
python benchmarks/test_hallucination.py

# Results saved to:
cat benchmarks/results/benchmark_results.json
```

---

## Integration Contracts (Day 5)

Team 1 provides `src/integration_interface.py`. Until Day 5, stubs are active.

```python
# Contract 1 — called for every parsed chunk
process_incoming_chunk(chunk_text: str, embedding: list[float]) -> str
# returns: "memory_updated" | "save_to_qdrant"

# Contract 2 — called for every user query
run_orchestrator(user_query: str) -> dict
# returns: {"answer": str, "citations": list[str], "confidence": float}
```

---

## Directory Structure

```
vyor-ai/
├── docker-compose.yml          # Containers
├── Dockerfile                  # App image
├── requirements.txt
├── .env.example
├── dashboard.py                # Streamlit UI
│
├── src/
│   ├── integration_interface.py   # Team 1 boundary
│   ├── api/
│   │   ├── app.py
│   │   └── routes/
│   │       ├── upload.py
│   │       └── query.py
│   ├── rag/
│   │   └── ingestion.py
│   └── vector_db/
│       └── qdrant_manager.py
│
├── scripts/
│   └── generate_needle_haystack.py
│
└── benchmarks/
    ├── test_retrieval.py
    ├── test_needle_haystack.py
    ├── test_hallucination.py
    └── results/
        └── benchmark_results.json
```

---

*VYOR AI | Team Member 2 — Infrastructure Engineer | v1.0 | June 2026*
