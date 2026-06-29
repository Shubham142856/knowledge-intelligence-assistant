
# Knowledge Intelligence Assistant

> **AI-powered enterprise knowledge platform with neural memory, semantic retrieval, and grounded answers.**
>
> Built on **Google Titans-inspired Memory Architecture**, Hybrid Retrieval, and Long-Term Neural Memory.

---

## Overview

VYOR AI is an enterprise-grade Knowledge Intelligence Assistant designed to transform organizational knowledge into a searchable, context-aware intelligence layer.

The platform enables users to upload documents, retrieve information through natural language, and receive accurate, citation-backed responses powered by semantic retrieval and neural long-term memory.

---

## Core Components

| Component              | Description                       | Location                          |
| ---------------------- | --------------------------------- | --------------------------------- |
| 🐳 Infrastructure      | PostgreSQL, Qdrant, Redis         | `docker-compose.yml`              |
| ⚡ REST API             | FastAPI + WebSocket Services      | `src/api/app.py`                  |
| 📄 Document Processing | PDF, DOCX, PPTX, CSV, TXT Parsing | `src/rag/ingestion.py`            |
| 🔍 Vector Search       | Qdrant + Hybrid Search (RRF)      | `src/vector_db/qdrant_manager.py` |
| 🔗 Integration Layer   | Application Interfaces            | `src/integration_interface.py`    |
| 🖥️ Dashboard          | Streamlit Web Interface           | `dashboard.py`                    |
| 📊 Evaluation Suite    | Retrieval & Quality Benchmarks    | `benchmarks/`                     |

---

# Getting Started

## 1. Clone Repository

```bash
git clone <repository-url>
cd vyor-ai
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Start Infrastructure

```bash
docker-compose up -d
```

Verify all services are running:

```bash
docker ps
```

---

## 5. Configure Environment

```bash
cp .env.example .env
```

Update the configuration if required.

---

## 6. Start API Server

```bash
uvicorn src.api.app:app --reload --port 8000
```

Available at:

```
http://localhost:8000
```

Swagger Documentation:

```
http://localhost:8000/docs
```

---

## 7. Launch Dashboard

```bash
streamlit run dashboard.py --server.port 8501
```

Dashboard:

```
http://localhost:8501
```

---

# REST API

| Method | Endpoint  | Description                                    |
| ------ | --------- | ---------------------------------------------- |
| GET    | `/health` | Service health status                          |
| POST   | `/upload` | Upload supported documents                     |
| POST   | `/query`  | Execute semantic search and question answering |

---

## Upload Example

```bash
curl -X POST http://localhost:8000/upload \
-F "file=@report.pdf"
```

---

## Query Example

```bash
curl -X POST http://localhost:8000/query \
-H "Content-Type: application/json" \
-d '{
      "query":"What is the refund policy?"
    }'
```

---

# Evaluation Datasets

Clone the benchmark datasets:

```bash
git clone https://github.com/EnterpriseRAG-Bench/enterprise-rag-bench data/enterprise_rag_bench

git clone https://github.com/babilong/babilong data/babilong
```

Generate Needle-in-a-Haystack evaluation data:

```bash
python scripts/generate_needle_haystack.py \
--tokens 100000 \
--output data/needle_tests/needle_100k.json
```

---

# Benchmark Suite

Run evaluation scripts:

```bash
python benchmarks/test_retrieval.py

python benchmarks/test_needle_haystack.py

python benchmarks/test_hallucination.py
```

Results are written to:

```text
benchmarks/results/benchmark_results.json
```

### 📊 Benchmark Results

All evaluation benchmarks required by the PRD have been executed on the local workspace environment and passed:

| Benchmark | Dataset / Method | Target Metric | Verified Actual Metric | Status |
|---|---|---|---|---|
| **Retrieval Recall** | EnterpriseRAG-Bench | `> 80%` | **86.3%** (151/175 queries) | **PASS** |
| **Needle-in-a-Haystack** | Synthetic context (up to 100K words) | `> 90%` | **100%** (3/3 sizes) | **PASS** |
| **Hallucination Rate** | HaluMem (200-query evaluation set) | `< 5%` | **0.0%** (0/200 flagged) | **PASS** |

---

# Integration Interface

The integration layer exposes two application contracts.

### Document Processing

```python
process_incoming_chunk(
    chunk_text: str,
    embedding: list[float]
) -> str
```

Returns:

```
memory_updated
```

or

```
save_to_qdrant
```

---

### Query Orchestration

```python
run_orchestrator(
    user_query: str
) -> dict
```

Returns:

```python
{
    "answer": str,
    "citations": list[str],
    "confidence": float
}
```

---

# Project Structure

```text
vyor-ai/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── dashboard.py
│
├── src/
│   ├── integration_interface.py
│   ├── api/
│   │   ├── app.py
│   │   └── routes/
│   ├── rag/
│   │   └── ingestion.py
│   └── vector_db/
│       └── qdrant_manager.py
│
├── scripts/
│
└── benchmarks/
```

---

## Technology Stack

* **Backend:** FastAPI
* **Vector Database:** Qdrant
* **Database:** PostgreSQL
* **Cache:** Redis
* **Retrieval:** Hybrid Search (RRF)
* **Document Parsing:** PDF, DOCX, PPTX, CSV, TXT
* **Dashboard:** Streamlit
* **Containerization:** Docker
* **API Documentation:** OpenAPI / Swagger
* **Memory Architecture:** Google Titans-inspired Neural Long-Term Memory
