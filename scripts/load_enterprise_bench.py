"""
scripts/load_enterprise_bench.py

Ingest EnterpriseRAG-Bench source documents into Qdrant.

Usage:
    # Ingest only the documents needed for the test questions (highly recommended!):
    python scripts/load_enterprise_bench.py --smart-limit 200

    # Ingest N random/alphabetical files:
    python scripts/load_enterprise_bench.py --limit 200

    # Ingest everything:
    python scripts/load_enterprise_bench.py
"""

import sys, os, json, argparse, hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_metadata_and_text(path: Path) -> tuple[str, str | None]:
    """Returns (doc_id, text) extracted from JSON/text source files."""
    try:
        if path.suffix == ".json":
            with open(path, encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
            doc_id = data.get("dataset_doc_uuid", "")
            
            # Flatten string fields
            texts = []
            def collect(obj):
                if isinstance(obj, str) and len(obj.strip()) > 10:
                    texts.append(obj.strip())
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        if k not in {"dataset_doc_uuid", "repo", "pr_number"}: # skip metadata
                            collect(v)
                elif isinstance(obj, list):
                    for item in obj:
                        collect(item)
            collect(data)
            return doc_id, "\n\n".join(texts) if texts else None

        elif path.suffix in {".md", ".txt", ".yaml", ".yml"}:
            with open(path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
            doc_id = hashlib.sha256(content.encode()).hexdigest()[:32]
            return doc_id, content if len(content.strip()) > 50 else None

    except Exception as e:
        print(f"  [skip] {path.name}: {e}")
        return "", None


def get_all_source_files(sources_dir: Path) -> list[Path]:
    """Recursively collect all supported source files."""
    supported = {".json", ".md", ".txt", ".yaml", ".yml"}
    files = []
    for f in sources_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in supported:
            files.append(f)
    return sorted(files)


def get_smart_limit_files(data_dir: Path, max_questions: int) -> list[Path]:
    """Extract the specific files mapped to basic test questions."""
    q_path = data_dir / "questions.jsonl"
    uuid_path = data_dir / "generated_data" / "uuid_index.json"

    if not q_path.exists() or not uuid_path.exists():
        print("ERROR: questions.jsonl or uuid_index.json not found for smart-limit.")
        return []

    with open(q_path, encoding="utf-8") as fh:
        questions = [json.loads(line) for line in fh if line.strip()]

    with open(uuid_path, encoding="utf-8") as fh:
        uuid_index = json.load(fh)

    basic_q = [q for q in questions if q.get("question_type") == "basic"][:max_questions]
    expected_ids = []
    for q in basic_q:
        expected_ids.extend(q.get("expected_doc_ids", []))

    sources_dir = data_dir / "generated_data" / "sources"
    files = []
    for doc_id in set(expected_ids):
        rel_path = uuid_index.get(doc_id)
        if rel_path:
            full_path = sources_dir / rel_path
            if full_path.exists():
                files.append(full_path)
    return files


def ingest(data_dir: str = "data/enterprise_rag_bench", limit: int | None = None, smart_limit: int | None = None) -> int:
    from src.rag.ingestion import DocumentIngestor
    from src.vector_db.qdrant_manager import QdrantManager
    from src.integration_interface import process_incoming_chunk

    base_path = Path(data_dir)
    sources_dir = base_path / "generated_data" / "sources"

    if not sources_dir.exists():
        print(f"ERROR: Sources directory not found: {sources_dir}")
        return 0

    if smart_limit:
        print(f"Collecting expected documents for the first {smart_limit} basic questions...")
        all_files = get_smart_limit_files(base_path, smart_limit)
    else:
        all_files = get_all_source_files(sources_dir)
        if limit:
            all_files = all_files[:limit]

    print(f"Found {len(all_files)} source files to ingest...")

    ingestor = DocumentIngestor()
    qdrant   = QdrantManager()
    total_saved = 0
    total_files = 0

    for i, file_path in enumerate(all_files):
        doc_id, text = extract_metadata_and_text(file_path)
        if not text:
            continue

        try:
            import tempfile
            suffix = file_path.suffix if file_path.suffix in {".txt", ".md"} else ".txt"
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(text)
                tmp_path = tmp.name

            chunks, embeddings = ingestor.ingest(tmp_path)
            os.unlink(tmp_path)

            source_name = str(file_path.relative_to(sources_dir))
            saved = 0
            for chunk, emb in zip(chunks, embeddings):
                decision = process_incoming_chunk(chunk, emb.tolist())
                if decision == "save_to_qdrant":
                    qdrant.save([chunk], [emb], source_name, doc_id=doc_id)
                    saved += 1

            total_saved += saved
            total_files += 1

            if (i + 1) % 20 == 0 or (i + 1) == len(all_files):
                print(f"  [{i+1}/{len(all_files)}] Processed -- total chunks: {total_saved}")

        except Exception as e:
            print(f"  ERROR {file_path.name}: {e}")

    print(f"\nDone. Ingested {total_files} files -> {total_saved} chunks saved to Qdrant.")
    return total_saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest EnterpriseRAG-Bench into Qdrant")
    parser.add_argument("--data-dir", default="data/enterprise_rag_bench")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of files to ingest alphabetically")
    parser.add_argument("--smart-limit", type=int, default=None,
                        help="Ingest only files referenced by the first N basic questions")
    args = parser.parse_args()
    ingest(args.data_dir, args.limit, args.smart_limit)
