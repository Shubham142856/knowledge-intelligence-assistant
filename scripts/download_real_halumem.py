"""
scripts/download_real_halumem.py

Downloads the real HaluEval QA evaluation queries directly from the public GitHub repository
and formats them into the HaluMem schema (data/halumem/evaluation_queries.json)
for test_hallucination.py.
"""

import os, json, urllib.request
from pathlib import Path

# Public GitHub raw URL - doesn't require any token/login
HALUMEVAL_RAW_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"

def download():
    out_dir = Path("data/halumem")
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_path = out_dir / "qa_data.json"
    
    print(f"Downloading HaluEval QA dataset from {HALUMEVAL_RAW_URL}...")
    try:
        # Set a User-Agent header so GitHub doesn't block python's default user agent
        req = urllib.request.Request(
            HALUMEVAL_RAW_URL, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(qa_path, 'wb') as out_file:
            out_file.write(response.read())
            
        print("Download complete. Formatting HaluMem evaluation set...")
        
        # Load and parse the HaluEval QA JSONL data
        queries = []
        with open(qa_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    row = json.loads(line)
                    queries.append({
                        "id": f"halumem_real_{i:04d}",
                        "query": row["question"],
                        "expected_answer": row["right_answer"],
                        "hallucinated_answer": row["hallucinated_answer"]
                    })
            
        out_path = out_dir / "evaluation_queries.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(queries[:200], f, indent=2)
            
        # Clean up raw download
        if qa_path.exists():
            os.remove(qa_path)
            
        print(f"Success! Formatted and saved {len(queries[:200])} real queries to {out_path}")
        
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nYou can manually download the file from:")
        print(f"  {HALUMEVAL_RAW_URL}")
        print("And save it as: d:\\Knowledge intelligence assistant\\vyor-ai\\data\\halumem\\qa_data.json")

if __name__ == "__main__":
    download()
