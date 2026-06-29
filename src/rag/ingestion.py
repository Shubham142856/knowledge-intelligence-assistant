"""
src/rag/ingestion.py — DocumentIngestor

Handles all document parsing (PDF, DOCX, PPTX, CSV, TXT), chunking into
500-token segments with 50-token overlap, and 384-dim sentence embedding.
"""

import os
import logging
from pathlib import Path

import PyPDF2
import pandas as pd
from docx import Document
from pptx import Presentation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

log = logging.getLogger("vyor_ai.ingestion")

# ── Singleton embedder — loaded once per process, never per-request ───────────
_EMBEDDER: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Return the module-level singleton embedder (loads on first call)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        log.info("Loading sentence-transformer model all-MiniLM-L6-v2 …")
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Embedder ready.")
    return _EMBEDDER


class DocumentIngestor:
    """
    Parse → chunk → embed any supported document type.

    Supported extensions: pdf, docx, pptx, csv, txt
    Chunk size:   500 tokens
    Chunk overlap: 50 tokens
    Embedding dim: 384  (all-MiniLM-L6-v2)
    """

    CHUNK_SIZE    = 500
    CHUNK_OVERLAP = 50
    SUPPORTED     = {"pdf", "docx", "pptx", "csv", "txt", "md"}

    def __init__(self) -> None:
        self.embedder = get_embedder()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ── Parsers ───────────────────────────────────────────────────────────────

    def parse_pdf(self, path: str) -> str:
        """Extract text from a PDF. Raises ValueError if it appears scanned."""
        text = ""
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        if len(text.strip()) < 50:
            raise ValueError(
                f"PDF appears to be scanned/image-only or empty: {path}"
            )
        return text

    def parse_docx(self, path: str) -> str:
        """Extract paragraph text from a DOCX file."""
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def parse_pptx(self, path: str) -> str:
        """Extract text from all shapes on every slide."""
        prs = Presentation(path)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text += shape.text + "\n"
        return text

    def parse_csv(self, path: str) -> str:
        """Convert CSV to a readable string table."""
        df = pd.read_csv(path)
        return df.to_string(index=False)

    def parse_txt(self, path: str) -> str:
        """Read plain text (UTF-8, lossy fallback)."""
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()

    # ── Main entry point ──────────────────────────────────────────────────────

    def ingest(self, path: str) -> tuple[list[str], list]:
        """
        Parse → chunk → embed a file.

        Args:
            path: Absolute path to the file.

        Returns:
            (chunks, embeddings)
            chunks:     list[str]   — Text chunks, each ≤ 500 tokens.
            embeddings: np.ndarray  — Shape (N, 384), dtype float32.

        Raises:
            ValueError: Unsupported extension or scanned PDF.
        """
        ext = Path(path).suffix.lstrip(".").lower()
        if ext not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported file type '.{ext}'. Allowed: {self.SUPPORTED}"
            )

        parsers = {
            "pdf":  self.parse_pdf,
            "docx": self.parse_docx,
            "pptx": self.parse_pptx,
            "csv":  self.parse_csv,
            "txt":  self.parse_txt,
            "md":   self.parse_txt,   # markdown treated as plain text
        }

        log.info(f"Parsing {Path(path).name} …")
        raw_text  = parsers[ext](path)
        log.info(f"  raw chars: {len(raw_text):,}")

        chunks = self.splitter.split_text(raw_text)
        log.info(f"  chunks: {len(chunks)}")

        embeddings = self.embedder.encode(chunks, show_progress_bar=False)
        log.info(f"  embeddings: {embeddings.shape}")

        return chunks, embeddings
