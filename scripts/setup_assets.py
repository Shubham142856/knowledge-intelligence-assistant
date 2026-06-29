"""
scripts/setup_assets.py
Run this ONCE to copy the VYOR AI logo into the assets folder.

Usage:
    & "d:\Knowledge intelligence assistant\vyor-ai\.venv\Scripts\python.exe" "d:\Knowledge intelligence assistant\vyor-ai\scripts\setup_assets.py"
"""

import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR   = PROJECT_ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

# ── Try to locate the logo image ─────────────────────────────────────────────
# Priority 1: already in assets/
# Priority 2: artifact path from the conversation (copy it over)
LOGO_DST = ASSETS_DIR / "logo.png"

ARTIFACT_SRC = Path(r"C:\Users\HP\.gemini\antigravity-ide\brain\1fa7f3fe-d004-4431-8cdf-75cfe1314ad8\media__1782589581704.png")

if LOGO_DST.exists():
    print(f"[OK] Logo already exists: {LOGO_DST}")
elif ARTIFACT_SRC.exists():
    shutil.copy(ARTIFACT_SRC, LOGO_DST)
    print(f"[OK] Logo copied from artifact -> {LOGO_DST}")
else:
    print(
        "[INFO] Logo source not found automatically.\n"
        "Please manually copy your VYOR AI logo PNG to:\n"
        f"  {LOGO_DST}\n"
        "Then rerun this script or just restart the dashboard."
    )
