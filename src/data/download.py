"""
download.py — pulls the Kaggle Customer Support on Twitter dataset into data/raw/

Usage:
    python -m src.data.download
"""

import os
import zipfile
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path("data/raw")
DATASET = "thoughtvector/customer-support-on-twitter"


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = list(RAW_DIR.glob("*.csv"))
    if csv_files:
        print(f"[download] Dataset already present in {RAW_DIR}/")
        for f in csv_files:
            print(f"  {f.name}  ({f.stat().st_size / 1_000_000:.1f} MB)")
        return

    print(f"[download] Pulling dataset: {DATASET}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("[download] ERROR:", result.stderr)
        raise RuntimeError("Kaggle download failed — check your kaggle.json token.")

    # unzip
    zips = list(RAW_DIR.glob("*.zip"))
    for z in zips:
        print(f"[download] Extracting {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW_DIR)
        z.unlink()

    csv_files = list(RAW_DIR.glob("*.csv"))
    print(f"[download] Done. Files in {RAW_DIR}/:")
    for f in csv_files:
        print(f"  {f.name}  ({f.stat().st_size / 1_000_000:.1f} MB)")


if __name__ == "__main__":
    download()