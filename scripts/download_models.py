#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# scripts/download_models.py
# Download all-MiniLM-L6-v2 ONNX model and vocabulary into data/models/
#
# Model:   sentence-transformers/all-MiniLM-L6-v2
#          384 dimensions · 22 MB · Apache-2.0
#          https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
#
# Usage:
#   python scripts/download_models.py               # download to data/models/
#   python scripts/download_models.py --force       # re-download even if present
#   python scripts/download_models.py --out <dir>   # custom output directory
# -----------------------------------------------------------------------------

import argparse
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("error: 'requests' not found.  pip install requests", file=sys.stderr)
    sys.exit(1)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent

MODEL_NAME = "all-MiniLM-L6-v2"
BASE_URL   = f"https://huggingface.co/sentence-transformers/{MODEL_NAME}/resolve/main"


def _human_size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def download(url: str, dst: Path, label: str, force: bool = False) -> None:
    if dst.exists() and not force:
        print(f"  [skip]  {label}  (already present — use --force to re-download)")
        return

    print(f"  [fetch] {label}")
    print(f"          {url}")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total    = int(response.headers.get("content-length", 0))
    received = 0

    with dst.open("wb") as fh:
        for chunk in response.iter_content(chunk_size=65536):
            fh.write(chunk)
            received += len(chunk)
            if total:
                pct  = received / total * 100
                done = int(pct / 2)
                bar  = "#" * done + "-" * (50 - done)
                print(f"\r          [{bar}] {pct:5.1f}%", end="", flush=True)

    if total:
        print()  # newline after progress bar

    print(f"  [ok]    {_human_size(dst)}  →  {dst}")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all-MiniLM-L6-v2 ONNX model and vocabulary."
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files are already present")
    parser.add_argument("--out",   type=str, default="",
                        help="Custom output directory (default: <repo>/data/models)")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve() if args.out else REPO_ROOT / "data" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dst = out_dir / f"{MODEL_NAME}.onnx"
    vocab_dst = out_dir / f"{MODEL_NAME}.vocab.txt"

    print()
    print(f"Downloading {MODEL_NAME}")
    print("=================================")

    download(f"{BASE_URL}/onnx/model.onnx", model_dst, "model  (model.onnx)", args.force)
    download(f"{BASE_URL}/vocab.txt",        vocab_dst, "vocab  (vocab.txt)",  args.force)

    print()
    print("Files")
    print(f"  model : {model_dst}")
    print(f"  vocab : {vocab_dst}")
    print()
    print("Load in C++:")
    print( "  ONNXAddon::Config cfg;")
    print(f'  cfg.model_path = "{model_dst}";')
    print(f'  cfg.vocab_path = "{vocab_dst}";')
    print( "  cfg.use_mean_pooling = true;   // all-MiniLM uses mean pool")
    print( "  addon.load_model(cfg);")
    print()


if __name__ == "__main__":
    main()
