#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# scripts/download_models.py
# Manage ONNX models for NLPEngine (check status and download).
#
# Models used by NLPEngine (data/models/):
#
#   embed.onnx      Sentence embeddings — all-MiniLM-L6-v2 (required)
#                   384 dims · ~22 MB · Apache-2.0
#
#   vocab.txt       Shared WordPiece vocabulary (required for tokeniser)
#                   Works with all BERT-family models above.
#
#   sentiment.onnx  Sentiment classifier — DistilBERT SST-2 (optional)
#                   Labels: NEGATIVE (0) · POSITIVE (1)
#
#   ner.onnx        Named-entity recognition — bert-base-NER (optional)
#                   CoNLL-2003 BIO labels: PER · ORG · LOC · MISC
#
#   toxicity.onnx   Multi-label toxicity — toxic-bert (optional)
#                   Labels: toxic · severe_toxic · obscene · threat ·
#                           insult · identity_hate   (sigmoid activation)
#
# Usage:
#   python3 scripts/download_models.py check
#   python3 scripts/download_models.py download
#   python3 scripts/download_models.py download --models embed,vocab
#   python3 scripts/download_models.py download --models sentiment,ner,toxicity
#   python3 scripts/download_models.py download --force
#   python3 scripts/download_models.py download --out /custom/dir
#
# Backward-compat (no sub-command = download all required models):
#   python3 scripts/download_models.py [--force] [--out <dir>]
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("error: 'requests' not found.  pip install requests", file=sys.stderr)
    sys.exit(1)


# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
HF         = "https://huggingface.co"

# ─── Model catalogue ─────────────────────────────────────────────────────────
#
# Each entry:
#   filename    : destination file name in the models directory
#   url         : direct HuggingFace download URL (resolve/main/…)
#   size_hint   : approximate size string shown in the check table
#   required    : printed as [required] vs [optional] in check output
#   notes       : extra information printed in download guidance
#
# Xenova models (https://huggingface.co/Xenova) are the community-maintained
# Optimum-exported ONNX versions of popular HuggingFace models and are
# consistently available as flat onnx/model.onnx files.
# ─────────────────────────────────────────────────────────────────────────────

_EMBED_BASE  = f"{HF}/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
_SENT_BASE   = f"{HF}/Xenova/distilbert-base-uncased-finetuned-sst-2-english/resolve/main"
_NER_BASE    = f"{HF}/Xenova/bert-base-NER/resolve/main"
_TOX_BASE    = f"{HF}/Xenova/toxic-bert/resolve/main"

MODELS: dict[str, dict] = {
    # ── Required ──────────────────────────────────────────────────────────────
    "embed": {
        "filename":  "embed.onnx",
        "url":       f"{_EMBED_BASE}/onnx/model.onnx",
        "size_hint": "~22 MB",
        "required":  True,
        "notes":     "sentence-transformers/all-MiniLM-L6-v2 · 384 dims · Apache-2.0",
    },
    "vocab": {
        "filename":  "vocab.txt",
        "url":       f"{_EMBED_BASE}/vocab.txt",
        "size_hint": "~230 KB",
        "required":  True,
        "notes":     "Shared WordPiece vocabulary (BERT-family tokeniser)",
    },
    # ── Optional neural classifiers ───────────────────────────────────────────
    "sentiment": {
        "filename":  "sentiment.onnx",
        "url":       f"{_SENT_BASE}/onnx/model.onnx",
        "size_hint": "~68 MB",
        "required":  False,
        "notes":     (
            "Xenova/distilbert-base-uncased-finetuned-sst-2-english\n"
            "          Labels (id2label): NEGATIVE(0) POSITIVE(1)\n"
            "          Fallback: lexicon-based heuristic"
        ),
    },
    "ner": {
        "filename":  "ner.onnx",
        "url":       f"{_NER_BASE}/onnx/model.onnx",
        "size_hint": "~420 MB",
        "required":  False,
        "notes":     (
            "Xenova/bert-base-NER (CoNLL-2003)\n"
            "          BIO labels: O B-PER I-PER B-ORG I-ORG B-LOC I-LOC B-MISC I-MISC\n"
            "          Fallback: regex + capitalisation heuristics"
        ),
    },
    "toxicity": {
        "filename":  "toxicity.onnx",
        "url":       f"{_TOX_BASE}/onnx/model.onnx",
        "size_hint": "~440 MB",
        "required":  False,
        "notes":     (
            "Xenova/toxic-bert (multi-label, sigmoid activation)\n"
            "          Labels: toxic severe_toxic obscene threat insult identity_hate\n"
            "          Fallback: word-list / pattern matching\n"
            "          If the URL is unavailable, export manually:\n"
            "            pip install optimum[onnxruntime]\n"
            "            optimum-cli export onnx --model unitary/toxic-bert toxicity_dir/\n"
            "            cp toxicity_dir/model.onnx data/models/toxicity.onnx"
        ),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} GB"


def _resolve_out_dir(raw: str) -> Path:
    if raw:
        return Path(raw).resolve()
    return REPO_ROOT / "data" / "models"


def _parse_model_keys(raw: str) -> list[str]:
    """Parse a comma-separated model key string; validate against MODELS."""
    keys = [k.strip().lower() for k in raw.split(",") if k.strip()]
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        print(
            f"error: unknown model(s): {', '.join(unknown)}\n"
            f"       valid keys: {', '.join(MODELS)}",
            file=sys.stderr,
        )
        sys.exit(1)
    return keys


# ─────────────────────────────────────────────────────────────────────────────
# Download helper
# ─────────────────────────────────────────────────────────────────────────────

def _download_file(url: str, dst: Path, label: str, force: bool = False) -> bool:
    """
    Download *url* to *dst*.  Returns True on success, False on failure.
    Prints a skip message when the file already exists and *force* is False.
    """
    if dst.exists() and not force:
        print(f"  [skip]  {label}  ({_human_size(dst.stat().st_size)} on disk"
              " — pass --force to re-download)")
        return True

    print(f"  [fetch] {label}")
    print(f"          {url}")

    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"\n  [fail]  {label}: {exc}", file=sys.stderr)
        return False

    total    = int(resp.headers.get("content-length", 0))
    received = 0

    try:
        with dst.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=65_536):
                fh.write(chunk)
                received += len(chunk)
                if total:
                    pct  = received / total * 100
                    done = int(pct / 2)
                    bar  = "#" * done + "-" * (50 - done)
                    print(f"\r          [{bar}] {pct:5.1f}%", end="", flush=True)
    except OSError as exc:
        print(f"\n  [fail]  {label}: {exc}", file=sys.stderr)
        return False

    if total:
        print()   # newline after progress bar

    print(f"  [ok]    {_human_size(dst.stat().st_size)}  →  {dst}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command: check
# ─────────────────────────────────────────────────────────────────────────────

def cmd_check(out_dir: Path) -> None:
    """Print a table showing which model files are present or missing."""
    col_w = max(len(m["filename"]) for m in MODELS.values()) + 2

    print()
    print("NLPEngine model status")
    print(f"  directory: {out_dir}")
    print()
    print(f"  {'File':<{col_w}} {'Required':<10} {'Status'}")
    print(f"  {'-' * col_w} {'-'*9} {'-'*30}")

    all_required_ok = True

    for key, meta in MODELS.items():
        path     = out_dir / meta["filename"]
        required = "required" if meta["required"] else "optional"
        if path.exists():
            status = f"OK  {_human_size(path.stat().st_size)}"
        else:
            status = "MISSING"
            if meta["required"]:
                all_required_ok = False

        print(f"  {meta['filename']:<{col_w}} {required:<10} {status}")

    print()
    if all_required_ok:
        print("  All required models present.")
    else:
        print("  Some required models are missing.")
        print("  Run:  python3 scripts/download_models.py download")

    print()
    _print_cpp_usage(out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-command: download
# ─────────────────────────────────────────────────────────────────────────────

def cmd_download(out_dir: Path, keys: list[str], force: bool) -> None:
    """Download the selected models to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("Downloading NLPEngine models")
    print(f"  directory : {out_dir}")
    print(f"  models    : {', '.join(keys)}")
    if force:
        print("  --force   : existing files will be re-downloaded")
    print()

    failed: list[str] = []

    for key in keys:
        meta = MODELS[key]
        dst  = out_dir / meta["filename"]
        ok   = _download_file(meta["url"], dst, meta["filename"], force)
        if not ok:
            failed.append(key)
            # Print guidance when a URL fails
            print(f"\n  NOTE  ({key}): {meta['notes']}\n", file=sys.stderr)

    print()
    if failed:
        print(f"  WARNING: {len(failed)} model(s) failed to download: {', '.join(failed)}")
        print("           See messages above for manual export instructions.")
    else:
        print("  All requested models downloaded successfully.")

    print()
    _print_cpp_usage(out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Usage hint
# ─────────────────────────────────────────────────────────────────────────────

def _print_cpp_usage(out_dir: Path) -> None:
    embed    = out_dir / "embed.onnx"
    vocab    = out_dir / "vocab.txt"
    sent     = out_dir / "sentiment.onnx"
    tox      = out_dir / "toxicity.onnx"
    ner_path = out_dir / "ner.onnx"

    print("C++ / main.cc wiring:")
    print()
    print("  // Resolve model directory from env or default:")
    print('  fs::path model_dir = std::getenv("NLP_MODEL_DIR")')
    print('                     ? std::getenv("NLP_MODEL_DIR")')
    print(f'                     : "{out_dir}";')
    print()
    print("  // Engine will load each file that exists at runtime:")
    print(f'  //   embed.onnx     → {embed}')
    print(f'  //   vocab.txt      → {vocab}')
    print(f'  //   sentiment.onnx → {sent}')
    print(f'  //   toxicity.onnx  → {tox}')
    print(f'  //   ner.onnx       → {ner_path}')
    print()
    print("  // Capability flags injected into the webview:")
    print("  //   window.__nlp.hasOnnx      (embed.onnx present)")
    print("  //   window.__nlp.hasSentiment (sentiment.onnx present)")
    print("  //   window.__nlp.hasToxicity  (toxicity.onnx present)")
    print("  //   window.__nlp.hasNer       (ner.onnx present)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_models.py",
        description=(
            "Manage ONNX model files for NLPEngine.\n\n"
            "Sub-commands:\n"
            "  check     Show which model files are present in the models directory.\n"
            "  download  Download one or more models (default: embed + vocab).\n\n"
            "Examples:\n"
            "  python3 scripts/download_models.py check\n"
            "  python3 scripts/download_models.py download\n"
            "  python3 scripts/download_models.py download --models embed,vocab\n"
            "  python3 scripts/download_models.py download --models sentiment,ner,toxicity\n"
            "  python3 scripts/download_models.py download --force\n"
            "  python3 scripts/download_models.py download --out /custom/dir\n\n"
            f"Available model keys: {', '.join(MODELS)}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global option (backward-compat — allows plain --out / --force with no sub-command)
    parser.add_argument(
        "--out", type=str, default="",
        help="Custom output directory (default: <repo>/data/models)"
    )

    subs = parser.add_subparsers(dest="command")

    # ── check ─────────────────────────────────────────────────────────────────
    subs.add_parser(
        "check",
        help="Show which model files are present / missing.",
    )

    # ── download ──────────────────────────────────────────────────────────────
    dl = subs.add_parser(
        "download",
        help="Download model files.",
    )
    dl.add_argument(
        "--models", type=str,
        default="embed,vocab",
        help=(
            "Comma-separated list of model keys to download "
            f"(default: embed,vocab).  "
            f"All models: {', '.join(MODELS)}"
        ),
    )
    dl.add_argument(
        "--force", action="store_true",
        help="Re-download even if the file already exists on disk.",
    )
    dl.add_argument(
        "--out", type=str, default="",
        help="Custom output directory (default: <repo>/data/models)",
    )

    # Backward-compat flags on the top-level parser
    parser.add_argument(
        "--force", action="store_true",
        help="(legacy) Same as: download --force",
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # Resolve output directory: sub-command --out takes priority, then global --out
    sub_out  = getattr(args, "out", "")  # may come from sub-command or top-level
    out_dir  = _resolve_out_dir(sub_out)

    if args.command == "check":
        cmd_check(out_dir)

    elif args.command == "download":
        keys = _parse_model_keys(args.models)
        cmd_download(out_dir, keys, force=args.force)

    else:
        # No sub-command → legacy behaviour: download embed + vocab
        force = getattr(args, "force", False)
        print(
            "\n[info] No sub-command given — downloading required models (embed, vocab).\n"
            "       For full control use:  download_models.py download --models <keys>\n"
        )
        cmd_download(out_dir, ["embed", "vocab"], force=force)


if __name__ == "__main__":
    main()
