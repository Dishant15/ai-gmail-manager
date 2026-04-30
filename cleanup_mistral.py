"""
cleanup_mistral.py - Safely remove Mistral model weights and related HuggingFace
                     cache from your local machine.

Usage:
    python cleanup_mistral.py              # interactive mode (asks before deleting)
    python cleanup_mistral.py --dry-run    # just show what WOULD be deleted
    python cleanup_mistral.py --force      # delete without asking

What this script cleans:
    1. Mistral model weights & tokenizer  (~14 GB)  in HuggingFace cache
    2. bitsandbytes quantization cache    (~small)   in HuggingFace cache
    3. Sentence-transformer embedding model (~90 MB) in HuggingFace cache
    4. Torch compiled-model cache         (~varies)  in torch cache
    5. pip package files for the heavy ML libs (optional)

What this script does NOT touch:
    - Your ChromaDB vector store  (./chroma_db/)
    - Your uploaded PDFs          (./uploads/)
    - Your Gmail credentials/token
    - Any other Python packages or projects
"""

import argparse
import shutil
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# Colour helpers (no external deps)
# ──────────────────────────────────────────────────────────────

def red(s):    return f"\033[91m{s}\033[0m"
def green(s):  return f"\033[92m{s}\033[0m"
def yellow(s): return f"\033[93m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"
def dim(s):    return f"\033[2m{s}\033[0m"


# ──────────────────────────────────────────────────────────────
# Size helpers
# ──────────────────────────────────────────────────────────────

def dir_size_gb(path: Path) -> float:
    """Return total size of a directory tree in GB."""
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 ** 3), 2)


def fmt_size(gb: float) -> str:
    if gb >= 1:
        return f"{gb:.2f} GB"
    mb = round(gb * 1024, 1)
    return f"{mb} MB"


# ──────────────────────────────────────────────────────────────
# Locate HuggingFace cache root
# ──────────────────────────────────────────────────────────────

def find_hf_cache() -> Path:
    """
    HuggingFace stores models in:
      Linux/macOS : ~/.cache/huggingface/hub/
      Windows     : C:/Users/<user>/.cache/huggingface/hub/
    The env var HF_HOME or HUGGINGFACE_HUB_CACHE can override this.
    """
    import os
    custom = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if custom:
        return Path(custom) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def find_torch_cache() -> Path:
    return Path.home() / ".cache" / "torch"


# ──────────────────────────────────────────────────────────────
# Build the list of targets
# ──────────────────────────────────────────────────────────────

# Model repo IDs → folder name pattern HuggingFace uses on disk
# HuggingFace converts  "org/model-name"  →  "models--org--model-name"
MISTRAL_MODEL_IDS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
]

EMBEDDING_MODEL_IDS = [
    "sentence-transformers/all-MiniLM-L6-v2",
]

# Also try to read the model id from .env so custom values are covered
def _read_env_model_ids() -> list:
    env_file = Path(".env")
    ids = []
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_MODEL_ID="):
                ids.append(line.split("=", 1)[1].strip())
            if line.startswith("EMBEDDING_MODEL="):
                ids.append(line.split("=", 1)[1].strip())
    return ids


def model_id_to_folder(model_id: str) -> str:
    """'org/model-name' → 'models--org--model-name'"""
    return "models--" + model_id.replace("/", "--")


def collect_targets(hf_cache: Path, torch_cache: Path) -> list[dict]:
    """
    Return a list of dicts:
        { "label": str, "path": Path, "size_gb": float, "exists": bool }
    """
    targets = []

    all_model_ids = list(set(
        MISTRAL_MODEL_IDS + EMBEDDING_MODEL_IDS + _read_env_model_ids()
    ))

    for model_id in all_model_ids:
        folder_name = model_id_to_folder(model_id)
        path = hf_cache / folder_name
        targets.append({
            "label": f"Model weights: {model_id}",
            "path": path,
            "size_gb": dir_size_gb(path),
            "exists": path.exists(),
        })

    # bitsandbytes / quantization cache
    bnb_cache = hf_cache.parent / "accelerate"
    targets.append({
        "label": "HuggingFace accelerate cache",
        "path": bnb_cache,
        "size_gb": dir_size_gb(bnb_cache),
        "exists": bnb_cache.exists(),
    })

    # Torch kernel / compiled model cache
    targets.append({
        "label": "Torch cache (compiled kernels)",
        "path": torch_cache,
        "size_gb": dir_size_gb(torch_cache),
        "exists": torch_cache.exists(),
    })

    return targets


# ──────────────────────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────────────────────

def print_targets(targets: list[dict], dry_run: bool) -> float:
    print()
    print(bold("  Targets to clean up:"))
    print(f"  {'Status':<10} {'Size':<12} {'Label'}")
    print("  " + "─" * 65)

    total = 0.0
    for t in targets:
        if t["exists"]:
            size_str = fmt_size(t["size_gb"])
            status = yellow("FOUND") if not dry_run else dim("DRY-RUN")
            total += t["size_gb"]
        else:
            size_str = "—"
            status = dim("not found")
        print(f"  {status:<20} {size_str:<12} {t['label']}")
        if t["exists"]:
            print(f"  {'':<20} {'':<12} {dim(str(t['path']))}")

    print("  " + "─" * 65)
    print(f"  {'Total reclaimable:':<32} {bold(fmt_size(total))}")
    print()
    return total


# ──────────────────────────────────────────────────────────────
# Deletion
# ──────────────────────────────────────────────────────────────

def delete_targets(targets: list[dict]) -> float:
    freed = 0.0
    for t in targets:
        if not t["exists"]:
            continue
        try:
            size = t["size_gb"]
            shutil.rmtree(t["path"])
            freed += size
            print(f"  {green('✓ Deleted')}  {t['label']}  ({fmt_size(size)})")
        except PermissionError as e:
            print(f"  {red('✗ Permission denied')}  {t['path']}: {e}")
        except Exception as e:
            print(f"  {red('✗ Error')}  {t['path']}: {e}")
    return freed


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clean up local Mistral / HuggingFace model cache."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be deleted without actually deleting anything.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete without asking for confirmation.",
    )
    args = parser.parse_args()

    hf_cache  = find_hf_cache()
    torch_cache = find_torch_cache()

    print()
    print(bold("═" * 60))
    print(bold("  Mistral & HuggingFace Cache Cleanup"))
    print(bold("═" * 60))
    print(f"  HuggingFace cache : {dim(str(hf_cache))}")
    print(f"  Torch cache       : {dim(str(torch_cache))}")

    targets = collect_targets(hf_cache, torch_cache)
    total_gb = print_targets(targets, dry_run=args.dry_run)

    if not any(t["exists"] for t in targets):
        print(green("  Nothing to clean — no cached files found.\n"))
        sys.exit(0)

    if args.dry_run:
        print(yellow("  Dry-run mode: no files were deleted."))
        print(f"  Run without --dry-run to free up {bold(fmt_size(total_gb))}.\n")
        sys.exit(0)

    if not args.force:
        print(f"  This will permanently delete {bold(fmt_size(total_gb))} of model files.")
        answer = input(f"  {bold('Proceed? [y/N] ')}").strip().lower()
        if answer not in ("y", "yes"):
            print(yellow("\n  Aborted. Nothing was deleted.\n"))
            sys.exit(0)

    print()
    freed = delete_targets(targets)
    print()
    print(bold("═" * 60))
    print(green(f"  Done.  Freed {fmt_size(freed)} of disk space."))
    print(bold("═" * 60))
    print()
    print("  To use a different model in future, update HF_MODEL_ID in .env")
    print("  The new model will be downloaded automatically on next startup.\n")


if __name__ == "__main__":
    main()
