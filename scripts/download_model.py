"""
Download Qwen3-30B-A3B-Base weights from HuggingFace.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --local-dir ./models/qwen3-30b-a3b-base
    python scripts/download_model.py --token hf_...  # if needed for gated models
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-30B-A3B-Base"
DEFAULT_LOCAL_DIR = "./models/qwen3-30b-a3b-base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--token", default=None, help="HuggingFace token (optional)")
    args = parser.parse_args()

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MODEL_ID} → {local_dir.resolve()}")
    print("This is ~60 GB. Download resumes automatically if interrupted.\n")

    path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(local_dir),
        token=args.token,
        resume_download=True,
    )

    print(f"\nDone. Weights at: {path}")


if __name__ == "__main__":
    main()
