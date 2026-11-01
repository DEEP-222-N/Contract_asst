"""Download the LegalBench-RAG CUAD files from HuggingFace.

Pulls only what the brief asks for:
  - benchmarks/cuad.json
  - corpus/cuad/*  (462 contract .txt files)

~32 MB on disk.
"""
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "data"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="awinml/legalbench-rag",
        repo_type="dataset",
        local_dir=str(TARGET),
        allow_patterns=["benchmarks/cuad.json", "corpus/cuad/*"],
    )
    print(f"Dataset downloaded to {TARGET}")


if __name__ == "__main__":
    main()
