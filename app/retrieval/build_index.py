# app/retrieval/build_index.py
"""CLI entry point for building the ChromaDB vector index from data/corpus.jsonl."""

import argparse
from pathlib import Path
from .load_corpus import build_index

DEFAULT_PERSIST_DIR = Path(__file__).resolve().parents[2] / "chroma"


def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB vector index from corpus.jsonl")
    parser.add_argument(
        "--corpus",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "corpus.jsonl"),
        help="Path to corpus.jsonl",
    )
    parser.add_argument(
        "--persist",
        type=str,
        default=str(DEFAULT_PERSIST_DIR),
        help="Directory to persist the ChromaDB index",
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    persist_dir = Path(args.persist)
    print(f"Building ChromaDB index from {corpus_path} into {persist_dir}...")
    build_index(corpus_path, persist_dir)
    print("Build complete.")


if __name__ == "__main__":
    main()
