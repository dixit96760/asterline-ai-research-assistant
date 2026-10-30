import os
from pathlib import Path
from typing import List, Dict

from .load_corpus import (
    build_index,
    query_index,
    load_corpus,
    validate_corpus,
    get_embedding_function,
)

# Default directory for the Chroma vector store
DEFAULT_PERSIST_DIR = Path(__file__).resolve().parents[2] / "chroma"


def _ensure_index(persist_dir: Path = DEFAULT_PERSIST_DIR) -> None:
    """Ensure the vector index exists, building it if necessary."""
    if not persist_dir.exists() or not any(persist_dir.iterdir()):
        corpus_path = Path(__file__).resolve().parents[2] / "data" / "corpus.jsonl"
        print(f"Building vector index at {persist_dir} from {corpus_path}")
        build_index(corpus_path, persist_dir)


def retrieve(query: str, top_k: int = 5, persist_dir: Path = DEFAULT_PERSIST_DIR) -> List[Dict]:
    """Retrieve the most relevant document chunks for a query."""
    if not query.strip():
        raise ValueError("Query string must not be empty.")
    _ensure_index(persist_dir)
    results = query_index(persist_dir, query, top_k=top_k)
    return results


def validate_and_load_corpus() -> List[Dict]:
    """Utility to load and validate the corpus, exposing it for other modules."""
    corpus_path = Path(__file__).resolve().parents[2] / "data" / "corpus.jsonl"
    records = load_corpus(corpus_path)
    valid, errors = validate_corpus(records)
    if not valid:
        raise ValueError("Corpus validation failed: " + "; ".join(errors))
    return records


__all__ = [
    "build_index",
    "query_index",
    "load_corpus",
    "validate_corpus",
    "retrieve",
    "DEFAULT_PERSIST_DIR",
    "validate_and_load_corpus",
    "get_embedding_function",
]
