# app/retrieval/load_corpus.py
"""Corpus loading, validation, and ChromaDB vector indexing utilities."""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None
    embedding_functions = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

REQUIRED_FIELDS = [
    "chunk_id",
    "doc_id",
    "title",
    "category",
    "owner",
    "source_url",
    "published",
    "version",
    "text",
]


def load_corpus(file_path: Path) -> List[Dict]:
    """Load the JSONL corpus file, returning a list of records.
    Empty lines are ignored.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {file_path}")
    records = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON on line {line_num}: {e}")
    return records


def validate_record(record: Dict, line_num: int) -> List[str]:
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"Missing required field '{field}' (line {line_num})")
    return errors


def validate_corpus(records: List[Dict], expected_count: Optional[int] = 75) -> Tuple[bool, List[str]]:
    """Validate the corpus.
    Returns (is_valid, list_of_error_messages).
    Checks record count (default 75), required fields, and duplicate chunk_id.
    """
    errors = []
    if expected_count is not None and len(records) != expected_count:
        errors.append(f"Expected {expected_count} records, found {len(records)}")
    # Validate each record
    for i, rec in enumerate(records, start=1):
        errors.extend(validate_record(rec, i))
    # Duplicate chunk_id check
    chunk_ids = [rec.get("chunk_id") for rec in records]
    duplicates = {cid for cid in chunk_ids if cid and chunk_ids.count(cid) > 1}
    if duplicates:
        errors.append(f"Duplicate chunk_id values found: {', '.join(sorted(duplicates))}")
    return (len(errors) == 0, errors)


class DummyEmbeddingFunction:
    """Fallback zero-vector embedding function when SentenceTransformer is unavailable."""
    def __call__(self, texts):
        return [[0.0] * 384 for _ in texts]


def get_embedding_function():
    """Return configured SentenceTransformer embedding function or safe fallback."""
    if SentenceTransformer and embedding_functions:
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return DummyEmbeddingFunction()


def build_index(corpus_path: Path, persist_dir: Path) -> None:
    """Build or update a ChromaDB vector store from the corpus.
    Uses SentenceTransformer embeddings and stores all metadata fields.
    Idempotent: uses upsert to safely support repeated executions.
    """
    if not chromadb:
        raise RuntimeError("ChromaDB is not installed; cannot build index.")

    records = load_corpus(corpus_path)
    valid, validation_errors = validate_corpus(records)
    if not valid:
        for err in validation_errors:
            print(f"[ERROR] {err}", file=sys.stderr)
        sys.exit(1)

    embed_fn = get_embedding_function()
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(name="asterline_corpus", embedding_function=embed_fn)

    ids = []
    documents = []
    metadatas = []
    for rec in records:
        ids.append(rec["chunk_id"])
        documents.append(rec["text"])
        meta = {k: rec.get(k) for k in REQUIRED_FIELDS if k != "text"}
        metadatas.append(meta)

    # Use upsert for idempotence (safe to run repeatedly)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Index built with {len(ids)} chunks and persisted to {persist_dir}")


def query_index(persist_dir: Path, query: str, top_k: int = 5) -> List[Dict]:
    """Search the index and return matching chunks with metadata."""
    if not chromadb:
        raise RuntimeError("ChromaDB is not installed; cannot query index.")
    if not persist_dir.exists():
        return []

    client = chromadb.PersistentClient(path=str(persist_dir))
    embed_fn = get_embedding_function()
    try:
        collection = client.get_collection(name="asterline_corpus", embedding_function=embed_fn)
    except Exception:
        return []

    results = collection.query(query_texts=[query], n_results=top_k)
    output = []
    if results and "ids" in results and results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            output.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build or query the Asterline corpus index.")
    parser.add_argument("--build", action="store_true", help="Build the index from corpus.jsonl")
    parser.add_argument("--query", type=str, help="Query string to search the index")
    parser.add_argument("--persist", type=str, default="./chroma", help="Directory to store ChromaDB data")
    args = parser.parse_args()
    persist_path = Path(args.persist)
    if args.build:
        corpus_path = Path(__file__).resolve().parents[2] / "data" / "corpus.jsonl"
        build_index(corpus_path, persist_path)
    elif args.query:
        results = query_index(persist_path, args.query)
        for res in results:
            print(f"Chunk ID: {res['chunk_id']}")
            print(f"Title: {res['metadata'].get('title')}")
            print(f"Score (distance): {res['distance']:.4f}")
            print(f"Snippet: {res['text'][:200]}...\n")
    else:
        parser.print_help()
