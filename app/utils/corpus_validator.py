import json
from pathlib import Path
from typing import List, Tuple

CORPUS_PATH = Path(__file__).resolve().parents[2] / "data" / "corpus.jsonl"

def load_corpus(path: Path = CORPUS_PATH) -> List[dict]:
    """Load the JSONL corpus file.

    Returns a list of JSON objects. Raises ``ValueError`` if the file cannot be read
    or contains malformed JSON.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    return records

def validate_corpus(records: List[dict], expected_count: int | None = 75) -> Tuple[bool, List[str]]:
    """Validate the corpus structure.

    Checks performed:
    1. Expected number of records (75 by default).
    2. Each record contains required keys: ``chunk_id``, ``doc_id``, ``title``,
       ``category``, ``owner``, ``source_url``, ``published``, ``version``, ``text``.
    3. No duplicate ``chunk_id`` values.
    4. ``published`` is an 8-digit string (YYYYMMDD).
    5. ``version`` is a non-empty string.
    Returns a tuple ``(is_valid, messages)`` where ``messages`` contains any
    validation errors.
    """
    messages: List[str] = []
    required_keys = {
        "chunk_id",
        "doc_id",
        "title",
        "category",
        "owner",
        "source_url",
        "published",
        "version",
        "text",
    }

    if expected_count is not None and len(records) != expected_count:
        messages.append(f"Expected {expected_count} records, found {len(records)}.")

    seen_ids = set()
    for idx, rec in enumerate(records, start=1):
        missing = required_keys - rec.keys()
        if missing:
            messages.append(f"Record {idx} missing keys: {', '.join(sorted(missing))}.")
        # duplicate chunk_id
        cid = rec.get("chunk_id")
        if cid:
            if cid in seen_ids:
                messages.append(f"Duplicate chunk_id '{cid}' found at record {idx}.")
            else:
                seen_ids.add(cid)
        # published format
        pub = rec.get("published", "")
        if not (isinstance(pub, str) and len(pub) == 8 and pub.isdigit()):
            messages.append(f"Record {idx} has invalid 'published' value: {pub!r}.")
        # version non-empty
        ver = rec.get("version", "")
        if not (isinstance(ver, str) and ver):
            messages.append(f"Record {idx} has empty 'version' field.")
        # text non-empty
        txt = rec.get("text", "")
        if not (isinstance(txt, str) and txt.strip()):
            messages.append(f"Record {idx} has empty 'text' field.")

    is_valid = not messages
    return is_valid, messages

def main() -> None:
    try:
        records = load_corpus()
    except Exception as exc:
        print(f"Error loading corpus: {exc}")
        return
    valid, msgs = validate_corpus(records)
    if valid:
        print(f"Corpus validation passed: all {len(records)} records are well-formed.")
    else:
        print("Corpus validation failed with the following issues:")
        for m in msgs:
            print(f"- {m}")

if __name__ == "__main__":
    main()
