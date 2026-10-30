# tests/test_corpus_validation.py
"""Tests for the corpus validation utilities."""

import json
import pytest
from pathlib import Path

from app.utils.corpus_validator import load_corpus, validate_corpus, CORPUS_PATH


def test_real_corpus_valid():
    records = load_corpus(CORPUS_PATH)
    valid, msgs = validate_corpus(records)
    assert valid, f"Corpus failed validation: {msgs}"


def test_real_corpus_count():
    records = load_corpus(CORPUS_PATH)
    assert len(records) == 75


def test_no_duplicate_ids():
    records = load_corpus(CORPUS_PATH)
    ids = [r["chunk_id"] for r in records]
    assert len(ids) == len(set(ids))


def test_missing_corpus(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nonexistent.jsonl")


def test_malformed_json(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"valid": 1}\nnot_json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_corpus(bad)


def test_wrong_record_count(tmp_path):
    good = {"chunk_id": "x:0", "doc_id": "x", "title": "T", "category": "c",
            "owner": "o", "source_url": "u", "published": "20260101", "version": "1", "text": "hi"}
    f = tmp_path / "short.jsonl"
    f.write_text(json.dumps(good) + "\n", encoding="utf-8")
    records = load_corpus(f)
    valid, msgs = validate_corpus(records)
    assert not valid
    assert any("75" in m for m in msgs)


def test_missing_required_field(tmp_path):
    rec = {"chunk_id": "x:0", "doc_id": "x"}  # missing many fields
    f = tmp_path / "missing.jsonl"
    f.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    records = load_corpus(f)
    valid, msgs = validate_corpus(records)
    assert not valid


def test_duplicate_chunk_id(tmp_path):
    rec = {"chunk_id": "a:0", "doc_id": "a", "title": "T", "category": "c",
            "owner": "o", "source_url": "u", "published": "20260101", "version": "1", "text": "txt"}
    f = tmp_path / "dup.jsonl"
    f.write_text(json.dumps(rec) + "\n" + json.dumps(rec) + "\n", encoding="utf-8")
    records = load_corpus(f)
    valid, msgs = validate_corpus(records)
    assert not valid
    assert any("Duplicate" in m or "duplicate" in m for m in msgs)
