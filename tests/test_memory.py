# tests/test_memory.py
"""Tests for the memory module. Uses a temporary SQLite DB so real user data is never touched."""

import pytest
from pathlib import Path
from app.memory.memory import MemoryStore, _init_db


@pytest.fixture
def tmp_db(tmp_path):
    db = tmp_path / "test_memory.db"
    _init_db(db)
    return db


def test_add_and_list(tmp_db):
    MemoryStore.add_memory("user1", "Remember to check SLA docs.", db_path=tmp_db)
    rows = MemoryStore.list_memories("user1", db_path=tmp_db)
    assert len(rows) == 1
    assert rows[0][1] == "Remember to check SLA docs."


def test_list_empty(tmp_db):
    rows = MemoryStore.list_memories("unknown_user", db_path=tmp_db)
    assert rows == []


def test_update(tmp_db):
    MemoryStore.add_memory("user1", "Old content.", db_path=tmp_db)
    mem_id = MemoryStore.list_memories("user1", db_path=tmp_db)[0][0]
    updated = MemoryStore.update_memory("user1", mem_id, "New content.", db_path=tmp_db)
    assert updated is True
    rows = MemoryStore.list_memories("user1", db_path=tmp_db)
    assert rows[0][1] == "New content."


def test_update_wrong_user(tmp_db):
    MemoryStore.add_memory("user1", "Content.", db_path=tmp_db)
    mem_id = MemoryStore.list_memories("user1", db_path=tmp_db)[0][0]
    updated = MemoryStore.update_memory("user2", mem_id, "Hacked.", db_path=tmp_db)
    assert updated is False
    # user1 data is unchanged
    rows = MemoryStore.list_memories("user1", db_path=tmp_db)
    assert rows[0][1] == "Content."


def test_delete(tmp_db):
    MemoryStore.add_memory("user1", "Temporary.", db_path=tmp_db)
    mem_id = MemoryStore.list_memories("user1", db_path=tmp_db)[0][0]
    deleted = MemoryStore.delete_memory("user1", mem_id, db_path=tmp_db)
    assert deleted is True
    assert MemoryStore.list_memories("user1", db_path=tmp_db) == []


def test_delete_wrong_user(tmp_db):
    MemoryStore.add_memory("user1", "Mine.", db_path=tmp_db)
    mem_id = MemoryStore.list_memories("user1", db_path=tmp_db)[0][0]
    deleted = MemoryStore.delete_memory("user2", mem_id, db_path=tmp_db)
    assert deleted is False
    assert len(MemoryStore.list_memories("user1", db_path=tmp_db)) == 1


def test_delete_all(tmp_db):
    MemoryStore.add_memory("user1", "A.", db_path=tmp_db)
    MemoryStore.add_memory("user1", "B.", db_path=tmp_db)
    count = MemoryStore.delete_all("user1", db_path=tmp_db)
    assert count == 2
    assert MemoryStore.list_memories("user1", db_path=tmp_db) == []


def test_user_isolation(tmp_db):
    MemoryStore.add_memory("user1", "User1 note.", db_path=tmp_db)
    MemoryStore.add_memory("user2", "User2 note.", db_path=tmp_db)
    u1 = MemoryStore.list_memories("user1", db_path=tmp_db)
    u2 = MemoryStore.list_memories("user2", db_path=tmp_db)
    assert len(u1) == 1 and "User1" in u1[0][1]
    assert len(u2) == 1 and "User2" in u2[0][1]


def test_persistence(tmp_db):
    """Simulates a process restart by creating a second MemoryStore instance using the same DB."""
    MemoryStore.add_memory("user1", "Persistent note.", db_path=tmp_db)
    # Re-init (simulates restart)
    _init_db(tmp_db)
    rows = MemoryStore.list_memories("user1", db_path=tmp_db)
    assert rows[0][1] == "Persistent note."


def test_reject_empty_content(tmp_db):
    with pytest.raises(ValueError, match="cannot be empty"):
        MemoryStore.add_memory("user1", "", db_path=tmp_db)


def test_reject_secret(tmp_db):
    with pytest.raises(ValueError, match="secret"):
        MemoryStore.add_memory("user1", "my api_key is abc123", db_path=tmp_db)
