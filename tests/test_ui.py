# tests/test_ui.py
"""Tests for Gradio UI helper functions and layout construction."""

import pytest
from app.ui import (
    build_demo,
    answer_question,
    list_memories,
    add_memory,
    update_memory,
    delete_memory,
    delete_all_memories,
)
from app.memory.memory import _init_db


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    test_db = tmp_path / "ui_test_mem.db"
    _init_db(test_db)
    monkeypatch.setenv("MEMORY_DB_PATH", str(test_db))
    from app.memory import memory
    monkeypatch.setattr(memory, "_DB_PATH", test_db)
    return test_db


def test_build_demo():
    """Verify that the Gradio app builds successfully without layout or syntax errors."""
    demo = build_demo()
    assert demo is not None


def test_answer_question_empty():
    answer, verification, citations = answer_question("")
    assert "Please enter a question" in answer
    assert verification == "insufficient"
    assert citations == []


def test_answer_question_whitespace():
    answer, verification, citations = answer_question("   \n\t  ")
    assert "Please enter a question" in answer
    assert verification == "insufficient"


def test_answer_question_handles_exception(monkeypatch):
    def broken_workflow(q):
        raise RuntimeError("Simulated workflow failure")

    monkeypatch.setattr("app.ui.run_workflow", broken_workflow)
    answer, verification, citations = answer_question("What is the company name?")
    assert "error occurred" in answer.lower()
    assert verification == "error"
    assert citations == []


def test_ui_memory_crud_operations(isolated_db):
    user = "ui_user"

    # Add memory
    msg = add_memory(user, "Important UI note")
    assert "successfully" in msg.lower()

    # List memories
    mems = list_memories(user)
    assert len(mems) == 1
    mem_id = mems[0][0]
    assert mems[0][1] == "Important UI note"

    # Update memory
    msg_upd = update_memory(user, mem_id, "Updated UI note")
    assert "successfully" in msg_upd.lower()

    # Check updated content
    mems = list_memories(user)
    assert mems[0][1] == "Updated UI note"

    # Delete memory
    msg_del = delete_memory(user, mem_id)
    assert "successfully" in msg_del.lower()
    assert len(list_memories(user)) == 0


def test_ui_memory_edge_cases(isolated_db):
    user = "ui_user"

    # Empty content
    assert "Validation error" in add_memory(user, "")
    assert "Validation error" in update_memory(user, 1, "")

    # None IDs
    assert "Validation error" in update_memory(user, None, "New content")
    assert "Validation error" in delete_memory(user, None)

    # Non-existent ID
    assert "not found" in delete_memory(user, 999999)
    assert "not found" in update_memory(user, 999999, "Content")

    # Empty user list
    assert list_memories("") == []
