# tests/test_workflow.py
"""Tests for the multi-agent LangGraph workflow.
Tests normal research, multi-part questions, conflict detection,
same-title complementary chunks, retry limits, invalid citations, and termination.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from app.agents.state import RetrievalChunk, AgentState
from app.agents import workflow
from app.agents.llm import GroundedAnswer, synthesize_answer


def make_chunk(chunk_id: str, text: str, title: str, doc_id: str = "", version: str = "1", distance: float = 0.0):
    return RetrievalChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={"title": title, "doc_id": doc_id or chunk_id.split(":")[0], "version": version},
        distance=distance,
    )


@pytest.fixture(autouse=True)
def patch_research(monkeypatch):
    def fake_research(state: AgentState) -> AgentState:
        state.retrieved = []
        return state
    monkeypatch.setattr("app.agents.research.research_node", fake_research)
    yield


# ---------------------------------------------------------------------------
# 1. Direct question – single search
# ---------------------------------------------------------------------------
def test_direct_question(monkeypatch):
    def stub(state):
        state.retrieved = [make_chunk("c1", "Pricing is tiered.", "Pricing")]
        return state
    monkeypatch.setattr("app.agents.research.research_node", stub)

    result = workflow.run_workflow("What is the pricing model?")
    assert result.answer is not None
    assert "Pricing is tiered" in result.answer
    assert result.citations == ["c1"]
    assert result.verification == "verified"


# ---------------------------------------------------------------------------
# 2. Multi-document question – planner sets need_multiple True & decomposes
# ---------------------------------------------------------------------------
def test_multi_document_question(monkeypatch):
    def stub(state):
        state.retrieved = [
            make_chunk("c1", "Pricing is tiered.", "Pricing"),
            make_chunk("c2", "SLA guarantees 99.9% uptime.", "SLA"),
        ]
        return state
    monkeypatch.setattr("app.agents.research.research_node", stub)

    result = workflow.run_workflow("What is the pricing model and SLA?")
    assert result.need_multiple is True
    assert len(result.subqueries) >= 2
    assert len(result.citations) == 2
    assert result.verification == "verified"


# ---------------------------------------------------------------------------
# 3. Conflicting documents – contradictory claims across versions
# ---------------------------------------------------------------------------
def test_conflict_detection(monkeypatch):
    def stub(state):
        state.retrieved = [
            make_chunk("c1", "Version A of the policy.", "Policy", version="1"),
            make_chunk("c2", "Version B of the policy.", "Policy", version="2"),
        ]
        return state
    monkeypatch.setattr("app.agents.research.research_node", stub)

    result = workflow.run_workflow("Explain the policy.")
    assert result.verification == "conflict"


# ---------------------------------------------------------------------------
# 4. Same-title complementary chunks (Must NOT be flagged as conflict)
# ---------------------------------------------------------------------------
def test_same_title_complementary_chunks(monkeypatch):
    """Verify that multiple chunks from the same document sharing a title do NOT cause false conflict."""
    def stub(state):
        state.retrieved = [
            make_chunk("pricing:0", "Launch costs $59/mo.", "Plans and billing", doc_id="pricing", version="3"),
            make_chunk("pricing:1", "Team costs $449/mo.", "Plans and billing", doc_id="pricing", version="3"),
        ]
        return state
    monkeypatch.setattr("app.agents.research.research_node", stub)

    result = workflow.run_workflow("What are the pricing tiers?")
    assert result.verification == "verified"
    assert "pricing:0" in result.citations
    assert "pricing:1" in result.citations


# ---------------------------------------------------------------------------
# 5. No relevant documents – fallback answer and insufficient verification
# ---------------------------------------------------------------------------
def test_no_documents(monkeypatch):
    def stub(state):
        state.retrieved = []
        return state
    monkeypatch.setattr("app.agents.research.research_node", stub)

    result = workflow.run_workflow("What is the unknown term?")
    assert "could not find" in result.answer.lower()
    assert result.verification == "insufficient"
    assert result.citations == []


# ---------------------------------------------------------------------------
# 6. Retry bounding and deterministic termination
# ---------------------------------------------------------------------------
def test_max_retry_exhaustion(monkeypatch):
    """Verify that retries are strictly bounded by MAX_RETRIES (2) and do not loop infinitely."""
    attempt_count = 0

    def repeatedly_failing_research(state):
        nonlocal attempt_count
        attempt_count += 1
        # Return chunks that trigger conflict
        state.retrieved = [
            make_chunk("c1", "Version A", "Policy", version="1"),
            make_chunk("c2", "Version B", "Policy", version="2"),
        ]
        return state

    monkeypatch.setattr("app.agents.research.research_node", repeatedly_failing_research)
    result = workflow.run_workflow("Test retry limits")
    assert result.verification == "conflict"
    # Should execute initial research + bounded retries (<= 3 research attempts)
    assert attempt_count <= 3
    assert result.retries >= 2


# ---------------------------------------------------------------------------
# 7. Invalid citation detection
# ---------------------------------------------------------------------------
def test_invalid_citation_detection(monkeypatch):
    """Verify that cited chunk IDs not in retrieved chunks are flagged as insufficient."""
    def stub_research(state):
        state.retrieved = [make_chunk("c1", "Passage text", "Title")]
        return state

    def stub_synthesis(state):
        state.answer = "Hallucinated statement."
        state.citations = ["nonexistent_chunk_id"]
        return state

    monkeypatch.setattr("app.agents.research.research_node", stub_research)
    monkeypatch.setattr("app.agents.synthesis.synthesis_node", stub_synthesis)

    result = workflow.run_workflow("Question")
    assert result.verification == "insufficient"


# ---------------------------------------------------------------------------
# 8. LLM synthesis integration test with mock client
# ---------------------------------------------------------------------------
def test_llm_synthesis_with_mock_client():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Asterline Launch costs $59/month according to [pricing:0]."
    mock_client.chat.completions.create.return_value.choices = [mock_choice]

    chunks = [
        make_chunk("pricing:0", "Launch costs $59 per month.", "Plans and billing"),
    ]
    result = synthesize_answer("What does Launch cost?", chunks, client=mock_client)
    assert result.has_sufficient_evidence is True
    assert "pricing:0" in result.citations
    assert "$59" in result.answer
    assert mock_client.chat.completions.create.called
