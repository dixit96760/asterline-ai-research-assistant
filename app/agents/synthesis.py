# app/agents/synthesis.py
"""Synthesis Agent
Generates an evidence-grounded answer from retrieved chunks and records citations using the LLM provider.
"""

from __future__ import annotations

from typing import List, Optional, Any
from .state import AgentState
from .llm import synthesize_answer


def synthesis_node(state: AgentState, client: Optional[Any] = None) -> AgentState:
    """Generate an evidence-grounded answer.
    
    Uses LLM synthesis when configured, with safe offline fallback.
    Records citations corresponding to the referenced document chunks.
    """
    if not state.retrieved:
        state.answer = "I could not find relevant information to answer the question."
        state.citations = []
        return state

    result = synthesize_answer(state.question, state.retrieved, client=client)
    state.answer = result.answer
    state.citations = result.citations
    return state
