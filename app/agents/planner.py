# app/agents/planner.py
"""Planner Agent
Analyzes the user question, determines whether multi-part retrieval is required,
and decomposes composite questions into targeted subqueries.
"""

from __future__ import annotations

import re
from typing import List
from .state import AgentState


def _decompose_query(question: str) -> List[str]:
    """Decompose compound questions into subqueries while preserving meaning."""
    # Split by conjunctions: ' and ', ' vs ', ' versus ', or comma lists
    pattern = r"\s+(?:and|vs|versus)\s+|,\s*"
    parts = re.split(pattern, question, flags=re.IGNORECASE)
    cleaned = [p.strip(" ?.,") for p in parts if p.strip(" ?.,")]
    if len(cleaned) <= 1:
        return [question.strip()]

    # Format into subquestions if they are fragments
    subqueries = []
    prefix = ""
    lower_q = question.lower().strip()
    if lower_q.startswith("what is the "):
        prefix = "What is the "
    elif lower_q.startswith("what are the "):
        prefix = "What are the "
    elif lower_q.startswith("describe the "):
        prefix = "Describe the "

    for part in cleaned:
        if not part.endswith("?"):
            part_q = part + "?"
        else:
            part_q = part
        if prefix and not part_q.lower().startswith(("what", "describe", "how", "why", "when", "where")):
            part_q = prefix + part_q
        subqueries.append(part_q)

    return subqueries


def planner_node(state: AgentState) -> AgentState:
    """Planner node: sets need_multiple flag and populates decomposed subqueries."""
    q = state.question.lower()
    split_tokens = [" and ", " or ", ", ", " vs ", " versus "]
    if any(tok in q for tok in split_tokens):
        state.need_multiple = True
        state.subqueries = _decompose_query(state.question)
    else:
        state.need_multiple = False
        state.subqueries = [state.question.strip()]
    return state
