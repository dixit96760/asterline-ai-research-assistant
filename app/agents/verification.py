# app/agents/verification.py
"""Verification Agent
Validates citations, detects insufficient evidence, and checks for genuine evidence conflicts.
Distinguishes between ordinary multi-chunk documents and true contradictions.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Set, Any
from .state import AgentState, RetrievalChunk


def _are_complementary_chunks(chunk_a: RetrievalChunk, chunk_b: RetrievalChunk) -> bool:
    """Determine if two chunks are normal consecutive parts of the same document."""
    doc_a = chunk_a.metadata.get("doc_id")
    doc_b = chunk_b.metadata.get("doc_id")
    ver_a = chunk_a.metadata.get("version")
    ver_b = chunk_b.metadata.get("version")

    # If both have matching doc_id and matching version, they belong to the same document
    if doc_a and doc_b and doc_a == doc_b:
        if ver_a and ver_b and ver_a == ver_b:
            return True
        if not ver_a and not ver_b:
            return True
        return False  # Same doc but different versions is a potential conflict

    # If chunk IDs have the same prefix before colon (e.g. pricing:0 and pricing:1)
    if ":" in chunk_a.chunk_id and ":" in chunk_b.chunk_id:
        prefix_a = chunk_a.chunk_id.split(":", 1)[0]
        prefix_b = chunk_b.chunk_id.split(":", 1)[0]
        if prefix_a == prefix_b:
            if ver_a and ver_b and ver_a != ver_b:
                return False
            return True

    return False


def _detect_conflicts(chunks: List[RetrievalChunk]) -> bool:
    """Check for genuine conflicts or contradictions across retrieved chunks.
    
    Normal multi-chunk passages from the same document are not considered conflicts.
    A conflict is flagged when:
    - Passages for the same topic/title have conflicting version numbers.
    - Independent documents with the same title contain contradictory statements (e.g. Version A vs Version B).
    """
    if len(chunks) < 2:
        return False

    # Group by title
    by_title: Dict[str, List[RetrievalChunk]] = defaultdict(list)
    for c in chunks:
        title = c.metadata.get("title", "").strip().lower()
        by_title[title].append(c)

    for title, group in by_title.items():
        if len(group) < 2:
            continue

        # Check version collisions within the same title
        versions = {c.metadata.get("version") for c in group if c.metadata.get("version")}
        if len(versions) > 1:
            return True

        # Check pairs in group
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                ca = group[i]
                cb = group[j]

                # If they are complementary chunks of the same document, not a conflict
                if _are_complementary_chunks(ca, cb):
                    continue

                text_a = ca.text.strip().lower()
                text_b = cb.text.strip().lower()
                if text_a == text_b:
                    continue  # Exact duplicates are not conflicts

                # Check for explicit version contradictions (e.g. "version a" vs "version b")
                va = re.search(r"version\s+([a-z0-9.]+)", text_a)
                vb = re.search(r"version\s+([a-z0-9.]+)", text_b)
                if va and vb and va.group(1) != vb.group(1):
                    return True

                # Different documents sharing the exact same title with different content
                doc_a = ca.metadata.get("doc_id")
                doc_b = cb.metadata.get("doc_id")
                if doc_a != doc_b or not doc_a:
                    return True

    return False


def verification_node(state: AgentState) -> AgentState:
    """Validate that all cited chunk IDs exist and detect genuine evidence conflicts.

    - Empty retrieval -> 'insufficient' and terminates retry loop.
    - Missing citations -> 'insufficient' and increments retry count.
    - Genuine conflicts -> 'conflict'.
    - Otherwise -> 'verified'.
    """
    # If no chunks were retrieved at all, evidence is insufficient; do not retry indefinitely
    if not state.retrieved:
        state.verification = "insufficient"
        state.retries = 99  # Terminal state: cannot recover from empty retrieval
        return state

    # Check for cited chunk IDs that were never retrieved
    retrieved_ids = {c.chunk_id for c in state.retrieved}
    missing = [cid for cid in state.citations if cid not in retrieved_ids]
    if missing:
        state.verification = "insufficient"
        state.retries += 1
        return state

    # Check for evidence conflicts
    if _detect_conflicts(state.retrieved):
        state.verification = "conflict"
        state.retries += 1
        return state

    # All checks passed
    state.verification = "verified"
    return state
