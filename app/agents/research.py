# app/agents/research.py
"""Research Agent
Performs vector search against the ChromaDB store and returns a list of RetrievalChunk objects.
Supports query decomposition for multi-part questions and respects configured persistence directories.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any, Set

from .state import AgentState, RetrievalChunk
from app.retrieval import query_index, DEFAULT_PERSIST_DIR, _ensure_index

logger = logging.getLogger("asterline.research")


def research_node(state: AgentState) -> AgentState:
    """Execute search against ChromaDB vector store.
    
    Respects state.persist_dir if specified, otherwise uses project DEFAULT_PERSIST_DIR.
    If state.need_multiple is True, queries for each subquery and merges chunks.
    """
    if state.persist_dir:
        persist_path = Path(state.persist_dir)
        if not persist_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            persist_path = project_root / persist_path
    else:
        persist_path = DEFAULT_PERSIST_DIR

    # Auto-ensure index exists if using the default persist directory
    if persist_path == DEFAULT_PERSIST_DIR:
        try:
            _ensure_index(persist_path)
        except Exception as exc:
            logger.warning("Could not auto-ensure default index: %s", exc)

    queries: List[str] = []
    if state.need_multiple and state.subqueries:
        queries = state.subqueries
    else:
        queries = [state.question]

    seen_ids: Set[str] = set()
    all_chunks: List[RetrievalChunk] = []

    for q in queries:
        try:
            results = query_index(persist_path, q, top_k=4)
        except Exception as e:
            logger.warning("Retrieval failed for query %r: %s", q, e)
            results = []

        for r in results:
            cid = r["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                chunk = RetrievalChunk(
                    chunk_id=cid,
                    text=r["text"],
                    metadata=r.get("metadata", {}),
                    distance=float(r.get("distance", 0.0)),
                )
                all_chunks.append(chunk)

    # Sort merged chunks by distance (lower distance = more relevant)
    all_chunks.sort(key=lambda c: c.distance)
    state.retrieved = all_chunks
    return state
