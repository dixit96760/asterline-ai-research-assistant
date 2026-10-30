# app/agents/state.py
"""Shared state models for the multi-agent LangGraph workflow.
Defines the Pydantic models used by all agents.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class RetrievalChunk(BaseModel):
    """A single chunk retrieved from the ChromaDB vector store.

    Attributes:
        chunk_id: Unique identifier of the chunk (matches corpus `chunk_id`).
        text: The raw text content of the chunk.
        metadata: Dictionary containing additional fields such as title, category, etc.
        distance: The similarity distance returned by the vector store (lower is more similar).
    """

    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Chunk text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata for citation")
    distance: float = Field(0.0, description="Similarity distance from query")


class AgentState(BaseModel):
    """State shared across all agents in the workflow.

    Fields are accessible and mutable by workflow nodes.
    """

    question: str = Field(..., description="User question")
    need_multiple: bool = Field(False, description="Planner flag for multiple searches")
    subqueries: List[str] = Field(default_factory=list, description="Decomposed subqueries if need_multiple is True")
    retrieved: List[RetrievalChunk] = Field(default_factory=list, description="Chunks returned by research")
    answer: str = Field("", description="Generated answer from synthesis")
    citations: List[str] = Field(default_factory=list, description="List of cited chunk IDs")
    verification: Optional[str] = Field(None, description="Verification status: verified, conflict, insufficient")
    retries: int = Field(0, description="Retry counter for verification failures")
    persist_dir: Optional[str] = Field(None, description="Optional ChromaDB persistence directory path")
