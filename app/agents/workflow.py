# app/agents/workflow.py
"""LangGraph workflow that orchestrates Planner, Research, Synthesis, and Verification agents.
The workflow operates on the shared :class:`AgentState` model defined in ``state.py``.
Supports runtime dependency injection, configurable persistence paths, and tracing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable, Any

from langgraph.graph import StateGraph, END

from .state import AgentState
from .planner import planner_node
from .research import research_node
from .synthesis import synthesis_node
from .verification import verification_node
from app.tracing import traced

MAX_RETRIES = 2


def _route_after_verification(state: AgentState) -> str:
    """Conditional edge: retry research+synthesis if verification failed and
    retry budget remains; otherwise terminate at END."""
    # Terminal: cannot retry empty evidence or exceeded budget
    if state.retries >= MAX_RETRIES:
        return END
    if state.verification in ("insufficient", "conflict"):
        return "research"
    return END


def build_workflow(
    planner_fn: Optional[Callable[[AgentState], AgentState]] = None,
    research_fn: Optional[Callable[[AgentState], AgentState]] = None,
    synthesis_fn: Optional[Callable[[AgentState], AgentState]] = None,
    verification_fn: Optional[Callable[[AgentState], AgentState]] = None,
) -> StateGraph:
    """Create a LangGraph StateGraph with optional dependency injection for nodes.
    
    Normal flow: planner -> research -> synthesis -> verification
    On failure with retries remaining: verification loops back to research.
    """
    p_node = planner_fn or planner_node
    r_node = research_fn or research_node
    s_node = synthesis_fn or synthesis_node
    v_node = verification_fn or verification_node

    graph = StateGraph(AgentState)
    graph.add_node("planner", p_node)
    graph.add_node("research", r_node)
    graph.add_node("synthesis", s_node)
    graph.add_node("verification", v_node)

    graph.add_edge("planner", "research")
    graph.add_edge("research", "synthesis")
    graph.add_edge("synthesis", "verification")
    graph.add_conditional_edges("verification", _route_after_verification)

    graph.set_entry_point("planner")
    return graph


# Default node references to detect monkeypatching in tests
_DEFAULT_PLANNER = planner_node
_DEFAULT_RESEARCH = research_node
_DEFAULT_SYNTHESIS = synthesis_node
_DEFAULT_VERIFICATION = verification_node

# Pre-compiled graph for production reuse
_DEFAULT_GRAPH = build_workflow().compile()


def _get_active_graph(
    planner_fn: Optional[Callable] = None,
    research_fn: Optional[Callable] = None,
    synthesis_fn: Optional[Callable] = None,
    verification_fn: Optional[Callable] = None,
):
    """Return pre-compiled graph if default nodes are unmodified, or compile with active nodes."""
    from . import planner, research, synthesis, verification

    cur_p = planner_fn or planner.planner_node
    cur_r = research_fn or research.research_node
    cur_s = synthesis_fn or synthesis.synthesis_node
    cur_v = verification_fn or verification.verification_node

    if (
        cur_p is _DEFAULT_PLANNER
        and cur_r is _DEFAULT_RESEARCH
        and cur_s is _DEFAULT_SYNTHESIS
        and cur_v is _DEFAULT_VERIFICATION
    ):
        return _DEFAULT_GRAPH

    # Dynamic graph if any node was patched or injected
    return build_workflow(cur_p, cur_r, cur_s, cur_v).compile()


@traced("run_workflow")
def run_workflow(
    question: str,
    persist_dir: Optional[Path | str] = None,
    planner_fn: Optional[Callable] = None,
    research_fn: Optional[Callable] = None,
    synthesis_fn: Optional[Callable] = None,
    verification_fn: Optional[Callable] = None,
) -> AgentState:
    """Run the multi-agent research workflow for a single question.

    Args:
        question: User query string.
        persist_dir: Optional path to the ChromaDB persistence directory.
        planner_fn: Optional injectable planner node callable.
        research_fn: Optional injectable research node callable.
        synthesis_fn: Optional injectable synthesis node callable.
        verification_fn: Optional injectable verification node callable.

    Returns:
        Final ``AgentState`` after verification.
    """
    p_dir_str = str(persist_dir) if persist_dir is not None else None
    init_state = AgentState(question=question, persist_dir=p_dir_str)

    graph = _get_active_graph(planner_fn, research_fn, synthesis_fn, verification_fn)
    raw = graph.invoke(init_state.model_dump())
    if isinstance(raw, dict):
        return AgentState(**raw)
    return raw
