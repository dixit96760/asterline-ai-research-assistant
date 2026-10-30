# app/tracing.py
"""LangSmith tracing configuration.

Tracing is optional and automatically activated when:
  LANGSMITH_TRACING=true (or LANGSMITH_TRACING_ENABLED=true / LANGCHAIN_TRACING_V2=true)
  LANGSMITH_API_KEY=<your-key> (or LANGCHAIN_API_KEY=<your-key>)
  LANGSMITH_PROJECT=asterline-assistant (optional)

When disabled or when credentials are missing, this module provides a safe no-op
``traced`` decorator and disables telemetry so application execution is unaffected.
"""

import os
import sys
import logging

logger = logging.getLogger("asterline.tracing")

_TRACING_ENABLED = (
    os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1")
    or os.getenv("LANGSMITH_TRACING_ENABLED", "").lower() in ("true", "1")
    or os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1")
)
_API_KEY = (os.getenv("LANGSMITH_API_KEY", "").strip() or os.getenv("LANGCHAIN_API_KEY", "").strip())
_PROJECT = (os.getenv("LANGSMITH_PROJECT", "").strip() or os.getenv("LANGCHAIN_PROJECT", "asterline-assistant").strip())

# Propagate environment variables expected by LangChain / LangSmith
if _TRACING_ENABLED and _API_KEY:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", _API_KEY)
    os.environ.setdefault("LANGSMITH_API_KEY", _API_KEY)
    os.environ.setdefault("LANGCHAIN_PROJECT", _PROJECT)
    os.environ.setdefault("LANGSMITH_PROJECT", _PROJECT)

tracer = None


def _try_init_tracer():
    """Attempt to instantiate a LangChainTracer, returning None on failure."""
    global tracer
    try:
        from langchain_core.tracers.langchain import LangChainTracer  # noqa: PLC0415
        tracer = LangChainTracer(project_name=_PROJECT)
    except Exception as exc:
        logger.debug("LangChainTracer initialization skipped or unavailable: %s", exc)
        tracer = None


if _TRACING_ENABLED:
    if not _API_KEY:
        logger.info("Tracing requested but LANGSMITH_API_KEY is empty. Tracing is disabled.")
    else:
        _try_init_tracer()


def traced(name: str):
    """Decorator that wraps a function with LangSmith tracing when enabled.

    Falls back to a transparent no-op wrapper when tracing is disabled or unavailable.
    """
    if _TRACING_ENABLED and _API_KEY:
        try:
            from langsmith import traceable  # noqa: PLC0415
            return traceable(name=name)
        except Exception:
            pass

    def _noop(fn):
        return fn
    return _noop
