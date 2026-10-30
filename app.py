"""Hugging Face Spaces Entry Point for Asterline AI Research Assistant."""

import os
from pathlib import Path

# Auto-ensure ChromaDB vector index is ready on startup
from app.retrieval import _ensure_index, DEFAULT_PERSIST_DIR
from app.ui import build_demo

try:
    _ensure_index(DEFAULT_PERSIST_DIR)
except Exception as e:
    print(f"[Startup] Index verification notice: {e}")

# Build Gradio application
demo = build_demo()

if __name__ == "__main__":
    demo.launch()
