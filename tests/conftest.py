# tests/conftest.py
"""Pytest configuration and environment sanitization."""

import os
import tempfile

# Guard against inaccessible system TORCHINDUCTOR_CACHE_DIR
torch_cache = os.getenv("TORCHINDUCTOR_CACHE_DIR")
if torch_cache:
    try:
        os.makedirs(torch_cache, exist_ok=True)
    except (PermissionError, OSError):
        fallback = os.path.join(tempfile.gettempdir(), "torch_inductor_cache")
        os.makedirs(fallback, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = fallback
