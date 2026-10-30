import shutil
from pathlib import Path
import subprocess
import sys

import pytest

# Path to the load_corpus module
MODULE_PATH = Path(__file__).parents[1] / "app" / "retrieval" / "load_corpus.py"

@pytest.fixture(scope="module")
def temp_chroma_dir(tmp_path_factory):
    # Create a temporary directory for ChromaDB persistence
    dir_path = tmp_path_factory.mktemp("chroma")
    yield dir_path
    # Cleanup after tests
    shutil.rmtree(dir_path, ignore_errors=True)

def test_build_and_query_index(temp_chroma_dir):
    # Build the index
    build_cmd = [sys.executable, str(MODULE_PATH), "--build", "--persist", str(temp_chroma_dir)]
    result = subprocess.run(build_cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Build failed: {result.stderr}"
    # Query the index
    query = "What is the pricing model?"
    query_cmd = [sys.executable, str(MODULE_PATH), "--query", query, "--persist", str(temp_chroma_dir)]
    result = subprocess.run(query_cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Query failed: {result.stderr}"
    # Ensure output contains expected fields
    output = result.stdout
    assert "Chunk ID:" in output
    assert "Title:" in output
    assert "Score (distance):" in output
