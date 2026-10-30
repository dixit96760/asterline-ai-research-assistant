import os
import pytest
from app.tracing import traced

# Define the sample function without decorator initially
def sample_function(x: int) -> int:
    """A simple function to be traced when tracing is enabled."""
    return x * 2

# If LangSmith API key is configured, wrap the function with the tracer decorator
if os.getenv("LANGSMITH_API_KEY"):
    sample_function = traced("sample_function")(sample_function)

def test_sample_function_tracing():
    # Call the function and verify it returns expected result
    result = sample_function(3)
    assert result == 6
    # No explicit LangSmith verification; the decorator logs a run when enabled.
