import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

# Guard against broken or inaccessible Torch inductor cache directories
torch_cache = os.getenv("TORCHINDUCTOR_CACHE_DIR")
if torch_cache:
    try:
        os.makedirs(torch_cache, exist_ok=True)
    except (PermissionError, OSError):
        fallback_dir = os.path.join(tempfile.gettempdir(), "torch_inductor_cache")
        os.makedirs(fallback_dir, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = fallback_dir

from . import tracing  # noqa: F401
