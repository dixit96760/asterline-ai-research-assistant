# app/memory/memory.py
"""Memory module for Asterline Assistant.

Database location is configured via the MEMORY_DB_PATH environment variable.
If unset, defaults to data/memory.db inside the project root.
All operations enforce strict user isolation.
"""

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

MAX_MEMORY_CONTENT_LENGTH = 10000


def _resolve_db_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    env_val = os.getenv("MEMORY_DB_PATH", "").strip()
    if env_val:
        p = Path(env_val)
        if not p.is_absolute():
            p = project_root / p
    else:
        p = project_root / "data" / "memory.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


_DB_PATH: Path = _resolve_db_path()
_LOCK = threading.RLock()


def _init_db(db_path: Optional[Path] = None) -> None:
    path = db_path or _DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        conn = sqlite3.connect(str(path))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id   TEXT    NOT NULL,
                    content   TEXT    NOT NULL,
                    timestamp TEXT    NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()


_init_db(_DB_PATH)


class MemoryStore:
    """CRUD helper for per-user memories stored in SQLite.
    
    Provides strict user isolation by requiring user_id on all operations.
    Note: Secret keyword filtering is a defense-in-depth heuristic, not a full secret scanner.
    """

    @staticmethod
    def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
        path = db_path or _DB_PATH
        _init_db(path)
        return sqlite3.connect(str(path), detect_types=sqlite3.PARSE_DECLTYPES)

    @staticmethod
    def _validate_user(user_id: str) -> None:
        if not user_id or not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("User ID must be a non-empty string.")

    @staticmethod
    def _validate_content(content: str) -> None:
        if not content or not isinstance(content, str) or not content.strip():
            raise ValueError("Memory content cannot be empty.")
        if len(content) > MAX_MEMORY_CONTENT_LENGTH:
            raise ValueError(f"Memory content exceeds maximum length of {MAX_MEMORY_CONTENT_LENGTH} characters.")
        lower = content.lower()
        if any(kw in lower for kw in ("api_key", "api key", "password", "secret", "token")):
            raise ValueError("Refusing to store a potential secret as a memory.")

    @staticmethod
    def add_memory(user_id: str, content: str, db_path: Optional[Path] = None) -> None:
        MemoryStore._validate_user(user_id)
        MemoryStore._validate_content(content)
        ts = datetime.now(timezone.utc).isoformat()
        with _LOCK:
            with MemoryStore._connect(db_path) as conn:
                conn.execute(
                    "INSERT INTO memories (user_id, content, timestamp) VALUES (?, ?, ?)",
                    (user_id.strip(), content.strip(), ts),
                )
                conn.commit()

    @staticmethod
    def list_memories(user_id: str, db_path: Optional[Path] = None) -> List[Tuple[int, str, str]]:
        MemoryStore._validate_user(user_id)
        with _LOCK:
            with MemoryStore._connect(db_path) as conn:
                cur = conn.execute(
                    "SELECT id, content, timestamp FROM memories WHERE user_id = ? ORDER BY id",
                    (user_id.strip(),),
                )
                return [(row[0], row[1], row[2]) for row in cur.fetchall()]

    @staticmethod
    def update_memory(user_id: str, memory_id: int, new_content: str, db_path: Optional[Path] = None) -> bool:
        MemoryStore._validate_user(user_id)
        MemoryStore._validate_content(new_content)
        if not isinstance(memory_id, int):
            try:
                memory_id = int(memory_id)
            except (ValueError, TypeError):
                raise ValueError("Memory ID must be an integer.")
        ts = datetime.now(timezone.utc).isoformat()
        with _LOCK:
            with MemoryStore._connect(db_path) as conn:
                cur = conn.execute(
                    "UPDATE memories SET content = ?, timestamp = ? WHERE id = ? AND user_id = ?",
                    (new_content.strip(), ts, memory_id, user_id.strip()),
                )
                conn.commit()
                return cur.rowcount > 0

    @staticmethod
    def delete_memory(user_id: str, memory_id: int, db_path: Optional[Path] = None) -> bool:
        MemoryStore._validate_user(user_id)
        if not isinstance(memory_id, int):
            try:
                memory_id = int(memory_id)
            except (ValueError, TypeError):
                raise ValueError("Memory ID must be an integer.")
        with _LOCK:
            with MemoryStore._connect(db_path) as conn:
                cur = conn.execute(
                    "DELETE FROM memories WHERE id = ? AND user_id = ?",
                    (memory_id, user_id.strip()),
                )
                conn.commit()
                return cur.rowcount > 0

    @staticmethod
    def delete_all(user_id: str, db_path: Optional[Path] = None) -> int:
        MemoryStore._validate_user(user_id)
        with _LOCK:
            with MemoryStore._connect(db_path) as conn:
                cur = conn.execute(
                    "DELETE FROM memories WHERE user_id = ?",
                    (user_id.strip(),),
                )
                conn.commit()
                return cur.rowcount
