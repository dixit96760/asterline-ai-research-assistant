# Asterline AI Research Assistant – Design Document

## 1. Project Objective and Scope
The Asterline AI Research Assistant is an evidence-grounded research assistant built with **LangGraph**, **ChromaDB**, **Sentence-Transformers**, **SQLite**, **Gradio**, and a configurable **OpenAI** / offline LLM synthesis layer. Its purpose is to answer user questions by retrieving relevant company documents, synthesizing answers, and providing citations, while maintaining isolated per-user memory and optional observability via **LangSmith** tracing.

## 2. Architecture Overview
```mermaid
flowchart TD
    subgraph UI[Gradio UI (app/ui.py)]
        UI_Research[Research Tab]
        UI_Memory[Memory Tab]
    end
    subgraph Workflow[LangGraph Workflow (app/agents/workflow.py)]
        Planner[Planner Node\n(Query Decomposition)]
        Research[Research Node\n(ChromaDB Multi-Query Retrieval)]
        Synthesis[Synthesis Node\n(LLM / Offline Grounded Synthesis)]
        Verification[Verification Node\n(Evidence & Conflict Validation)]
    end
    subgraph Retrieval[Retrieval Layer (app/retrieval/)]
        Chroma[ChromaDB Persistent Client]
        Embeddings[Sentence-Transformers all-MiniLM-L6-v2]
        CorpusIngest[load_corpus.py & build_index.py\n(Idempotent Upsert)]
    end
    subgraph Memory[SQLite Memory Store (app/memory/)]
        SQLite[data/memory.db\n(RLock, UTC, User-Isolated)]
    end
    subgraph LLM[LLM Provider (app/agents/llm.py)]
        OpenAI[OpenAI / Compatible Endpoint\nor Deterministic Offline Fallback]
    end
    subgraph Tracing[Observability (app/tracing.py)]
        Tracer[LangSmith Tracer / @traced]
    end

    UI_Research -->|question| Planner
    Planner -->|need_multiple & subqueries| Research
    Research -->|retrieval query| Retrieval
    Retrieval --> Chroma & Embeddings
    Research -->|retrieved chunks| Synthesis
    Synthesis -->|synthesize_answer| LLM
    LLM -->|grounded answer & citations| Synthesis
    Synthesis -->|AgentState| Verification
    Verification -.->|retry if conflict/insufficient and budget remains| Research
    Verification -->|final verified state| UI_Research
    UI_Memory -->|user-scoped CRUD| SQLite
    Workflow -.->|spans| Tracer
```

## 3. Agent Responsibilities & LangGraph Flow
| Agent | Responsibility |
|-------|----------------|
| **Planner** (`app/agents/planner.py`) | Analyzes the user question. Decomposes compound questions containing conjunctions (`and`, `or`, `vs`) into targeted subqueries. |
| **Research** (`app/agents/research.py`) | Performs similarity searches against **ChromaDB** using the configured embeddings for each subquery. Merges and deduplicates chunks. |
| **Synthesis** (`app/agents/synthesis.py`) | Synthesizes a factual, evidence-grounded answer using `app/agents/llm.py`. Extracts citations (`[{chunk_id}]`) and protects against prompt injection. |
| **Verification** (`app/agents/verification.py`) | Validates citations, detects insufficient evidence, and checks for genuine contradictions without falsely flagging normal multi-chunk documents. Limits retries to `MAX_RETRIES = 2`. |

The workflow operates on the shared `AgentState` model defined in `app/agents/state.py` and supports runtime dependency injection for unit testing.

## 4. Document Ingestion, Embeddings & Retrieval
* Documents are stored in `data/corpus.jsonl` (75 records).
* Each record contains `chunk_id`, `doc_id`, `title`, `category`, `owner`, `source_url`, `published`, `version`, and `text`.
* Ingestion (`app/retrieval/load_corpus.py` and `app/retrieval/build_index.py`) validates the corpus schema, generates local embeddings via `all-MiniLM-L6-v2`, and executes idempotent `collection.upsert(...)` into ChromaDB.
* The vector store defaults to `chroma/` and is fully configurable via `persist_dir`.

## 5. Grounded Answers & Citations
* Synthesis treats reference passages as untrusted source material and instructs the model to answer exclusively from evidence.
* Citations are parsed from the response or matched against retrieved IDs in `[{chunk_id}]` format.
* The Gradio UI displays citations alongside the answer and shows the verification status (`verified`, `conflict`, or `insufficient`).

## 6. Conflict & Insufficient Evidence Handling
* **Complementary Chunks**: Chunks belonging to the same document (matching `doc_id` or chunk prefix) sharing a title are recognized as normal multi-chunk sections, preventing false-positive conflicts.
* **Genuine Conflict**: Different versions of the same document or opposing statements across distinct sources trigger `verification = "conflict"`.
* **Insufficient Evidence**: Queries with zero retrieved chunks or missing citations trigger `verification = "insufficient"`. Empty retrieval immediately halts the retry loop, avoiding infinite recursion.
* **Bounded Retries**: The retry budget is capped at `MAX_RETRIES = 2`.

## 7. SQLite Memory Schema & User Isolation
The memory store (`app/memory/memory.py`) manages personal user notes in `data/memory.db`:
```sql
CREATE TABLE IF NOT EXISTS memories (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   TEXT    NOT NULL,
    content   TEXT    NOT NULL,
    timestamp TEXT    NOT NULL
);
```
* **Thread Safety**: Uses `threading.RLock` to prevent deadlocks on concurrent access and nested calls.
* **User Isolation**: All SQL queries strictly filter by `user_id`. One user cannot view, update, or delete another user's notes.
* **Data Validation**: Enforces non-empty user IDs, maximum content length (10,000 characters), timezone-aware UTC ISO timestamps, and heuristic secret rejection (`password`, `token`, `secret`, `api_key`).

## 8. Separation of Personal Memory & Authoritative Evidence
* **Personal Memory**: Stored in SQLite, managed via the Memory Management tab in the UI for user-specific notes.
* **Authoritative Evidence**: Stored in ChromaDB, derived exclusively from `data/corpus.jsonl`. Retrieval never queries SQLite personal memory.

## 9. LangSmith Tracing & Observability
* Tracing is configured in `app/tracing.py` and wrapped by the `@traced` decorator.
* Activated when `LANGSMITH_TRACING_ENABLED=true` (or `LANGSMITH_TRACING=true` / `LANGCHAIN_TRACING_V2=true`) and an API key is provided.
* Degrades to a transparent no-op wrapper when tracing credentials are absent.

## 10. Evaluation & Test Suite
The automated test suite runs via `pytest` and contains 45 tests:
* `tests/test_corpus_validation.py` (8 tests): Real corpus schema, record counts, duplicate checks, missing fields, malformed JSONL.
* `tests/test_evaluation_suite.py` (10 tests): Factual QA, multi-chunk retrieval, conflict detection, unsupported questions, citation validity, memory persistence, user isolation.
* `tests/test_memory.py` (11 tests): CRUD operations, cross-user isolation, empty inputs, secret rejection, thread safety.
* `tests/test_retrieval.py` (1 test): Subprocess CLI index building and querying on isolated temporary directories.
* `tests/test_tracing.py` (1 test): Traced decorator behavior.
* `tests/test_ui.py` (6 tests): UI block generation, input validation, memory CRUD UI flows, exception handling.
* `tests/test_workflow.py` (8 tests): Direct questions, multi-part query decomposition, complementary chunks, true conflicts, empty retrieval, retry bounding, invalid citation rejection, mock LLM client integration.

## 11. Known Limitations and Trade-offs
* **Authentication**: Users are selected via a prototype dropdown (`user_1`, `user_2`, etc.); production deployments require cryptographic session authentication.
* **Single-Node SQLite**: SQLite is optimal for local prototyping and personal notes, but multi-node production deployments should use PostgreSQL.
* **Local Embeddings**: `all-MiniLM-L6-v2` runs locally without API costs; very large corpora (millions of chunks) may benefit from distributed vector storage.
