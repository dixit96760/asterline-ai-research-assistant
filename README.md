---
title: Asterline AI Research Assistant
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
---

# Asterline AI Research Assistant

## 1. Project Overview
The Asterline AI Research Assistant is an evidence-grounded research assistant built in Python. It answers user questions by retrieving relevant company documents, synthesizing grounded answers with traceable citations, and validating factual consistency. The system uses **LangGraph** to orchestrate specialized agents, **ChromaDB** with **Sentence-Transformers** for local vector search, **SQLite** for user-isolated memory, **Gradio** for a responsive web UI, a configurable **OpenAI** / offline LLM synthesis backend, and optional **LangSmith** tracing.

## 2. Main Features
- **LangGraph Multi-Agent Workflow**:
  - **Planner Agent**: Analyzes research questions and decomposes multi-part queries into targeted subqueries.
  - **Research Agent**: Queries ChromaDB, retrieving relevant passages with metadata and similarity distances.
  - **Synthesis Agent**: Calls the configured LLM backend to produce evidence-grounded answers with citations, with deterministic offline fallback when API keys are absent.
  - **Verification Agent**: Validates cited source IDs, prevents infinite retry loops on insufficient evidence, and accurately distinguishes between ordinary multi-chunk documents and true contradictions.
- **ChromaDB Vector Store**: Local persistent vector index with `Sentence-Transformers` (`all-MiniLM-L6-v2`) embeddings, supporting idempotent builds.
- **Per-User Memory Management**: Thread-safe SQLite store with strict user isolation and full CRUD via the UI.
- **Gradio Web Interface**: Complete two-tab UI for conversational research and memory management.
- **LangSmith Tracing**: Optional end-to-end observability of workflow runs and agent execution.

## 3. Architecture Summary
```mermaid
flowchart TD
    subgraph UI[Gradio Web UI]
        UI_Research[Research Assistant Tab]
        UI_Memory[Memory Management Tab]
    end
    subgraph Workflow[LangGraph Workflow]
        Planner[Planner Agent]
        Research[Research Agent]
        Synthesis[Synthesis Agent]
        Verification[Verification Agent]
    end
    subgraph Retrieval[Retrieval Layer]
        Chroma[ChromaDB Local Store]
        Embeddings[all-MiniLM-L6-v2 Embeddings]
    end
    subgraph Storage[Data Layer]
        SQLite[data/memory.db (SQLite)]
        Corpus[data/corpus.jsonl]
    end
    subgraph LLM[LLM Provider]
        OpenAI[OpenAI API / Offline Fallback]
    end
    subgraph Observability[LangSmith]
        Tracer[@traced Decorator]
    end

    UI_Research --> Planner
    Planner --> Research
    Research --> Retrieval
    Retrieval --> Chroma & Embeddings
    Research --> Synthesis
    Synthesis --> LLM
    LLM --> Synthesis
    Synthesis --> Verification
    Verification -.->|retry on conflict (budget: 2)| Research
    Verification --> UI_Research
    UI_Memory --> SQLite
    Corpus --> Chroma
    Workflow -.-> Tracer
```

## 4. Prerequisites
- Windows 10/11 (or macOS / Linux)
- Python 3.11, 3.12, or 3.13 installed and added to `PATH`
- PowerShell (on Windows)

## 5. Windows Virtual-Environment Setup
Open Windows PowerShell in the project directory:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 6. Dependency Installation
```powershell
pip install -r requirements.txt
```

## 7. Configuration (`.env`)
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Key configuration settings in `.env`:
```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-4o-mini
MEMORY_DB_PATH=data/memory.db

# Optional LangSmith tracing
LANGSMITH_TRACING_ENABLED=false
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=asterline-assistant
```
*Note: If no `OPENAI_API_KEY` is provided or if `LLM_PROVIDER=offline`, the application uses a deterministic, offline evidence-synthesis engine. No paid API key is required to test or run the application locally.*

## 8. Validating the Corpus
Ensure all 75 records in `data/corpus.jsonl` are well-formed:
```powershell
python -m app.utils.corpus_validator
```

## 9. Building the Vector Index
Build or update the ChromaDB vector store from `data/corpus.jsonl`:
```powershell
python -m app.retrieval.build_index
```
This indexes the corpus into the local `./chroma` directory. Re-running the command safely updates existing records using idempotent upserts.

## 10. Launching the Web Application
Start the Gradio interface:
```powershell
python app.py
```
Open the local URL displayed in the terminal (default `http://127.0.0.1:7860`) in any browser.

## 11. Deploying to Hugging Face Spaces (Free Cloud Hosting)
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Choose a Space name (e.g. `asterline-assistant`).
3. Select **Gradio** as the Space SDK and **Public** or **Private**.
4. Clone the space repository locally or add it as a git remote:
   ```bash
   git init
   git remote add space https://huggingface.co/spaces/<your-username>/asterline-assistant
   git add .
   git commit -m "Deploy Asterline Assistant"
   git push space main
   ```
5. If using OpenAI or LangSmith, add your secret keys in your Space's **Settings -> Variables and secrets**.

## 12. Running the Automated Test Suite
Run all unit, integration, and evaluation tests:
```powershell
pytest -v
```

## 13. Known Limitations & Technical Details
- **Authentication**: User context is demonstrated using a selectable user dropdown (`user_1`, `user_2`, `user_3`). Production systems should replace this with cryptographic JWT/OAuth sessions.
- **SQLite Concurrency**: SQLite with thread-level `RLock` ensures safety for local use and desktop workloads; multi-tenant web servers should connect to PostgreSQL.
- **Prompt Injection Defense**: Passages passed to the LLM are delimited strictly as untrusted reference material, preventing documents from overriding system instructions.
- **Offline Reliability**: The entire workflow and test suite can execute completely offline without external network dependencies.
