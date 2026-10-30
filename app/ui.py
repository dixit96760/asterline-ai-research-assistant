# app/ui.py
"""Gradio interface for the Asterline AI Research Assistant.

Two tabs are provided:
1. Research Assistant: chat with the multi-agent workflow and inspect citations.
2. Memory Management: isolated per-user CRUD operations on personal notes.
"""

from __future__ import annotations

import logging
from typing import Tuple, List, Optional, Any, Dict
import gradio as gr

from app.agents.workflow import run_workflow
from app.memory.memory import MemoryStore

logger = logging.getLogger("asterline.ui")

# ---------------------------------------------------------------------------
# Helper functions for the Research Assistant tab
# ---------------------------------------------------------------------------

def answer_question(question: str) -> Tuple[str, str, List[str]]:
    """Run the LangGraph workflow and format the result.

    Returns a tuple of (answer, verification_status, citations).
    """
    clean_q = (question or "").strip()
    if not clean_q:
        return "Please enter a question to research.", "insufficient", []

    try:
        state = run_workflow(clean_q)
        answer = state.answer or "I could not find an answer in the reference documents."
        verification = state.verification or "unknown"
        # Build citation strings "[chunk_id] title"
        citations: List[str] = []
        if state.retrieved:
            for chunk in state.retrieved:
                title = chunk.metadata.get("title", "")
                citations.append(f"[{chunk.chunk_id}] {title}")
        return answer, verification, citations
    except Exception as exc:
        logger.exception("Error in answer_question: %s", exc)
        return "An error occurred while processing your request.", "error", []


# ---------------------------------------------------------------------------
# Helper functions for the Memory Management tab
# ---------------------------------------------------------------------------

def list_memories(user_id: str) -> List[Tuple[int, str, str]]:
    """Return a list of memories for *user_id*.
    Each entry is (id, content, timestamp).
    """
    if not user_id or not user_id.strip():
        return []
    try:
        return MemoryStore.list_memories(user_id.strip())
    except Exception as exc:
        logger.error("Error listing memories for %s: %s", user_id, exc)
        return []


def add_memory(user_id: str, content: str) -> str:
    clean_content = (content or "").strip()
    if not clean_content:
        return "Validation error: Memory content cannot be empty."
    try:
        MemoryStore.add_memory(user_id, clean_content)
        return "Memory added successfully."
    except ValueError as ve:
        return f"Validation error: {ve}"
    except Exception as exc:
        logger.error("Error adding memory: %s", exc)
        return "Failed to add memory due to an internal error."


def update_memory(user_id: str, memory_id: Optional[int], new_content: str) -> str:
    if memory_id is None:
        return "Validation error: Please provide a valid integer Memory ID to update."
    clean_content = (new_content or "").strip()
    if not clean_content:
        return "Validation error: Updated memory content cannot be empty."
    try:
        success = MemoryStore.update_memory(user_id, int(memory_id), clean_content)
        return "Memory updated successfully." if success else f"Memory #{memory_id} not found for user {user_id}."
    except ValueError as ve:
        return f"Validation error: {ve}"
    except Exception as exc:
        logger.error("Error updating memory: %s", exc)
        return "Failed to update memory due to an internal error."


def delete_memory(user_id: str, memory_id: Optional[int]) -> str:
    if memory_id is None:
        return "Validation error: Please provide a valid integer Memory ID to delete."
    try:
        success = MemoryStore.delete_memory(user_id, int(memory_id))
        return "Memory deleted successfully." if success else f"Memory #{memory_id} not found for user {user_id}."
    except ValueError as ve:
        return f"Validation error: {ve}"
    except Exception as exc:
        logger.error("Error deleting memory: %s", exc)
        return "Failed to delete memory due to an internal error."


def delete_all_memories(user_id: str) -> str:
    try:
        count = MemoryStore.delete_all(user_id)
        return f"Deleted {count} memories for user {user_id}."
    except Exception as exc:
        logger.error("Error deleting all memories: %s", exc)
        return "Failed to delete memories due to an internal error."


# ---------------------------------------------------------------------------
# Gradio UI definition
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""
    with gr.Blocks(title="Asterline AI Research Assistant") as demo:
        gr.Markdown("# Asterline AI Research Assistant")
        gr.Markdown("Evidence-grounded research assistant orchestrating LangGraph, ChromaDB, and SQLite.")

        with gr.Tabs():
            # ------------------- Research Assistant Tab -------------------
            with gr.TabItem("Research Assistant"):
                user_selector = gr.Dropdown(
                    choices=["user_1", "user_2", "user_3"],
                    label="Current User Context",
                    value="user_1",
                )
                chat_history = gr.Chatbot(label="Conversation", height=420)
                txt_input = gr.Textbox(
                    label="Your question",
                    placeholder="Ask about Asterline plans, features, SLAs, or technical specs...",
                )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear Conversation")

                def submit_message(user, message, history):
                    clean_msg = (message or "").strip()
                    if not clean_msg:
                        return history or [], ""
                    # Gradio 6 expects list of dicts: {'role': 'user'|'assistant', 'content': str}
                    curr_history = list(history or [])
                    answer, verification, citations = answer_question(clean_msg)
                    citations_str = "; ".join(citations) if citations else "None"
                    bot_response = (
                        f"{answer}\n\n"
                        f"**Citations:** {citations_str}\n\n"
                        f"**Verification:** `{verification}`"
                    )
                    curr_history.append({"role": "user", "content": clean_msg})
                    curr_history.append({"role": "assistant", "content": bot_response})
                    return curr_history, ""

                submit_btn.click(
                    submit_message,
                    inputs=[user_selector, txt_input, chat_history],
                    outputs=[chat_history, txt_input],
                )
                txt_input.submit(
                    submit_message,
                    inputs=[user_selector, txt_input, chat_history],
                    outputs=[chat_history, txt_input],
                )
                clear_btn.click(lambda: [], None, chat_history)

            # ------------------- Memory Management Tab -------------------
            with gr.TabItem("Memory Management"):
                gr.Markdown("### Personal User Notes & Memory Isolation")
                mem_user = gr.Dropdown(
                    choices=["user_1", "user_2", "user_3"],
                    label="Select User",
                    value="user_1",
                )
                mem_list = gr.Dataframe(
                    headers=["ID", "Content", "Timestamp"],
                    datatype=["number", "str", "str"],
                    label="Your Stored Memories",
                )
                refresh_btn = gr.Button("Refresh List")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Add Memory")
                        add_input = gr.Textbox(label="New Memory Content", placeholder="Enter notes or preferences...")
                        add_btn = gr.Button("Add Memory", variant="primary")

                    with gr.Column():
                        gr.Markdown("#### Update Memory")
                        upd_id = gr.Number(label="Memory ID to Update", precision=0)
                        upd_input = gr.Textbox(label="New Content", placeholder="Enter replacement text...")
                        upd_btn = gr.Button("Update Memory")

                    with gr.Column():
                        gr.Markdown("#### Delete Memory")
                        del_id = gr.Number(label="Memory ID to Delete", precision=0)
                        del_btn = gr.Button("Delete Single Memory", variant="stop")
                        del_all_btn = gr.Button("Delete All My Memories", variant="stop")

                status = gr.Label(label="Operation Status")

                def refresh(user):
                    return list_memories(user)

                def add_mem(user, content):
                    msg = add_memory(user, content)
                    return msg, refresh(user)

                def upd_mem(user, mem_id, new_content):
                    msg = update_memory(user, mem_id, new_content)
                    return msg, refresh(user)

                def del_mem(user, mem_id):
                    msg = delete_memory(user, mem_id)
                    return msg, refresh(user)

                def del_all(user):
                    msg = delete_all_memories(user)
                    return msg, refresh(user)

                mem_user.change(refresh, inputs=mem_user, outputs=mem_list)
                refresh_btn.click(refresh, inputs=mem_user, outputs=mem_list)
                add_btn.click(add_mem, inputs=[mem_user, add_input], outputs=[status, mem_list])
                upd_btn.click(upd_mem, inputs=[mem_user, upd_id, upd_input], outputs=[status, mem_list])
                del_btn.click(del_mem, inputs=[mem_user, del_id], outputs=[status, mem_list])
                del_all_btn.click(del_all, inputs=mem_user, outputs=[status, mem_list])

    return demo


def launch_ui(share: Optional[bool] = None) -> None:
    """Create and launch the Gradio app."""
    import os
    if share is None:
        share = os.getenv("GRADIO_SHARE", "true").lower() in ("true", "1")
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share)
    print(f"ASTERLINE_LOCAL_URL: {demo.local_url}", flush=True)
    print(f"ASTERLINE_SHARE_URL: {demo.share_url}", flush=True)


if __name__ == "__main__":
    launch_ui()
