"""Gradio front end for Hugging Face Spaces (or `python app.py` locally)."""
from __future__ import annotations

import os

import gradio as gr

from rag import DocQA
from rag.llm import make_llm

EMBEDDER = os.getenv("EMBEDDER", "minilm")


def _fmt_sources(cites: list[dict]) -> str:
    if not cites:
        return "_No sources cited._"
    lines = []
    for c in cites:
        snippet = c["text"][:220].replace("\n", " ")
        lines.append(f"**[{c['n']}] {c['source']} — page {c['page']}** (similarity {c['score']})\n> {snippet}…")
    return "\n\n".join(lines)


def index_files(files, store_kind, state):
    if not files:
        return state, "Upload at least one PDF, DOCX, TXT or MD file."
    qa = DocQA(embedder=EMBEDDER, store=store_kind, log_path="logs/queries.jsonl")
    stats = qa.ingest([f if isinstance(f, str) else f.name for f in files])
    msg = (f"Indexed **{stats['files']} file(s)**, {stats['pages']} pages → {stats['chunks']} chunks "
           f"in **{store_kind.upper()}** using `{qa.embedder.model_name}`.")
    return qa, msg


def ask(question, provider, api_key, history, qa):
    history = history or []
    if qa is None:
        return history + [{"role": "user", "content": question},
                          {"role": "assistant", "content": "Index some documents first."}], "", ""
    try:
        qa.set_llm(make_llm(provider, api_key.strip() or None))
        res = qa.ask(question)
    except Exception as exc:  # surface provider/auth errors in the chat instead of crashing
        return history + [{"role": "user", "content": question},
                          {"role": "assistant", "content": f"Error: {exc}"}], "", ""
    history = history + [{"role": "user", "content": question}, {"role": "assistant", "content": res["answer"]}]
    trace = "\n".join(f"{i}. {t}" for i, t in enumerate(res["trace"], 1)) + f"\n\nLatency: {res['latency_ms']} ms"
    return history, _fmt_sources(res["citations"]), trace


with gr.Blocks(title="Agentic RAG Document Q&A") as demo:
    gr.Markdown(
        "# Agentic RAG Document Q&A\n"
        "Upload documents, then ask questions. Answers are grounded in your files with **page-level citations**. "
        "A LangGraph agent retrieves, grades relevance, rewrites the query when retrieval is weak, then answers."
    )
    qa_state = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(label="Documents", file_count="multiple", file_types=[".pdf", ".docx", ".txt", ".md"])
            store_kind = gr.Radio(["faiss", "chroma"], value="faiss", label="Vector store")
            index_btn = gr.Button("Index documents", variant="primary")
            status = gr.Markdown()
            provider = gr.Dropdown(["auto", "anthropic", "huggingface", "extractive"], value="auto",
                                   label="Answer model",
                                   info="auto = Claude if an Anthropic key is set, else HF, else extractive (no key)")
            api_key = gr.Textbox(label="API key (optional)", type="password",
                                 placeholder="sk-ant-… or hf_… — never stored")
        with gr.Column(scale=2):
            chat = gr.Chatbot(height=380, label="Conversation")
            question = gr.Textbox(placeholder="Ask about your documents…", label="Question")
            with gr.Accordion("Sources", open=True):
                sources = gr.Markdown()
            with gr.Accordion("Agent trace", open=False):
                trace = gr.Markdown()

    index_btn.click(index_files, [files, store_kind, qa_state], [qa_state, status])
    question.submit(ask, [question, provider, api_key, chat, qa_state], [chat, sources, trace]).then(
        lambda: "", None, question)

if __name__ == "__main__":
    demo.launch()
