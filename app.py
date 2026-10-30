"""Gradio front end. Runs locally (`python app.py`), on Render (binds $PORT), or on Hugging Face Spaces."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import gradio as gr

from rag import DocQA
from rag.embeddings import make_embedder
from rag.llm import make_llm

EMBEDDER = os.getenv("EMBEDDER", "minilm")
SAMPLES = sorted((Path(__file__).parent / "eval" / "corpus").iterdir())
EXAMPLES = [
    "When does the locker code expire?",
    "What should we do if a recent deploy caused the outage?",
    "How many partitions does the click events topic have?",
    "What is the parental leave policy?",  # not in the docs: the agent should abstain
]


@lru_cache(maxsize=1)
def embedder():
    """Load the embedding model once per process and share it across sessions (saves memory)."""
    return make_embedder(EMBEDDER)


def _fmt_sources(cites: list[dict]) -> str:
    if not cites:
        return "_No sources cited._"
    lines = []
    for c in cites:
        snippet = c["text"][:220].replace("\n", " ")
        lines.append(f"**[{c['n']}] {c['source']} — page {c['page']}** (similarity {c['score']})\n> {snippet}…")
    return "\n\n".join(lines)


def _index(paths, store_kind):
    qa = DocQA(embedder=embedder(), store=store_kind, log_path="logs/queries.jsonl")
    stats = qa.ingest(paths)
    if stats["chunks"] < 1:
        raise ValueError(
            "No readable text was extracted. The PDF may be image-only/scanned; OCR is not enabled."
        )
    msg = (
        f"Indexed **{stats['files']} file(s)**, {stats['pages']} pages, "
        f"{stats.get('characters', 0):,} characters → {stats['chunks']} chunks "
        f"in **{store_kind.upper()}** using `{qa.embedder.model_name}`."
    )
    return qa, msg


def index_files(files, store_kind, state):
    if not files:
        return state, "Upload at least one PDF, DOCX, TXT or MD file."
    try:
        return _index([f if isinstance(f, str) else f.name for f in files], store_kind)
    except Exception as exc:
        return state, f"Could not index those files: {exc}"


def index_samples(store_kind):
    qa, msg = _index(SAMPLES, store_kind)
    names = ", ".join(p.name for p in SAMPLES)
    return qa, f"{msg}\n\nSample docs: {names}. Try one of the example questions below."


def ask(question, provider, api_key, history, qa):
    history = history or []
    if not question.strip():
        return history, gr.update(), gr.update()
    if qa is None:
        return history + [{"role": "user", "content": question},
                          {"role": "assistant", "content": "Index some documents first (or click **Load sample documents**)."}], "", ""
    try:
        qa.set_llm(make_llm(provider, (api_key or "").strip() or None))
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
        "A LangGraph agent retrieves, grades relevance, rewrites the query when retrieval is weak, and abstains "
        "instead of guessing. [Source on GitHub](https://github.com/mrchandrabhan/agentic-rag-docqa)"
    )
    qa_state = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(label="Documents", file_count="multiple", file_types=[".pdf", ".docx", ".txt", ".md"])
            store_kind = gr.Radio(["faiss", "chroma"], value="faiss", label="Vector store")
            with gr.Row():
                index_btn = gr.Button("Index documents", variant="primary")
                sample_btn = gr.Button("Load sample documents")
            status = gr.Markdown()
            provider = gr.Dropdown(["auto", "anthropic", "huggingface", "extractive"], value="auto",
                                   label="Answer model",
                                   info="auto = Claude with an Anthropic key, Mistral-7B with an HF token, otherwise extractive (no key)")
            api_key = gr.Textbox(label="API key (optional)", type="password",
                                 placeholder="sk-ant-… or hf_… — used for this request only, never stored")
        with gr.Column(scale=2):
            chat = gr.Chatbot(height=380, label="Conversation")
            question = gr.Textbox(placeholder="Ask about your documents…", label="Question")
            gr.Examples(EXAMPLES, inputs=question, label="Example questions (sample documents)")
            with gr.Accordion("Sources", open=True):
                sources = gr.Markdown()
            with gr.Accordion("Agent trace", open=False):
                trace = gr.Markdown()

    index_btn.click(index_files, [files, store_kind, qa_state], [qa_state, status])
    sample_btn.click(index_samples, [store_kind], [qa_state, status])
    question.submit(ask, [question, provider, api_key, chat, qa_state], [chat, sources, trace]).then(
        lambda: "", None, question)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.queue(default_concurrency_limit=2).launch(server_name="0.0.0.0" if os.getenv("PORT") else None,
                                                   server_port=port)
