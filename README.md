---
title: Agentic RAG Document QA
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.30.0"
app_file: app.py
pinned: false
license: mit
---

# Agentic RAG Document Q&A

[![CI](https://github.com/mrchandrabhan/agentic-rag-docqa/actions/workflows/ci.yml/badge.svg)](https://github.com/mrchandrabhan/agentic-rag-docqa/actions/workflows/ci.yml)
[![Deploy to HF Space](https://github.com/mrchandrabhan/agentic-rag-docqa/actions/workflows/deploy-space.yml/badge.svg)](https://github.com/mrchandrabhan/agentic-rag-docqa/actions/workflows/deploy-space.yml)

Upload PDFs, Word docs or text files and ask questions. Every answer is grounded in your documents and cites the **file and page** it came from. A **LangGraph** agent decides when retrieval is good enough to answer, rewrites the query when it isn't, and says *"I couldn't find that"* instead of guessing.

**Live demo:** https://huggingface.co/spaces/mrchandrabhan/agentic-rag-docqa

## How it works

```
upload ─► load (PDF/DOCX/TXT/MD, page-aware) ─► chunk (800 chars, 120 overlap)
       ─► embed (all-MiniLM-L6-v2) ─► index (FAISS or ChromaDB)

question ─► retrieve top-k ─► grade relevance ──relevant──► generate ─► validate citations ─► answer + sources
                  ▲                │
                  └── rewrite ◄────┘ weak (1 retry)  ──still weak──► abstain
```

| Stage | Implementation |
|---|---|
| Loading | `pypdf` (per page), `docx2txt`, plain text/Markdown; multi-file upload |
| Chunking | LangChain `RecursiveCharacterTextSplitter`, page metadata kept on every chunk |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-d, normalized → cosine) |
| Vector store | FAISS `IndexFlatIP` or ChromaDB (cosine HNSW), switchable in the UI |
| Agent | LangGraph state machine: retrieve → grade → rewrite/generate → cite |
| Generation | Anthropic Claude (LangChain `ChatAnthropic`), Hugging Face Inference (Mistral-7B-Instruct), or a no-key extractive fallback |
| Guardrails | Answers only from context, `[n]` citations required, out-of-range citations dropped, abstains when nothing relevant |
| Observability | Per-query agent trace in the UI and JSONL query logs (`logs/queries.jsonl`) with latency |
| UI | Gradio Blocks with chat, sources panel and agent-trace panel |

## Evaluation

`eval/` holds a small labeled set: 16 questions over 3 documents, including 2 that the documents can't answer.

```bash
python -m eval.run_eval                  # MiniLM + FAISS + extractive answers
python -m eval.run_eval --llm anthropic  # grade Claude's answers (needs ANTHROPIC_API_KEY)
python -m eval.run_eval --store chroma   # compare vector stores
```

It reports **hit@k**, **MRR**, **answer accuracy**, **citation precision** and **abstention accuracy**, and lists every failing case. CI runs the same eval with an offline embedder and fails the build if retrieval drops below the baseline.

## Run locally

```bash
git clone https://github.com/mrchandrabhan/agentic-rag-docqa
cd agentic-rag-docqa
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # optional; or HF_TOKEN=hf_...; or neither (extractive mode)
python app.py                          # http://localhost:7860
```

Docker: `docker build -t agentic-rag . && docker run -p 7860:7860 -e ANTHROPIC_API_KEY agentic-rag`

## Deploy

**One command from your machine** (needs `gh auth login` and `hf auth login` once):

```bash
./publish.sh
```

It creates the GitHub repo, pushes, creates the Gradio Space and uploads the app.

**Continuous deploy:** add a Hugging Face *Write* token as the GitHub Actions secret `HF_TOKEN`. Every push to `main` then redeploys the Space via `.github/workflows/deploy-space.yml`, and CI publishes a MiniLM eval table (FAISS vs ChromaDB) in the run summary.

> Don't add your Anthropic key as a Space secret unless you're happy for visitors to spend your credits. Without one, the Space answers in extractive mode, or visitors can paste their own key (never stored).

## Tests

```bash
pip install -r requirements-ci.txt
pytest -q
```

21 offline tests (no model downloads or API keys) cover loaders, chunking, both vector stores, the agent's rewrite/abstain paths, citation validation, JSONL logging, and the eval regression gate.

## Project layout

```
app.py              Gradio UI
rag/loaders.py      page-aware file loading
rag/chunking.py     chunking with source/page metadata
rag/embeddings.py   MiniLM embedder + offline hash embedder
rag/stores.py       FAISS and ChromaDB behind one interface
rag/llm.py          Claude / Hugging Face / extractive backends
rag/graph.py        LangGraph agent
rag/pipeline.py     DocQA: ingest + ask + JSONL logging
eval/               labeled dataset, sample corpus, metrics
tests/              pytest suite
```

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Use Claude for answers |
| `ANTHROPIC_MODEL` | `claude-haiku-5-5` | Claude model |
| `HF_TOKEN` | – | Use the Hugging Face Inference API |
| `HF_MODEL` | `mistralai/Mistral-7B-Instruct-v0.3` | HF model |
| `EMBEDDER` | `minilm` | `hash` for an offline, download-free embedder |

## Roadmap

- Hybrid search (BM25 + vectors) and a cross-encoder reranker
- LLM-as-judge faithfulness scoring in the eval
- Persisted indexes between sessions

MIT © Chandrabhan Singh Vaghela
