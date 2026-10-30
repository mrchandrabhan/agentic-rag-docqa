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

Upload PDFs, Word docs or text files and ask questions. Every answer is grounded in your documents and cites the file and page it came from. A LangGraph agent retrieves, grades relevance, rewrites the query when it isn't strong enough, and abstains instead of guessing.

**Live demo:** https://agentic-rag-docqa-dwvv.onrender.com

## Role in the portfolio

This repository is the reusable knowledge and retrieval engine for the separate **Nexus AI** enterprise workspace. It remains independently runnable and independently tested. Nexus calls its HTTP adapter; it does not reimplement this repository's document loaders, chunking, embeddings, vector search, LangGraph workflow, or citation validation.

## Existing RAG pipeline

```
upload -> page-aware load -> chunk -> MiniLM embeddings -> FAISS or Chroma
question -> retrieve -> grade -> rewrite/generate -> validate citations -> answer
```

See the existing rag package for the pipeline and eval directory for the labeled retrieval/answer regression set.

## HTTP API for Nexus AI

A small FastAPI adapter exposes the existing DocQA engine. It is a separate local process from the Gradio demo; it uses the same implementation under rag.

From the repository root:

**Windows PowerShell**

```powershell
$env:EMBEDDER = "hash"
uvicorn api:app --host 0.0.0.0 --port 7861
```

Use EMBEDDER=minilm for the MiniLM semantic embedder after the model is available locally. The hash embedder is quick and deterministic for offline development but is less semantically capable. Set ANTHROPIC_API_KEY or HF_TOKEN to use a hosted answer model; without credentials the existing extractive answer mode is used.

Endpoints:

- GET /health — process liveness, no model load.
- GET /ready — initialize/check the RAG engine and return model/index details.
- GET /api/v1/documents — list sample and uploaded documents in this process.
- POST /api/v1/documents — index a PDF, DOCX, TXT, or Markdown file (10 MiB maximum).
- POST /api/v1/ask — answer a question with citations, trace, and latency.

Open http://localhost:7861/docs for interactive API documentation.

The service currently uses a process-local vector index. Sample corpus files are indexed when the engine first initializes; additional uploads are added to that in-memory index. Restarting the API rebuilds the sample corpus and clears additional uploads. This is a local integration adapter, not a multi-tenant production service. Keep it on a trusted development machine until authentication, scoped retrieval, rate limits, and persistent indexing are implemented.

## Evaluate

```bash
python -m eval.run_eval
python -m eval.run_eval --llm anthropic
python -m eval.run_eval --store chroma
```

The 16-question evaluation set measures hit@k, MRR, answer accuracy, citation precision, and abstention accuracy.

## Run the Gradio UI

```bash
python app.py
```

Open http://localhost:7860. Click **Load sample documents** to try the included runbooks.

## Tests

```bash
pip install -r requirements-ci.txt
pytest -q
```

Offline tests use the deterministic hash embedder and do not need model downloads or API credentials.
