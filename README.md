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
question -> retrieve -> grade -> rewrite (when needed) -> extractive answer -> validate citations
```

See the existing rag package for the pipeline and eval directory for the labeled retrieval/answer regression set.

## HTTP API for Nexus AI

A small FastAPI adapter exposes the existing DocQA engine. It is a separate local process from the Gradio demo; it uses the same implementation under rag.

From the repository root:

**Windows PowerShell**

```powershell
$env:EMBEDDER = "minilm"
uvicorn api:app --host 0.0.0.0 --port 7861
```

Use EMBEDDER=minilm for normal use. MiniLM provides semantic vector retrieval. The hash embedder is only a quick deterministic option for offline CI or smoke tests; it is not a semantic model. Retrieval combines local MiniLM vectors with BM25-style lexical ranking to improve exact technology, acronym, and name matching. Answer generation is fully local and deterministic: it selects relevant sentences from retrieved passages, adds source citations, and abstains when retrieval finds no evidence. No hosted LLM API keys or external inference service are used.

Endpoints:

- GET /health — process liveness, no model load.
- GET /ready — initialize/check the RAG engine and return model/index details.
- GET /api/v1/documents — list sample and uploaded documents in this process.
- POST /api/v1/documents — index a PDF, DOCX, TXT, or Markdown file (10 MiB maximum).
- POST /api/v1/ask — answer a question with citations, trace, and latency.
- POST /api/v1/retrieve — diagnostic top passages with source/page and retrieval score (useful when debugging a document that was indexed but is not being retrieved).

Open http://localhost:7861/docs for interactive API documentation.

The vector index is rebuilt in memory at service startup. Uploaded source files are stored on disk in `.data/uploads` by default (or in `RAG_UPLOAD_DIR`) and re-indexed after a restart. Nexus Docker Compose mounts a named volume for uploads and a separate volume for the MiniLM model cache. The service is still a local integration adapter, not a multi-tenant production service: keep it on a trusted development machine until authentication, scoped retrieval, service authentication, and rate limits are implemented.

## Evaluate

```bash
python -m eval.run_eval
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
