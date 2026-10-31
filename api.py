"""HTTP adapter that exposes the existing DocQA engine to Nexus AI."""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field, field_validator

from rag import DocQA
from rag.embeddings import make_embedder
from rag.llm import make_llm
from rag.loaders import SUPPORTED


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
CORPUS_DIR = Path(__file__).parent / "eval" / "corpus"


class NoExtractableTextError(ValueError):
    """Upload parsed, but contained no text chunks that can be indexed."""


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Question cannot be blank.")
        return normalized


class RetrieveRequest(AskRequest):
    limit: int = Field(default=8, ge=1, le=20)


class RAGEngine:
    """Process-local index over the existing agentic RAG implementation."""

    def __init__(self) -> None:
        embedder_kind = os.getenv("EMBEDDER", "minilm")
        self._lock = threading.RLock()
        self.qa = DocQA(
            embedder=make_embedder(embedder_kind),
            store=os.getenv("RAG_STORE", "faiss"),
            llm=make_llm(),
            log_path=os.getenv("RAG_QUERY_LOG", "logs/nexus-queries.jsonl"),
        )
        self.upload_root = Path(tempfile.mkdtemp(prefix="agentic-rag-api-"))
        self.documents: dict[str, dict[str, Any]] = {}

        sample_files = sorted(
            path for path in CORPUS_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED
        )
        if sample_files:
            total_stats = self.qa.ingest(sample_files)
            per_file_chunks = max(
                1, total_stats["chunks"] // len(sample_files)
            )
            for path in sample_files:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.documents[digest] = {
                    "filename": path.name,
                    "sha256": digest,
                    "source": "sample",
                    "chunks": per_file_chunks,
                }

    def ingest_bytes(self, filename: str, content: bytes) -> dict[str, Any]:
        extension = PurePosixPath(filename.lower()).suffix
        if extension not in SUPPORTED:
            raise ValueError("Supported file types are PDF, DOCX, TXT, and Markdown.")
        if not content:
            raise ValueError("The uploaded file is empty.")

        digest = hashlib.sha256(content).hexdigest()
        safe_name = PurePosixPath(filename.replace("\\", "/")).name
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("A valid filename is required.")

        destination = self.upload_root / str(uuid.uuid4()) / safe_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        with self._lock:
            if digest in self.documents:
                raise FileExistsError("This exact document has already been indexed.")
            stats = self.qa.ingest([destination])
            if stats.get("chunks", 0) < 1 or stats.get("pages", 0) < 1:
                destination.unlink(missing_ok=True)
                try:
                    destination.parent.rmdir()
                except OSError:
                    pass
                raise NoExtractableTextError(
                    "No readable text was extracted. This may be an image-only scanned PDF; OCR is not enabled."
                )
            self.documents[digest] = {
                "filename": safe_name,
                "sha256": digest,
                "source": "upload",
                "chunks": stats["chunks"],
            }
            return {"filename": safe_name, "sha256": digest, **stats}

    def ask(self, question: str) -> dict[str, Any]:
        with self._lock:
            result = self.qa.ask(question)
        return {
            "question": result["question"],
            "answer": result["answer"],
            "citations": result["citations"],
            "trace": result["trace"],
            "latency_ms": result["latency_ms"],
            "engine": "agentic-rag-docqa",
        }

    def retrieve(self, question: str, limit: int = 8) -> dict[str, Any]:
        with self._lock:
            hits = self.qa.retrieve(question, k=limit)
            return {
                "question": question,
                "embedding_model": self.qa.embedder.model_name,
                "answer_model": self.qa.llm.name,
                "results": [
                    {
                        "rank": rank,
                        "source": chunk.source,
                        "page": chunk.page,
                        "score": round(float(score), 4),
                        "text": chunk.text[:1200],
                    }
                    for rank, (chunk, score) in enumerate(hits, start=1)
                ],
            }

    def list_documents(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "filename": item["filename"],
                    "source": item["source"],
                    "chunks": item.get("chunks", 0),
                }
                for item in self.documents.values()
            ]


engine_init_lock = threading.Lock()


def get_qa(request: Request) -> RAGEngine:
    engine = getattr(request.app.state, "rag_engine", None)
    if engine is None:
        with engine_init_lock:
            engine = getattr(request.app.state, "rag_engine", None)
            if engine is None:
                try:
                    engine = RAGEngine()
                except Exception as exc:
                    raise HTTPException(
                        status_code=503,
                        detail="RAG engine could not initialize. Check model and dependency configuration.",
                    ) from exc
                request.app.state.rag_engine = engine
    return engine


app = FastAPI(
    title="Agentic RAG Service API",
    description="HTTP adapter for the existing hybrid retrieval and citation engine.",
    version="1.1.0",
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "agentic-rag-docqa-api"}


@app.get("/ready", tags=["system"])
def ready(engine: RAGEngine = Depends(get_qa)) -> dict[str, Any]:
    return {
        "status": "ready",
        "engine": "agentic-rag-docqa",
        "embedding_model": engine.qa.embedder.model_name,
        "answer_model": engine.qa.llm.name,
        "retrieval": "hybrid-vector-and-lexical",
        "indexed_documents": len(engine.documents),
        "indexed_chunks": len(engine.qa.store),
    }


@app.get("/api/v1/documents", tags=["documents"])
def list_documents(engine: RAGEngine = Depends(get_qa)) -> dict[str, Any]:
    return {"documents": engine.list_documents()}


@app.post("/api/v1/documents", tags=["documents"])
async def upload_document(
    file: UploadFile = File(...),
    engine: RAGEngine = Depends(get_qa),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=422, detail="A filename is required.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Document exceeds the 10 MiB upload limit.")
    try:
        return engine.ingest_bytes(file.filename, content)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except NoExtractableTextError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail="The document could not be parsed and indexed.") from exc


@app.post("/api/v1/retrieve", tags=["retrieval"])
def retrieve(request: RetrieveRequest, engine: RAGEngine = Depends(get_qa)) -> dict[str, Any]:
    return engine.retrieve(request.question, request.limit)


@app.post("/api/v1/ask", tags=["retrieval"])
def ask(request: AskRequest, engine: RAGEngine = Depends(get_qa)) -> dict[str, Any]:
    try:
        return engine.ask(request.question)
    except Exception as exc:
        raise HTTPException(status_code=502, detail="The RAG engine failed to answer the question.") from exc
