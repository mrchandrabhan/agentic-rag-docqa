"""HTTP adapter that exposes the existing DocQA engine to Nexus AI."""
from __future__ import annotations

import hashlib
import logging
import os
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


logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
CORPUS_DIR = Path(__file__).parent / "eval" / "corpus"
DEFAULT_UPLOAD_DIR = Path(__file__).parent / ".data" / "uploads"


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
    """Hybrid RAG engine with persistent uploaded source files and a rebuilt in-memory index."""

    def __init__(self) -> None:
        embedder_kind = os.getenv("EMBEDDER", "minilm")
        self._lock = threading.RLock()
        self.qa = DocQA(
            embedder=make_embedder(embedder_kind),
            store=os.getenv("RAG_STORE", "faiss"),
            llm=make_llm(),
            log_path=os.getenv("RAG_QUERY_LOG", "logs/nexus-queries.jsonl"),
        )
        upload_dir = Path(os.getenv("RAG_UPLOAD_DIR", str(DEFAULT_UPLOAD_DIR))).expanduser()
        self.upload_root = upload_dir.resolve()
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self.documents: dict[str, dict[str, Any]] = {}

        sample_files = sorted(
            path for path in CORPUS_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED
        )
        if sample_files:
            self.qa.ingest(sample_files)
            for path in sample_files:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.documents[digest] = {
                    "filename": path.name,
                    "sha256": digest,
                    "source": "sample",
                    "chunks": sum(
                        1 for chunk in self.qa.store.chunks if chunk.source == path.name
                    ),
                }

        self._restore_uploaded_documents()

    def _restore_uploaded_documents(self) -> None:
        """Re-index files from the persistent upload directory after a service restart."""
        for path in sorted(self.upload_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                continue
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest in self.documents:
                    continue
                stats = self.qa.ingest([path])
                if stats.get("chunks", 0) < 1 or stats.get("pages", 0) < 1:
                    logger.warning("Skipping persisted upload with no extractable text: %s", path.name)
                    continue
                self.documents[digest] = {
                    "filename": path.name,
                    "sha256": digest,
                    "source": "upload",
                    "chunks": stats["chunks"],
                }
            except Exception:
                logger.exception("Could not restore persisted upload: %s", path.name)

    @staticmethod
    def _remove_upload(destination: Path) -> None:
        destination.unlink(missing_ok=True)
        try:
            destination.parent.rmdir()
        except OSError:
            pass

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

        with self._lock:
            if digest in self.documents:
                raise FileExistsError("This exact document has already been indexed.")

            destination = self.upload_root / str(uuid.uuid4()) / safe_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            try:
                stats = self.qa.ingest([destination])
                if stats.get("chunks", 0) < 1 or stats.get("pages", 0) < 1:
                    raise NoExtractableTextError(
                        "No readable text was extracted. This may be an image-only scanned PDF; OCR is not enabled."
                    )
            except Exception:
                self._remove_upload(destination)
                raise

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
    description="HTTP adapter for the existing hybrid retrieval and local citation engine.",
    version="1.2.0",
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
        "upload_persistence": "disk-backed; reindexed on startup",
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
