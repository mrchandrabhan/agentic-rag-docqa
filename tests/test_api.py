from fastapi.testclient import TestClient

from api import NoExtractableTextError, RAGEngine, app, get_qa


class FakeEngine:
    def __init__(self):
        self.indexed = []

    def ask(self, question):
        return {
            "question": question,
            "answer": "The incident acknowledgement target is five minutes [1].",
            "citations": [
                {
                    "n": 1,
                    "source": "incident-runbook.md",
                    "page": 1,
                    "score": 0.91,
                    "text": "Acknowledge a production incident within five minutes.",
                }
            ],
            "trace": ["retrieve: 1 chunks", "grade: 1/1", "generate: extractive", "cite: 1 citations"],
            "latency_ms": 8.2,
            "engine": "agentic-rag-docqa",
        }

    def ingest_bytes(self, filename, content):
        self.indexed.append((filename, content))
        return {"filename": filename, "files": 1, "pages": 1, "chunks": 1}

    def list_documents(self):
        return [{"filename": name, "source": "upload", "chunks": 1} for name, _ in self.indexed]

    def retrieve(self, question, limit=8):
        return {
            "question": question,
            "embedding_model": "hash-1024",
            "answer_model": "extractive",
            "results": [
                {
                    "rank": 1,
                    "source": "incident-runbook.md",
                    "page": 1,
                    "score": 0.02,
                    "text": "Acknowledge a production incident within five minutes.",
                }
            ][:limit],
        }


def setup_client():
    fake = FakeEngine()
    app.dependency_overrides[get_qa] = lambda: fake
    return TestClient(app), fake


def teardown_client(client):
    client.close()
    app.dependency_overrides.clear()


def test_health_is_live_without_loading_model():
    client = TestClient(app)
    try:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    finally:
        client.close()


def test_ask_returns_answer_with_page_citations():
    client, _ = setup_client()
    try:
        response = client.post("/api/v1/ask", json={"question": "What is the incident acknowledgement target?"})
        assert response.status_code == 200
        body = response.json()
        assert "five minutes" in body["answer"]
        assert body["citations"][0]["source"] == "incident-runbook.md"
        assert body["citations"][0]["page"] == 1
    finally:
        teardown_client(client)


def test_upload_and_list_document():
    client, fake = setup_client()
    try:
        response = client.post(
            "/api/v1/documents",
            files={"file": ("runbook.md", b"# Runbook\nAcknowledge incidents quickly.", "text/markdown")},
        )
        assert response.status_code == 200
        assert response.json()["chunks"] == 1
        listing = client.get("/api/v1/documents")
        assert listing.json()["documents"][0]["filename"] == "runbook.md"
        assert len(fake.indexed) == 1
    finally:
        teardown_client(client)


def test_empty_question_is_rejected():
    client, _ = setup_client()
    try:
        response = client.post("/api/v1/ask", json={"question": "   "})
        assert response.status_code == 422
    finally:
        teardown_client(client)


def test_retrieve_endpoint_exposes_model_and_source_debug_details():
    client, _ = setup_client()
    try:
        response = client.post(
            "/api/v1/retrieve",
            json={"question": "What is the incident acknowledgement target?", "limit": 5},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["embedding_model"] == "hash-1024"
        assert body["results"][0]["source"] == "incident-runbook.md"
        assert body["results"][0]["page"] == 1
    finally:
        teardown_client(client)



def test_empty_extraction_is_not_registered_as_indexed(tmp_path):
    class EmptyQA:
        def ingest(self, paths):
            return {"files": 0, "pages": 0, "chunks": 0, "characters": 0}

    engine = RAGEngine.__new__(RAGEngine)
    engine.qa = EmptyQA()
    engine._lock = __import__("threading").RLock()
    engine.upload_root = tmp_path
    engine.documents = {}

    import pytest
    with pytest.raises(NoExtractableTextError, match="No readable text"):
        engine.ingest_bytes("scanned-resume.pdf", b"%PDF-1.7 image-only content")
    assert engine.documents == {}
