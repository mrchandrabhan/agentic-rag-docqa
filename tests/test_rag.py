"""Offline test suite: no model downloads, no API keys (HashEmbedder + fake/extractive LLMs)."""
from pathlib import Path

import pytest

from rag import DocQA
from rag.chunking import chunk_pages
from rag.embeddings import HashEmbedder
from rag.graph import NO_ANSWER
from rag.llm import ExtractiveLLM, make_llm
from rag.loaders import Page, load_file

CORPUS = Path(__file__).parent.parent / "eval" / "corpus"


class FakeLLM:
    """Scriptable LLM that records prompts so tests can assert on them."""

    name = "fake"

    def __init__(self, answer="Answer [1].", rewrite_to=None):
        self.answer, self.rewrite_to, self.prompts, self.rewrites = answer, rewrite_to, [], 0

    def complete(self, system, user):
        self.prompts.append((system, user))
        return self.answer

    def rewrite(self, question):
        self.rewrites += 1
        return self.rewrite_to or question


@pytest.fixture
def qa():
    q = DocQA(embedder=HashEmbedder(), store="faiss", llm=ExtractiveLLM(), chunk_size=300, chunk_overlap=50)
    q.ingest(sorted(CORPUS.iterdir()))
    return q


# ---------- loaders ----------
def test_load_txt_and_md(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# Title\nhello world")
    pages = load_file(f)
    assert len(pages) == 1 and pages[0].source == "a.md" and pages[0].page == 1


def test_unsupported_type_rejected(tmp_path):
    f = tmp_path / "a.xlsx"
    f.write_text("x")
    with pytest.raises(ValueError):
        load_file(f)


def test_empty_file_produces_no_pages(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("   ")
    assert load_file(f) == []


def test_pdf_pages_keep_page_numbers(tmp_path):
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(200, 200)
    w.add_blank_page(200, 200)
    p = tmp_path / "blank.pdf"
    with p.open("wb") as fh:
        w.write(fh)
    assert load_file(p) == []  # blank pages have no text and are skipped, not crashed on


# ---------- chunking ----------
def test_chunks_carry_metadata_and_respect_size():
    pages = [Page("word " * 400, "doc.txt", 3)]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.source == "doc.txt" and c.page == 3 and len(c.text) <= 200 for c in chunks)
    assert [c.id for c in chunks] == list(range(len(chunks)))


def test_citation_format():
    c = chunk_pages([Page("abc", "x.pdf", 7)])[0]
    assert c.citation() == "x.pdf, p.7"


# ---------- embeddings & stores ----------
def test_hash_embedder_is_normalized_and_deterministic():
    e = HashEmbedder()
    a, b = e.embed(["kafka partitions"]), e.embed(["kafka partitions"])
    assert abs(float((a * a).sum()) - 1.0) < 1e-5
    assert (a == b).all()


def test_faiss_returns_most_similar_first(qa):
    hits = qa.retrieve("How many partitions does the click events topic have?")
    assert hits[0][0].source == "kafka_pipeline_guide.txt"
    assert hits == sorted(hits, key=lambda h: -h[1])


def test_ids_stay_unique_across_multiple_ingests(tmp_path):
    q = DocQA(embedder=HashEmbedder(), llm=ExtractiveLLM())
    (tmp_path / "a.txt").write_text("alpha beta gamma")
    (tmp_path / "b.txt").write_text("delta epsilon")
    q.ingest([tmp_path / "a.txt"])
    q.ingest([tmp_path / "b.txt"])
    ids = [c.id for c in q.store.chunks]
    assert len(ids) == len(set(ids)) and q.sources == ["a.txt", "b.txt"]


def test_empty_store_search_is_safe():
    q = DocQA(embedder=HashEmbedder(), llm=ExtractiveLLM())
    assert q.retrieve("anything") == []


# ---------- agentic graph ----------
def test_answer_has_valid_page_citations(qa):
    res = qa.ask("When does the locker code expire?")
    assert "48 hours" in res["answer"]
    assert res["citations"] and res["citations"][0]["source"] == "pantry_locker_policy.md"
    assert res["citations"][0]["page"] == 1


def test_trace_records_each_graph_step(qa):
    steps = [t.split(":")[0] for t in qa.ask("What is the SEV2 acknowledgement time?")["trace"]]
    assert steps[:2] == ["retrieve", "grade"] and steps[-1] == "cite"


def test_out_of_range_citations_are_dropped(qa):
    qa.set_llm(FakeLLM(answer="Made up [1][99]."))
    res = qa.ask("How long is raw topic data retained?")
    assert [c["n"] for c in res["citations"]] == [1]
    assert "dropped 1 invalid" in res["trace"][-1]


def test_weak_retrieval_triggers_one_rewrite_then_abstains():
    q = DocQA(embedder=HashEmbedder(), llm=None)
    fake = FakeLLM(rewrite_to="still nothing relevant")
    q.set_llm(fake)
    res = q.ask("anything at all")  # empty store -> nothing passes grading
    assert fake.rewrites == 1
    assert res["answer"] == NO_ANSWER and res["citations"] == []
    assert any(t.startswith("rewrite") for t in res["trace"])


def test_rewrite_recovers_when_query_improves(qa):
    qa.min_score = 0.01  # one lexical RRF result passes after the query is rewritten
    qa.semantic_min_score = 1.1  # isolate the lexical rewrite path in this graph test
    qa._graph = None
    fake = FakeLLM(answer="Twelve [1].", rewrite_to="click events topic web.clicks.v1 has 12 partitions keyed by user id")
    qa.set_llm(fake)
    res = qa.ask("qzxvblarg123")
    assert fake.rewrites == 1 and res["citations"]


def test_prompt_contains_only_graded_context(qa):
    fake = FakeLLM()
    qa.set_llm(fake)
    qa.ask("Where do failed messages go after three deserialization failures?")
    system, user = fake.prompts[0]
    assert "ONLY the numbered context" in system
    assert "[1]" in user and "QUESTION:" in user


def test_unanswerable_question_abstains(qa):
    assert qa.ask("What is the parental leave policy?")["answer"] == NO_ANSWER


def test_jsonl_query_log(tmp_path):
    log = tmp_path / "q.jsonl"
    q = DocQA(embedder=HashEmbedder(), llm=ExtractiveLLM(), log_path=str(log), chunk_size=300)
    q.ingest(sorted(CORPUS.iterdir()))
    q.ask("When does the locker code expire?")
    import json

    rec = json.loads(log.read_text().splitlines()[0])
    assert {"question", "answer", "trace", "latency_ms", "sources"} <= rec.keys()


# ---------- llm selection ----------
def test_make_llm_stays_local_even_if_external_tokens_are_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-used")
    monkeypatch.setenv("HF_TOKEN", "not-used")
    assert make_llm("auto").name == "extractive"
    assert make_llm("extractive").name == "extractive"

    with pytest.raises(ValueError, match="local extractive"):
        make_llm("hosted")


# ---------- eval regression gate ----------
def test_eval_meets_baseline():
    from eval.run_eval import evaluate

    r = evaluate(embedder="hash")
    assert r["hit@4"] >= 0.9
    assert r["mrr"] >= 0.75
    assert r["abstention_accuracy"] == 1.0


def test_chroma_store_matches_faiss_top_hit():
    pytest.importorskip("chromadb")
    q = DocQA(embedder=HashEmbedder(), store="chroma", llm=ExtractiveLLM(), chunk_size=300, chunk_overlap=50)
    q.ingest(sorted(CORPUS.iterdir()))
    hits = q.retrieve("How many partitions does the click events topic have?")
    assert hits[0][0].source == "kafka_pipeline_guide.txt"
    assert "48 hours" in q.ask("When does the locker code expire?")["answer"]



def test_extractive_fallback_surfaces_ranked_passage_when_question_uses_synonyms():
    prompt = (
        "CONTEXT:\n"
        "[1] Skills include Python, Terraform, Ansible, Azure and Kubernetes. "
        "The candidate has automated infrastructure deployments.\n\n"
        "QUESTION: Which technologies are part of this person's technical toolkit?"
    )
    answer = ExtractiveLLM().complete("Use only the context.", prompt)
    assert answer.startswith("Relevant details from the uploaded documents:")
    assert "Terraform" in answer and "Kubernetes" in answer
    assert answer.endswith("[1]")


def test_extractive_answer_returns_multiple_grounded_technology_details():
    prompt = (
        "CONTEXT:\n"
        "[1] Technical Skills: Python, C++, SQL, Terraform, Kubernetes, Docker, and Microsoft Azure.\n"
        "[2] Built a CI/CD release pipeline with GitHub Actions and deployed services to AWS.\n\n"
        "QUESTION: Which cloud platforms, DevOps tools, programming languages, and infrastructure technologies are listed?"
    )
    answer = ExtractiveLLM().complete("Use only the context.", prompt)
    assert "Python" in answer and "C++" in answer and "Terraform" in answer
    assert "Kubernetes" in answer and "Azure" in answer and "AWS" in answer
    assert "[1]" in answer and "[2]" in answer
    assert answer.count("\n- ") >= 1


def test_extractive_answer_preserves_exact_acronyms_and_versions():
    prompt = (
        "CONTEXT:\n"
        "[1] The service uses C++, C#, CI/CD, and .NET 8 for its build and deployment workflow.\n\n"
        "QUESTION: Which programming and deployment technologies are mentioned?"
    )
    answer = ExtractiveLLM().complete("Use only the context.", prompt)
    assert "C++" in answer and "C#" in answer and "CI/CD" in answer
    assert ".NET 8" in answer and "[1]" in answer
