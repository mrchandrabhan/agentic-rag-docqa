from pathlib import Path

from rag import DocQA
from rag.embeddings import HashEmbedder


def test_hybrid_retrieval_finds_exact_resume_skills_with_hash_embedder(tmp_path: Path):
    resume = tmp_path / "resume.txt"
    resume.write_text(
        "Infrastructure as Code: Terraform and Ansible. "
        "Container orchestration using Kubernetes. "
        "Cloud administration on Microsoft Azure.",
        encoding="utf-8",
    )
    qa = DocQA(
        embedder=HashEmbedder(),
        store="faiss",
        chunk_size=300,
        chunk_overlap=30,
    )
    qa.ingest([resume])

    hits = qa.retrieve("Which tools are used for infrastructure as code?")
    assert hits
    assert hits[0][0].source == "resume.txt"
    assert "Terraform" in hits[0][0].text
    assert "Ansible" in hits[0][0].text

    answer = qa.ask("Which tools are used for infrastructure as code?")
    assert "Terraform" in answer["answer"]
    assert answer["citations"][0]["source"] == "resume.txt"


def test_hybrid_retrieval_still_abstains_for_unrelated_question(tmp_path: Path):
    runbook = tmp_path / "runbook.txt"
    runbook.write_text(
        "The Kubernetes deployment is rolled back when a release causes an outage.",
        encoding="utf-8",
    )
    qa = DocQA(embedder=HashEmbedder(), store="faiss")
    qa.ingest([runbook])

    result = qa.ask("What is the parental leave policy?")
    assert result["answer"] == "I couldn't find that in the uploaded documents."
    assert result["citations"] == []
