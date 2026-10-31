"""Local, deterministic answer extraction from retrieved document passages."""
from __future__ import annotations

import re

_STOP = set(
    "a an the is are was were be been of to in on for and or with what which who whom how why when where "
    "does do did can could should would i we you it this that these those my our your about tell me please "
    "list summarize summary give show describe explain identify provide include including from into onto "
    "used use using has have had do does their they them there here"
    .split()
)

# A deliberately finite vocabulary helps broad resume/skills questions find evidence
# without guessing. Only terms found in retrieved source text can appear in answers.
_TECH_TERMS = set("""
    aws azure gcp google cloud amazon microsoft python java javascript typescript go golang rust
    c++ c# dotnet net bash powershell sql mysql postgresql postgres mongodb redis sqlite
    docker kubernetes k8s terraform ansible pulumi cloudformation helm jenkins github gitlab
    github actions gitlab ci/cd linux windows macos nginx prometheus grafana datadog splunk
    kafka spark airflow snowflake databricks pandas numpy pytorch tensorflow scikit-learn
    react nextjs nodejs node fastapi flask django spring .net rest graphql api
    devops sre cicd automation orchestration infrastructure cloudformation azure devops
    azure functions lambda ec2 s3 iam vpc eks aks gke openshift
""".split())

_EXPERIENCE_TERMS = set("""
    experience professional employment internship intern engineer developer analyst architect
    administrator responsibilities responsible worked work leading led managed management
    designed developed implemented maintained supported operated collaborated delivered
""".split())

_PROJECT_TERMS = set("""
    project projects built developed implemented designed created deployed application system
    platform prototype architecture pipeline dashboard website service solution
""".split())

_TOKEN_RE = re.compile(
    r"(?<![a-z0-9])(?:\.[a-z0-9]+|[a-z0-9]+(?:[+#./-][a-z0-9+#./-]*)*)(?![a-z0-9])",
    re.I,
)
_PASSAGE_RE = re.compile(r"(?ms)^\[(\d+)\]\s*(.*?)(?=^\[\d+\]\s*|\Z)")


class ExtractiveLLM:
    """Answer by selecting source sentences; never calls a hosted generation service."""

    name = "extractive"
    max_sentences = 5

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            token.lower().strip(".")
            for token in _TOKEN_RE.findall(text)
            if token.lower().strip(".") not in _STOP and len(token.strip(".").lower()) > 1
        }

    @staticmethod
    def _is_technical_question(question: str) -> bool:
        lowered = question.lower()
        return any(
            term in lowered
            for term in (
                "skill", "technolog", "tool", "tech stack", "toolkit", "platform",
                "programming", "language", "framework", "database", "devops",
                "cloud", "infrastructure",
            )
        )

    @staticmethod
    def _is_experience_question(question: str) -> bool:
        lowered = question.lower()
        return any(term in lowered for term in ("experience", "employment", "work history", "career", "background"))

    @staticmethod
    def _is_project_question(question: str) -> bool:
        return "project" in question.lower()

    @staticmethod
    def _passages(user: str) -> list[tuple[str, str]]:
        context = user.split("QUESTION:", 1)[0]
        if context.startswith("CONTEXT:"):
            context = context[len("CONTEXT:"):]
        return [(match.group(1), match.group(2).strip()) for match in _PASSAGE_RE.finditer(context)]

    @staticmethod
    def _sentences(body: str) -> list[str]:
        # Keep each bullet/line intact (important for comma-separated skill lists),
        # while splitting normal prose into independently citable sentences.
        cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", body)
        pieces = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
        sentences = []
        for piece in pieces:
            sentence = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", piece).strip()
            if len(sentence) >= 12:
                sentences.append(sentence)
        return sentences

    def complete(self, system: str, user: str) -> str:
        question = user.split("QUESTION:", 1)[-1].strip()
        base_terms = self._terms(question)
        query_terms = set(base_terms)
        technical = self._is_technical_question(question)
        experience = self._is_experience_question(question)
        projects = self._is_project_question(question)

        if technical:
            query_terms.update(_TECH_TERMS)
        if experience:
            query_terms.update(_EXPERIENCE_TERMS)
        if projects:
            query_terms.update(_PROJECT_TERMS)

        # Topic questions can legitimately match a specific named technology,
        # role, or project verb even when the user's wording differs.
        topic_mode = technical or experience or projects
        required_overlap = 1 if topic_mode else min(2, len(base_terms))
        ranked: list[tuple[int, float, int, int, str, str]] = []

        for passage_rank, (citation, body) in enumerate(self._passages(user)):
            for sentence_rank, sentence in enumerate(self._sentences(body)):
                sentence_terms = self._terms(sentence)
                overlap = query_terms & sentence_terms
                if len(overlap) < required_overlap:
                    continue
                # Prefer evidence with more question-term coverage, but do not
                # favor tiny headings over informative sentences.
                coverage = len(overlap) / max(1, len(query_terms))
                density = len(overlap) / max(1, len(sentence_terms) ** 0.5)
                ranked.append(
                    (len(overlap), coverage + density, -passage_rank, -sentence_rank, sentence, citation)
                )

        if not ranked:
            # Retrieval and grading already admitted these passages. In local-only
            # mode, expose a clearly labeled source excerpt rather than inventing
            # a paraphrase when the wording does not overlap.
            for citation, body in self._passages(user):
                sentences = self._sentences(body)
                if sentences:
                    return f"Relevant passage: {sentences[0]} [{citation}]"
            return "I couldn't find that in the uploaded documents."

        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
        selected: list[tuple[str, str]] = []
        seen: set[str] = set()
        for _, _, _, _, sentence, citation in ranked:
            key = re.sub(r"\W+", " ", sentence.lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            selected.append((sentence, citation))
            if len(selected) >= self.max_sentences:
                break

        return "Relevant details from the uploaded documents:\n" + "\n".join(
            f"- {sentence} [{citation}]" for sentence, citation in selected
        )

    def rewrite(self, question: str) -> str:
        return " ".join(sorted(self._terms(question), key=question.lower().find)) or question


def make_llm(provider: str = "extractive"):
    """Return the local-only answer engine; hosted inference providers are unsupported."""
    normalized = provider.strip().lower()
    if normalized not in {"extractive", "auto"}:
        raise ValueError("Only the local extractive answer engine is supported.")
    return ExtractiveLLM()
