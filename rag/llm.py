"""Generation backends. Each exposes complete(system, user) -> str and rewrite(question) -> str."""
from __future__ import annotations

import os
import re

_STOP = set(
    "a an the is are was were be been of to in on for and or with what which who whom how why when where "
    "does do did can could should would i we you it this that these those my our your about tell me please".split()
)

REWRITE_SYSTEM = (
    "Rewrite the user's question as a short keyword-rich search query for a document index. "
    "Keep names, numbers and technical terms. Return only the query."
)


class AnthropicLLM:
    """Claude via LangChain's ChatAnthropic. Needs ANTHROPIC_API_KEY."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from langchain_anthropic import ChatAnthropic

        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
        self.chat = ChatAnthropic(model=self.model, api_key=api_key or os.environ["ANTHROPIC_API_KEY"],
                                  temperature=0, max_tokens=700)

    def complete(self, system: str, user: str) -> str:
        return self.chat.invoke([("system", system), ("human", user)]).content.strip()

    def rewrite(self, question: str) -> str:
        return self.complete(REWRITE_SYSTEM, question)


class HuggingFaceLLM:
    """Open-weight model via the Hugging Face Inference API. Needs an HF token."""

    name = "huggingface"

    def __init__(self, token: str | None = None, model: str | None = None):
        from huggingface_hub import InferenceClient

        self.model = model or os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
        self.client = InferenceClient(model=self.model, token=token or os.environ["HF_TOKEN"])

    def complete(self, system: str, user: str) -> str:
        out = self.client.chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=700, temperature=0.1,
        )
        return out.choices[0].message.content.strip()

    def rewrite(self, question: str) -> str:
        return self.complete(REWRITE_SYSTEM, question)


class ExtractiveLLM:
    """No-key fallback: answers with the best-matching sentences from the retrieved context.

    Keeps the demo usable without an API key and gives tests a deterministic generator.
    """

    name = "extractive"

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1}

    def complete(self, system: str, user: str) -> str:
        question = user.split("QUESTION:", 1)[-1].strip()
        q = self._terms(question)
        need = min(2, len(q))  # require real overlap so unrelated questions abstain
        scored = []
        for m in re.finditer(r"\[(\d+)\] (.+?)(?=\n\[\d+\] |\n\nQUESTION:|\Z)", user, re.S):
            n, body = m.group(1), m.group(2)
            body = re.sub(r"^#+\s*.*$", "", body, flags=re.M)  # drop markdown headings
            for sent in re.split(r"(?<=[.!?])\s+", body.replace("\n", " ")):
                overlap = len(q & self._terms(sent))
                if overlap >= need:
                    scored.append((overlap, sent.strip(), n))
        if not scored:
            # The retrieval graph has already filtered the context for relevance.
            # In no-key mode we cannot safely paraphrase synonyms, so expose the
            # leading source sentence instead of silently discarding useful evidence.
            for match in re.finditer(r"\[(\d+)\] (.+?)(?=\n\[\d+\] |\n\nQUESTION:|\Z)", user, re.S):
                citation, body = match.group(1), match.group(2)
                body = re.sub(r"^#+\s*.*$", "", body, flags=re.M)
                sentences = [
                    sentence.strip()
                    for sentence in re.split(r"(?<=[.!?])\s+", body.replace("\n", " "))
                    if len(sentence.strip()) >= 25
                ]
                if sentences:
                    return f"Relevant passage: {sentences[0]} [{citation}]"
            return "I couldn't find that in the uploaded documents."
        scored.sort(key=lambda x: -x[0])
        return " ".join(f"{s} [{n}]" for _, s, n in scored[:2])

    def rewrite(self, question: str) -> str:
        return " ".join(sorted(self._terms(question), key=question.lower().find)) or question


def make_llm(provider: str = "auto", api_key: str | None = None):
    provider = provider.lower()
    if provider == "auto":
        if api_key and api_key.startswith("sk-ant") or os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif api_key and api_key.startswith("hf_") or os.getenv("HF_TOKEN"):
            provider = "huggingface"
        else:
            provider = "extractive"
    if provider == "anthropic":
        return AnthropicLLM(api_key=api_key if api_key and api_key.startswith("sk-ant") else None)
    if provider == "huggingface":
        return HuggingFaceLLM(token=api_key if api_key and api_key.startswith("hf_") else None)
    return ExtractiveLLM()
