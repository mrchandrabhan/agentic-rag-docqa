#!/usr/bin/env bash
# One-shot publish from your own computer: creates the GitHub repo, pushes, and deploys the Hugging Face Space.
# Needs: git, GitHub CLI (`gh auth login` once), Python 3 (you'll be asked for a Hugging Face Write token the first time).
#   ./publish.sh               # repo + Space both named agentic-rag-docqa
#   ./publish.sh my-other-name
set -euo pipefail
NAME="${1:-agentic-rag-docqa}"
cd "$(dirname "$0")"

command -v gh >/dev/null || { echo "Install the GitHub CLI first: https://cli.github.com"; exit 1; }
gh auth status >/dev/null 2>&1 || gh auth login

# 1) GitHub
[ -d .git ] || { git init -q -b main; git add -A; git commit -qm "Initial commit: agentic RAG document Q&A"; }
if ! git remote get-url origin >/dev/null 2>&1; then
  gh repo create "$NAME" --public --source . --remote origin \
    --description "Agentic RAG document Q&A: LangGraph, FAISS/ChromaDB, Claude/Mistral, page-level citations, labeled eval in CI"
fi
git push -u origin main
GH_USER="$(gh api user -q .login)"
gh repo edit "$GH_USER/$NAME" --add-topic rag,langgraph,langchain,faiss,chromadb,llm,gradio,huggingface,agentic-ai >/dev/null || true

# 2) Hugging Face Space
python3 -m pip install -q "huggingface_hub>=1.0"
python3 scripts/deploy_space.py "$NAME"
echo "Code:  https://github.com/$GH_USER/$NAME"
echo "Tip: add an HF_TOKEN repo secret on GitHub so every push to main redeploys the Space automatically."
