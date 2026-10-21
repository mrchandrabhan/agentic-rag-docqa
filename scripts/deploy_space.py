"""Create (if needed) and upload the Hugging Face Space. Used by publish.ps1, publish.sh and GitHub Actions.

python scripts/deploy_space.py [space-name]
Token comes from HF_TOKEN, a saved `hf auth login`, or a prompt (paste a Write token).
"""
import getpass
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, get_token, login

IGNORE = [".git/*", ".github/*", "tests/*", "logs/*", "scripts/*", "**/__pycache__/*",
          ".pytest_cache/*", ".models/*", "publish.sh", "publish.ps1", "render.yaml"]


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else os.getenv("SPACE_NAME", "agentic-rag-docqa")
    token = os.getenv("HF_TOKEN") or get_token()
    if not token:
        print("Paste a Hugging Face token with WRITE access (https://huggingface.co/settings/tokens):")
        token = getpass.getpass("HF token: ").strip()
        login(token=token, add_to_git_credential=False)

    api = HfApi(token=token)
    repo_id = f"{api.whoami()['name']}/{name}"
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id, repo_type="space", folder_path=str(Path(__file__).resolve().parent.parent),
        ignore_patterns=IGNORE,
        commit_message=f"Deploy {os.getenv('GITHUB_SHA', 'local')[:7]}",
    )
    print(f"Live:  https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
