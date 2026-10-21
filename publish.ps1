# One-shot publish for Windows PowerShell: installs Git/GitHub CLI if missing, then GitHub repo + Hugging Face Space.
#   powershell -ExecutionPolicy Bypass -File .\publish.ps1
#   powershell -ExecutionPolicy Bypass -File .\publish.ps1 -Name my-other-name
param([string]$Name = "agentic-rag-docqa")
$ErrorActionPreference = "Continue"
function Check($what) { if ($LASTEXITCODE -ne 0) { Write-Host "$what failed (exit $LASTEXITCODE)." -ForegroundColor Red; exit 1 } }
Set-Location $PSScriptRoot

function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
}
function Need($cmd, $wingetId) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) { return }
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Installing $wingetId ..." -ForegroundColor Cyan
        winget install --id $wingetId -e --accept-source-agreements --accept-package-agreements
        Refresh-Path
    }
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Host "Missing '$cmd'. Install it with:  winget install --id $wingetId -e" -ForegroundColor Yellow
        Write-Host "Then CLOSE and REOPEN PowerShell and run this script again."
        exit 1
    }
}
Need git "Git.Git"
Need gh  "GitHub.cli"
$py = $null
foreach ($c in "py", "python") {   # skip the Microsoft Store "python" stub, which exists but doesn't run
    if (Get-Command $c -ErrorAction SilentlyContinue) { & $c -c "import sys" *> $null; if ($LASTEXITCODE -eq 0) { $py = $c; break } }
}
if (-not $py) { Write-Host "Install Python with:  winget install --id Python.Python.3.12 -e  (then reopen PowerShell)" -ForegroundColor Yellow; exit 1 }

# ---- 1) GitHub ----
gh auth status *> $null
if ($LASTEXITCODE -ne 0) { gh auth login --web --git-protocol https; Check "GitHub login" }
gh auth setup-git

if (-not (Test-Path .git)) {
    git init -q -b main
    git add -A
    git commit -qm "Initial commit: agentic RAG document Q&A"
    Check "git commit"
}
git remote get-url origin *> $null
if ($LASTEXITCODE -ne 0) {
    gh repo create $Name --public --source . --remote origin `
        --description "Agentic RAG document Q&A: LangGraph, FAISS/ChromaDB, Claude/Mistral, page-level citations, labeled eval in CI"
    Check "Creating the GitHub repo"
}
git push -u origin main
Check "git push"
$ghUser = gh api user -q .login
gh repo edit "$ghUser/$Name" --add-topic "rag,langgraph,langchain,faiss,chromadb,llm,gradio,huggingface,agentic-ai" *> $null

# ---- 2) Hugging Face Space ----
& $py -m pip install -q "huggingface_hub>=1.0"
& $py scripts/deploy_space.py $Name
Check "Hugging Face deploy"

Write-Host "Code:  https://github.com/$ghUser/$Name" -ForegroundColor Green
Write-Host "Tip: add the same HF token as GitHub secret HF_TOKEN so every push to main redeploys the Space:"
Write-Host "     gh secret set HF_TOKEN --repo $ghUser/$Name"
