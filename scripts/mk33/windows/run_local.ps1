# Mk33 on a Windows laptop: Ollama + Qwen2.5-3B, no GPU and no API key needed.
#
# Run from the repository folder in PowerShell:
#     powershell -ExecutionPolicy Bypass -File scripts\mk33\windows\run_local.ps1            # set up + pilot run
#     powershell -ExecutionPolicy Bypass -File scripts\mk33\windows\run_local.ps1 -Full      # full grid (overnight)
#
# Needs: Python 3.10+, Git for Windows (the agent's commands run in Git Bash), Ollama.
# Guide: docs/mk33/WINDOWS.md

param(
    [switch]$Full,                       # run the full grid instead of the pilot
    [string]$Model = "qwen2.5:3b",       # any Ollama model with tool calling
    [int]$Context = 10240,               # Ollama context length; must exceed -Window
    [int]$Window = 8000,                 # agent's text window (tokens)
    [string]$Out = "bench\local"
)

$ErrorActionPreference = "Stop"
# Native commands do not stop the script on failure by themselves; check their exit code.
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" } }
function Need($cmd, $hint) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { throw "$cmd not found. $hint" }
}

Write-Host "== 1. Checking tools"
Need python "Install Python 3.10+ from https://www.python.org/downloads/ (tick 'Add python.exe to PATH')."
Need git    "Install Git for Windows from https://git-scm.com/download/win (it includes Git Bash)."
Need ollama "Install Ollama from https://ollama.com/download/windows, then open a new PowerShell."
python -c "import sys; assert sys.version_info >= (3, 10), sys.version"; Check "Python 3.10+ check"

Write-Host "== 2. Python environment (.venv)"
if (-not (Test-Path .venv)) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1
python -m pip install -q --upgrade pip
python -m pip install -q -e ".[mk33,dev]"; Check "pip install"

Write-Host "== 3. Quick tests"
python -m pytest tests/mk33/test_sandbox.py tests/mk33/test_end_to_end.py -q; Check "tests"

Write-Host "== 4. Model"
# Ollama cuts every prompt to its context length (4096 by default) without any error,
# which would silently break the memory comparison. Make a copy with a longer context.
$local = "mk33-" + ($Model -replace "[:/]", "-")
ollama pull $Model; Check "ollama pull"
$modelfile = Join-Path $env:TEMP "mk33.Modelfile"
"FROM $Model`nPARAMETER num_ctx $Context`nPARAMETER temperature 0.7" | Set-Content -Encoding ascii $modelfile
ollama create $local -f $modelfile; Check "ollama create"
Write-Host "using $local (context $Context tokens)"

$env:MODEL_API_KEY = "ollama"       # Ollama ignores the key, the client needs a value
$common = @("--model", $local, "--base-url", "http://localhost:11434/v1", "--workers", "1",
            "--window", "$Window", "--timeout", "900", "--max-tokens", "1024", "--out", $Out)

Write-Host "== 5. Environments"
mk33 bench-prepare --out bench/envs; Check "bench-prepare"

if (-not $Full) {
    Write-Host "== 6. Pilot: one task, two memory modes, short budget (about 1 to 2 hours on a laptop CPU)"
    mk33 bench-run @common --tasks knapsack --modes truncate,rlc --repeats 1 --max-rounds 5 --max-tool-calls 30 --max-minutes 60
} else {
    Write-Host "== 6. Full grid: 2 tasks x 3 modes x 3 repeats (overnight; safe to stop and rerun, finished runs are kept)"
    mk33 bench-run @common --tasks knapsack,regression --modes truncate,rlc,retrieve --repeats 3 --max-rounds 8 --max-tool-calls 40 --max-minutes 90
}

Write-Host "== Done. Report:"
Get-Content (Join-Path $Out "report.md")
