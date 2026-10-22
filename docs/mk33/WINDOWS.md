# Running Mk33 on Windows

Mk33 runs on a normal Windows laptop with no GPU and no API key: the model runs
locally in [Ollama](https://ollama.com), and the agent, scorer and benchmark run in
Python. This guide was written for an Intel laptop with 16 GB of RAM and integrated
graphics; anything similar works.

What runs locally, and what needs a GPU:

| Works on a laptop CPU | Needs an NVIDIA GPU (use the Colab notebooks) |
| --- | --- |
| the agent, round ledger (RLC) and retrieval memory | KV-cache compression (`kv:` modes) |
| the bench and its report | training (`notebooks/mk33_train_palm.ipynb`) |
| `mk33 ask` (web search agent) | the task-success study (`notebooks/mk33_task_success.ipynb`) |

## 1. Install

1. **Python 3.10 or newer** from python.org. Tick "Add python.exe to PATH".
2. **Git for Windows** from git-scm.com. It includes Git Bash, which the agent uses for
   its shell commands, so commands like `ls` and `python solution.py` behave as on Linux.
   Without it the agent falls back to `cmd.exe` and is told so in its prompt.
3. **Ollama** from ollama.com/download/windows.

Then, in PowerShell:

```powershell
git clone https://github.com/dulguuntsatsralt-cell/Mk33
cd Mk33
powershell -ExecutionPolicy Bypass -File scripts\mk33\windows\run_local.ps1
```

The script makes a virtual environment, installs Mk33, runs a few tests, downloads
Qwen2.5-3B (about 2 GB), and starts a short pilot run.

## 2. The model

The default is `qwen2.5:3b` (4-bit, about 2 GB of RAM). On a recent laptop CPU it
generates roughly 5 to 15 tokens per second, so one agent turn takes 30 seconds to a
couple of minutes. `qwen2.5:1.5b` is about twice as fast and noticeably weaker;
`qwen2.5:7b` fits in 16 GB but is slow on a CPU.

**Context length.** Ollama cuts every prompt to its context length (4,096 tokens by
default) without reporting an error. With an 8,000-token agent window, that would
silently turn every memory mode into truncation. The script therefore creates a copy of
the model, `mk33-qwen2.5-3b`, with a 10,240-token context. If you run models yourself,
do the same:

```powershell
"FROM qwen2.5:3b`nPARAMETER num_ctx 10240" | Set-Content mk33.Modelfile
ollama create mk33-qwen2.5-3b -f mk33.Modelfile
```

## 3. Runs

Pilot (one task, `truncate` vs `rlc`, about 1 to 2 hours):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\mk33\windows\run_local.ps1
```

Full grid (2 tasks, 3 memory modes, 3 repeats; leave it overnight):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\mk33\windows\run_local.ps1 -Full
```

Runs are resumable: stop with Ctrl+C or close the laptop, run the same command again,
and finished runs are skipped. Results and `report.md` are in `bench\local\`.

Keep the laptop plugged in and set it not to sleep (Settings > System > Power) during
long runs.

## 4. Web search agent

```powershell
.venv\Scripts\Activate.ps1
$env:MODEL_API_KEY = "ollama"
mk33 ask "Which river flows through Darkhan, Mongolia?" --model mk33-qwen2.5-3b --base-url http://localhost:11434/v1 --window 8000
```

Search uses DuckDuckGo with no key. It rate-limits heavy use; for more reliable search,
get a free key from tavily.com and set `$env:TAVILY_API_KEY` before running. The agent
cannot fetch addresses on your own network (localhost, 192.168.x.x and similar), so a
web page cannot trick it into reaching your router or local services.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `running scripts is disabled on this system` | Use the `powershell -ExecutionPolicy Bypass -File ...` form above. |
| `python3` not found inside agent runs | Expected on Windows; the agent is told to use `python`. |
| Runs end with many `[no tool call ...]` turns | The model is not producing tool calls. Use `qwen2.5` models; very small models often cannot. |
| Model calls time out | Raise `--timeout`, or use `qwen2.5:1.5b`. |
| Everything is very slow | Close other heavy programs; Ollama uses all CPU cores. |
