# Mk33 memory in LangChain and LangGraph agents

`mk33.integrations.langgraph` brings Mk33's round-ledger memory to agents built with
LangChain's `create_agent`, which runs on LangGraph.

```python
from langchain.agents import create_agent
from mk33.integrations.langgraph import LedgerMiddleware, mk33_web_tools

agent = create_agent(
    model,                                   # any LangChain chat model
    tools=mk33_web_tools(),                  # Mk33's web_search / web_fetch (optional)
    middleware=[LedgerMiddleware(window=8000, retrieve_k=2)],
)
agent.invoke({"messages": [("user", "Which river flows through Darkhan?")]})
```

Install with `pip install -e '.[langgraph]'`. A complete command-line example is in
`scripts/mk33/langgraph_agent.py`.

## What it does

Before each model call, if the conversation is longer than `theta * window` tokens
(default 75%), the middleware rewrites what the model sees:

1. The system prompt and the task stay at the top.
2. Older steps (an AI message plus the tool results it asked for) are replaced by a
   **ledger**: one line per step with the tool, its arguments and the first line of the
   result. Failed tool calls are marked `FAILED` and stay on the ledger.
3. With `retrieve_k > 0`, BM25 retrieval adds the full text of the old steps most
   relevant to the latest message.
4. The newest `keep_steps` steps follow verbatim. A tool result is never separated from
   the call that produced it.

The graph state is not changed, so the full history is kept (and saved by any
checkpointer). Only the model's input is compacted.

## Compared with SummarizationMiddleware

LangChain's `SummarizationMiddleware` asks the model to summarize old messages and
replaces them in the state. The ledger takes a different approach:

| | SummarizationMiddleware | LedgerMiddleware |
| --- | --- | --- |
| Extra model calls | one per summary | none |
| Deterministic | no | yes |
| History in state | replaced by the summary | kept in full |
| Failed attempts | kept only if the summary mentions them | always on the ledger |

Which works better for a given agent is an empirical question; the Mk33 bench is one way
to measure it.

## Options

| Argument | Default | Meaning |
| --- | --- | --- |
| `window` | 8000 | the model's usable context, in tokens |
| `theta` | 0.75 | compact once the history passes this share of the window |
| `keep_steps` | 6 | newest steps kept verbatim (fewer if they alone do not fit) |
| `ledger_cap` | 2000 | tokens for the ledger; the oldest lines go first when it is full |
| `retrieve_k` | 0 | old steps recalled in full by BM25 |
| `retrieve_budget` | 1500 | tokens for recalled steps |
| `count_tokens` | about 4 characters per token | pass a tokenizer's counter for exact counts |

After a run, `middleware.events` lists each compaction: total tokens, tokens sent,
and how many steps were compacted or recalled.
