"""KV-cache compression for long-horizon agents on local models.

Needs ``pip install -e '.[mk33-kv]'`` (PyTorch, transformers; Triton on Linux).

    KVPolicy             which tokens to keep (full, random, streaming, keynorm, h2o)
    CompressedGenerator  chunked prefill and decoding under a fixed cache budget
    HFChatClient         drop-in ChatClient for the agent and the bench

See docs/mk33/MEMORY.md for the study design.
"""

from .names import POLICIES

__all__ = ["POLICIES", "KVPolicy", "select_indices", "CompressedGenerator", "KVStats", "HFChatClient",
           "parse_tool_calls", "summarize_stats"]


def __getattr__(name):  # keep `import mk33.kv` free of torch until something needs it
    if name in ("KVPolicy", "select_indices"):
        from . import policies

        return getattr(policies, name)
    if name in ("CompressedGenerator", "KVStats"):
        from . import generate

        return getattr(generate, name)
    if name in ("HFChatClient", "parse_tool_calls", "summarize_stats"):
        from . import client

        return getattr(client, name)
    raise AttributeError(name)
