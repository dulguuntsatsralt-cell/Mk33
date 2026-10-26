"""Tool calls written as text (seen in a 40-round Qwen2.5-7B run after the first compaction)."""

from types import SimpleNamespace

from mk33.llm import OpenAIChatClient
from mk33.scaffold.tools import TOOL_SCHEMAS
from mk33.toolcalls import calls_from_text, tool_names

ALLOWED = {"submit", "edit_file", "read_file"}

# Close to what the model wrote on turn 16 of the run.
FENCED = '''I will proceed with submitting the updated solution to see if the issue is resolved.

```json
{"name": "submit", "arguments": {"hypothesis": "Imported numpy as np and ensured proper usage.", "lesson": "Ensuring imports are correctly set up should resolve NameError."}}
```'''

EDIT_WITH_CODE = '''First, I will ensure the file is correctly edited:

```json
{"name": "edit_file", "arguments": {"path": "solution.py", "old_str": "", "new_str": "def f(x):\\n    return {'a': x}\\n"}}
```
Then I will submit.'''


def test_fenced_call_is_recovered():
    [c] = calls_from_text(FENCED, ALLOWED)
    assert c["name"] == "submit"
    assert c["arguments"]["hypothesis"] == "Imported numpy as np and ensured proper usage."


def test_braces_inside_code_strings_do_not_break_parsing():
    [c] = calls_from_text(EDIT_WITH_CODE, ALLOWED)
    assert c["name"] == "edit_file" and c["arguments"]["new_str"] == "def f(x):\n    return {'a': x}\n"


def test_only_offered_tools_and_call_shaped_objects_are_accepted():
    assert calls_from_text('{"name": "rm_rf", "arguments": {}}', ALLOWED) == []
    assert calls_from_text('Config: {"name": "submit", "lr": 0.1}', ALLOWED) == []  # no arguments field
    assert calls_from_text('{"score": 1.2}', ALLOWED) == []
    assert calls_from_text("no json here {not json", ALLOWED) == []
    assert calls_from_text(FENCED, set()) == []


def test_parameters_key_string_arguments_and_limit():
    [c] = calls_from_text('{"name": "read_file", "parameters": "{\\"path\\": \\"a.py\\"}"}', ALLOWED)
    assert c["arguments"] == {"path": "a.py"}
    two = '{"name": "read_file", "arguments": {"path": "a"}} then {"name": "submit", "arguments": {}}'
    assert [c["name"] for c in calls_from_text(two, ALLOWED, limit=2)] == ["read_file", "submit"]
    assert len(calls_from_text(two, ALLOWED)) == 1


def _openai_client_replying(content):
    msg = SimpleNamespace(content=content, tool_calls=None)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    c = object.__new__(OpenAIChatClient)
    c.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: resp)))
    c.model, c.temperature, c.max_tokens, c.text_tool_calls, c.call_log = "m", 0.7, 100, 0, []
    return c


def test_openai_client_falls_back_to_text_calls_and_counts_them():
    tools = [TOOL_SCHEMAS["submit"], TOOL_SCHEMAS["edit_file"]]
    assert tool_names(tools) == {"submit", "edit_file"}
    c = _openai_client_replying(FENCED)
    out = c.chat([{"role": "user", "content": "go"}], tools=tools)
    assert out["tool_calls"][0]["name"] == "submit" and c.text_tool_calls == 1
    # Without tools on the request nothing is recovered.
    c2 = _openai_client_replying(FENCED)
    assert c2.chat([{"role": "user", "content": "go"}])["tool_calls"] == [] and c2.text_tool_calls == 0
