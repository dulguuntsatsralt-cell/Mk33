"""Per-call logging and the native Ollama client (no server needed: mocked HTTP)."""

import json

import httpx

from mk33.llm import OllamaChatClient, prompt_chars, summarize_calls, to_ollama_messages
from mk33.scaffold.tools import TOOL_SCHEMAS


def _client(reply: dict, seen: list) -> OllamaChatClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=reply)

    c = OllamaChatClient("qwen", "http://localhost:11434/v1")
    c.http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


REPLY = {
    "message": {"role": "assistant", "content": "",
                "tool_calls": [{"function": {"name": "submit", "arguments": {"hypothesis": "h", "lesson": "l"}}}]},
    "prompt_eval_count": 4000, "prompt_eval_duration": 2_000_000_000,
    "eval_count": 50, "eval_duration": 5_000_000_000, "load_duration": 1_000_000,
}


def test_native_url_strips_v1():
    assert OllamaChatClient("m", "http://localhost:11434/v1").url == "http://localhost:11434/api/chat"
    assert OllamaChatClient("m").url == "http://localhost:11434/api/chat"


def test_ollama_client_parses_calls_and_logs_timings():
    seen = []
    c = _client(REPLY, seen)
    out = c.chat([{"role": "user", "content": "go"}], tools=[TOOL_SCHEMAS["submit"]])
    assert out["tool_calls"][0]["name"] == "submit"
    assert out["tool_calls"][0]["arguments"] == {"hypothesis": "h", "lesson": "l"}
    log = c.call_log[0]
    assert log["prompt_tokens"] == 4000 and log["completion_tokens"] == 50
    assert log["prompt_s"] == 2.0 and log["completion_s"] == 5.0
    assert log["prompt_chars"] == 2
    body = seen[0]
    assert body["stream"] is False and body["tools"][0]["function"]["name"] == "submit"


def test_ollama_client_recovers_text_calls():
    reply = {"message": {"role": "assistant",
                         "content": '```json\n{"name": "submit", "arguments": {"hypothesis": "h", "lesson": "l"}}\n```'}}
    c = _client(reply, [])
    out = c.chat([{"role": "user", "content": "go"}], tools=[TOOL_SCHEMAS["submit"]])
    assert out["tool_calls"][0]["name"] == "submit" and c.text_tool_calls == 1


def test_messages_keep_text_and_decode_arguments():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "a.py"},
    ]
    out = to_ollama_messages(msgs)
    assert [m["content"] for m in out] == ["s", "", "a.py"]
    assert out[1]["tool_calls"] == [{"function": {"name": "bash", "arguments": {"cmd": "ls"}}}]
    assert prompt_chars(msgs) == len("s") + len("bash") + len('{"cmd": "ls"}') + len("a.py")


def test_summary_reports_prompt_reading_speed():
    log = [{"s": 10.0, "prompt_tokens": 4000, "completion_tokens": 40, "prompt_s": 0.5, "completion_s": 9.0},
           {"s": 30.0, "prompt_tokens": 4000, "completion_tokens": 40, "prompt_s": 20.0, "completion_s": 9.0},
           {"s": 12.0, "prompt_tokens": 4200, "completion_tokens": 60, "prompt_s": 1.0, "completion_s": 10.0}]
    s = summarize_calls(log)
    assert s["call_s_median"] == 12.0 and s["call_prompt_s_median"] == 1.0
    assert s["prompt_tok_per_s_median"] == 4200.0  # 8000, 200, 4200
    # A backend without timings (OpenAI-compatible) gives only what it has.
    assert set(summarize_calls([{"s": 1.0, "prompt_tokens": 5, "completion_tokens": 2}])) == {
        "call_s_median", "call_prompt_tokens_median", "call_completion_tokens_median"}
