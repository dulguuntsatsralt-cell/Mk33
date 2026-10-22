"""web_search / web_fetch providers and the ask flow. No network: HTTP and DNS are faked."""

import json

import pytest

from conftest import ScriptedClient
from mk33 import web
from mk33.research import ask
from mk33.web import WebError, check_public_url, fetch_url, html_to_text, make_search, parse_duckduckgo


def fake_resolver(table):
    return lambda host: table[host]


@pytest.mark.parametrize("url,ips", [
    ("http://localhost:8080/", ["127.0.0.1"]),
    ("http://router.lan/", ["192.168.1.1"]),
    ("http://intranet/", ["10.0.0.5"]),
    ("http://metadata/", ["169.254.169.254"]),  # cloud metadata endpoint
    ("http://v6local/", ["::1"]),
    ("http://mapped/", ["::ffff:127.0.0.1"]),
    ("http://mixed.example/", ["93.184.216.34", "10.0.0.1"]),  # one private address is enough to refuse
])
def test_refuses_non_public_hosts(url, ips):
    host = url.split("/")[2].split(":")[0]
    with pytest.raises(WebError):
        check_public_url(url, resolve=fake_resolver({host: ips}))


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x", "gopher://x", "/relative"])
def test_refuses_other_schemes(url):
    with pytest.raises(WebError):
        check_public_url(url, resolve=fake_resolver({}))


def test_allows_public_host():
    check_public_url("https://example.com/a?b=1", resolve=fake_resolver({"example.com": ["93.184.216.34"]}))


def test_html_to_text_drops_scripts_and_keeps_structure():
    page = """<html><head><title>Darkhan</title><style>p{color:red}</style></head><body>
      <nav>Home | About</nav><h1>Darkhan city</h1><script>evil()</script>
      <p>Darkhan is the third largest city&nbsp;in Mongolia.</p><ul><li>Kharaa river</li></ul>
      <footer>(c) site</footer></body></html>"""
    title, text = html_to_text(page)
    assert title == "Darkhan"
    assert "# Darkhan city" in text
    assert "third largest city in Mongolia" in text
    assert "- Kharaa river" in text
    for junk in ("evil()", "color:red", "Home | About", "(c) site"):
        assert junk not in text


DDG_PAGE = """
<div class="result"><h2><a rel="nofollow" class="result__a"
  href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FDarkhan&amp;rut=abc">Darkhan - <b>Wikipedia</b></a></h2>
  <a class="result__snippet" href="x">Darkhan is the <b>third</b> largest city.</a></div>
<div class="result"><h2><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">Ad</a></h2>
  <a class="result__snippet" href="x">buy now</a></div>
<div class="result"><h2><a class="result__a" href="https://example.org/mongolia">Mongolia &amp; cities</a></h2>
  <a class="result__snippet" href="x">A list.</a></div>
"""


def test_parse_duckduckgo_unwraps_links_and_skips_ads():
    rs = parse_duckduckgo(DDG_PAGE, k=5)
    assert [r.url for r in rs] == ["https://en.wikipedia.org/wiki/Darkhan", "https://example.org/mongolia"]
    assert rs[0].title == "Darkhan - Wikipedia"
    assert rs[0].snippet == "Darkhan is the third largest city."
    assert rs[1].title == "Mongolia & cities"


def test_provider_auto_selection(monkeypatch):
    for k in ("TAVILY_API_KEY", "BRAVE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert web.resolve_provider("auto") == "duckduckgo"
    monkeypatch.setenv("BRAVE_API_KEY", "b")
    assert web.resolve_provider("auto") == "brave"
    monkeypatch.setenv("TAVILY_API_KEY", "t")
    assert web.resolve_provider("auto") == "tavily"
    with pytest.raises(ValueError):
        web.resolve_provider("bing")


def test_tavily_request_and_parse(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "secret")
    seen = {}

    def fake_request(url, data=None, headers=None, timeout=20.0, check=True):
        seen.update(url=url, data=json.loads(data), headers=headers)
        return json.dumps({"results": [{"title": "T", "url": "https://a.example", "content": "C"}]}).encode(), "application/json"

    monkeypatch.setattr(web, "_request", fake_request)
    rs = web.search_tavily("darkhan population", k=3)
    assert seen["url"] == "https://api.tavily.com/search"
    assert seen["data"] == {"query": "darkhan population", "max_results": 3}
    assert seen["headers"]["Authorization"] == "Bearer secret"
    assert rs[0].url == "https://a.example" and rs[0].snippet == "C"


def test_make_search_caches_and_reports_errors(monkeypatch):
    calls = []

    def fake_ddg(q, k):
        calls.append(q)
        if q == "boom":
            raise WebError("rate limited")
        return [web.SearchResult("Title", "https://x.example", "snip")]

    monkeypatch.setitem(web._SEARCHERS, "duckduckgo", fake_ddg)
    search = make_search("duckduckgo")
    out = search("mongolia  coal")
    assert "1. Title" in out and "https://x.example" in out
    search("mongolia coal")  # same query after whitespace normalization: cached
    assert calls == ["mongolia coal"]
    assert search("boom").startswith("[web_search error (duckduckgo): rate limited")


def test_fetch_url_converts_html_and_reports_errors(monkeypatch):
    def fake_request(url, data=None, headers=None, timeout=20.0, check=True):
        if "bad" in url:
            raise WebError("HTTP 404 from bad.example")
        return b"<html><title>Hi</title><body><p>Hello world</p></body></html>", "text/html; charset=utf-8"

    monkeypatch.setattr(web, "_request", fake_request)
    out = fetch_url("https://ok.example/page")
    assert out.startswith("URL: https://ok.example/page") and "# Hi" in out and "Hello world" in out
    assert fetch_url("https://bad.example/") == "[web_fetch error: HTTP 404 from bad.example]"
    assert "truncated" in fetch_url("https://ok.example/page", max_chars=5)


def test_fetch_url_blocks_localhost_for_real():
    # Real resolution of "localhost", no network needed: must be refused before any request.
    assert "non-public address" in fetch_url("http://localhost:1/secret")


def test_ask_end_to_end_with_scripted_model():
    client = ScriptedClient([
        ("find sources", "web_search", {"query": "largest river near Darkhan"}),
        ("read it", "web_fetch", {"url": "https://en.wikipedia.org/wiki/Darkhan"}),
        ("done", "answer", {"answer": "Kharaa River", "confidence": 0.8,
                            "evidence": ["https://en.wikipedia.org/wiki/Darkhan"]}),
    ])
    searched, fetched, logs = [], [], []
    r = ask("Which river flows through Darkhan?", client,
            search=lambda q: searched.append(q) or "1. Darkhan - Wikipedia\n   https://en.wikipedia.org/wiki/Darkhan",
            fetch=lambda u: fetched.append(u) or "Darkhan lies on the Kharaa River.",
            log=logs.append)
    assert r.answer == "Kharaa River" and r.confidence == 0.8
    assert r.evidence == ["https://en.wikipedia.org/wiki/Darkhan"]
    assert (r.searches, r.fetches, r.stop) == (1, 1, "answer")
    assert searched == ["largest river near Darkhan"] and fetched == ["https://en.wikipedia.org/wiki/Darkhan"]
    assert len(logs) == 3
    # The fetched page reached the model as a tool result.
    assert any("Kharaa River" in str(m.get("content")) for m in client.contexts[-1])
