"""Providers for the ``web_search`` and ``web_fetch`` tools.

Standard library only. Search providers:

    duckduckgo   no key; parses DuckDuckGo's HTML results page (rate-limited, best effort)
    tavily       TAVILY_API_KEY  (free tier: https://tavily.com)
    brave        BRAVE_API_KEY   (free tier: https://brave.com/search/api)
    auto         tavily if its key is set, else brave, else duckduckgo

``fetch_url`` downloads a page and returns readable text. It only follows http(s)
URLs on public addresses: anything that resolves to localhost, a private network, or a
link-local address (cloud metadata endpoints) is refused, including after redirects,
so a prompt-injected page cannot point the agent at services on the user's machine.
"""

from __future__ import annotations

import html
import ipaddress
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable

USER_AGENT = "Mozilla/5.0 (compatible; Mk33-agent/0.1; +https://github.com/dulguuntsatsralt-cell/mk33)"
MAX_FETCH_BYTES = 3_000_000
PROVIDERS = ("auto", "duckduckgo", "tavily", "brave")


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class WebError(RuntimeError):
    pass


# --------------------------------------------------------------------- safety
def _is_public_ip(ip: str) -> bool:
    a = ipaddress.ip_address(ip)
    if a.version == 6 and a.ipv4_mapped:
        a = a.ipv4_mapped
    return a.is_global and not a.is_multicast


def check_public_url(url: str, resolve: Callable[[str], list[str]] | None = None) -> None:
    """Raise WebError unless ``url`` is http(s) and its host resolves only to public IPs."""
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https"):
        raise WebError(f"only http and https URLs are allowed, not {u.scheme or 'none'!r}")
    host = u.hostname
    if not host:
        raise WebError("URL has no host")
    resolve = resolve or _resolve
    try:
        ips = resolve(host)
    except OSError as e:
        raise WebError(f"cannot resolve {host}: {e}") from e
    bad = [ip for ip in ips if not _is_public_ip(ip)]
    if not ips or bad:
        raise WebError(f"refusing to fetch {host}: it resolves to a non-public address ({', '.join(bad) or 'none'})")


def _resolve(host: str) -> list[str]:
    return sorted({info[4][0] for info in socket.getaddrinfo(host, None)})


class _SafeRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_SafeRedirects)


def _request(url: str, data: bytes | None = None, headers: dict | None = None, timeout: float = 20.0,
             check: bool = True) -> tuple[bytes, str]:
    if check:
        check_public_url(url)
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with _opener.open(req, timeout=timeout) as resp:
            body = resp.read(MAX_FETCH_BYTES + 1)
            ctype = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        raise WebError(f"HTTP {e.code} from {urllib.parse.urlsplit(url).hostname}") from e
    except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
        raise WebError(f"could not reach {urllib.parse.urlsplit(url).hostname}: {getattr(e, 'reason', e)}") from e
    return body[:MAX_FETCH_BYTES], ctype


# --------------------------------------------------------------- html -> text
class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form", "iframe"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
             "table", "ul", "ol", "pre", "blockquote", "header"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")
        if tag in ("h1", "h2", "h3") and not self._skip:
            self.parts.append("#" * int(tag[1]) + " ")
        if tag == "li" and not self._skip:
            self.parts.append("- ")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def html_to_text(page: str) -> tuple[str, str]:
    """(title, text) with scripts, styles and navigation removed."""
    p = _TextExtractor()
    p.feed(page)
    text = "".join(p.parts).replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return p.title.strip(), text


def _decode(body: bytes, ctype: str) -> str:
    m = re.search(r"charset=([\w-]+)", ctype or "")
    for enc in ([m.group(1)] if m else []) + ["utf-8", "latin-1"]:
        try:
            return body.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", errors="replace")


def fetch_url(url: str, max_chars: int = 40_000, timeout: float = 20.0) -> str:
    """Page as readable text. Errors come back as a bracketed message for the agent."""
    try:
        body, ctype = _request(url, timeout=timeout)
    except WebError as e:
        return f"[web_fetch error: {e}]"
    if "pdf" in ctype.lower() or body[:5] == b"%PDF-":
        return "[web_fetch: this URL is a PDF; PDF text extraction is not supported]"
    text = _decode(body, ctype)
    if "html" in ctype.lower() or re.search(r"<\s*(html|body|p|div)\b", text[:2000], re.I):
        title, text = html_to_text(text)
        text = f"# {title}\n\n{text}" if title else text
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n[... truncated, {len(text) - max_chars} more chars]"
    return f"URL: {url}\n\n{text}"


# -------------------------------------------------------------------- search
_RESULT_RE = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'.*?(?:<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(?P<snip>.*?)</a>|(?=<a[^>]*class="[^"]*result__a))',
    re.S,
)


def _strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


def _unwrap_ddg(href: str) -> str:
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(href).query)
    return q.get("uddg", [href])[0]


def parse_duckduckgo(page: str, k: int) -> list[SearchResult]:
    out = []
    for m in _RESULT_RE.finditer(page):
        url = _unwrap_ddg(m.group("href"))
        if "duckduckgo.com/y.js" in url:  # ads
            continue
        out.append(SearchResult(_strip_tags(m.group("title")), url, _strip_tags(m.group("snip") or "")))
        if len(out) >= k:
            break
    return out


def search_duckduckgo(query: str, k: int = 8) -> list[SearchResult]:
    data = urllib.parse.urlencode({"q": query}).encode()
    body, ctype = _request("https://html.duckduckgo.com/html/", data=data,
                           headers={"Content-Type": "application/x-www-form-urlencoded"})
    page = _decode(body, ctype)
    results = parse_duckduckgo(page, k)
    if not results and "anomaly" in page.lower():
        raise WebError("DuckDuckGo is rate-limiting this machine; wait a minute or set TAVILY_API_KEY")
    return results


def search_tavily(query: str, k: int = 8) -> list[SearchResult]:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise WebError("TAVILY_API_KEY is not set")
    payload = json.dumps({"query": query, "max_results": k}).encode()
    body, _ = _request("https://api.tavily.com/search", data=payload,
                       headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    rows = json.loads(body).get("results", [])
    return [SearchResult(r.get("title", ""), r.get("url", ""), r.get("content", "")) for r in rows[:k]]


def search_brave(query: str, k: int = 8) -> list[SearchResult]:
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        raise WebError("BRAVE_API_KEY is not set")
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": query, "count": k})
    body, _ = _request(url, headers={"Accept": "application/json", "X-Subscription-Token": key})
    rows = (json.loads(body).get("web") or {}).get("results", [])
    return [SearchResult(r.get("title", ""), r.get("url", ""), _strip_tags(r.get("description", "")))
            for r in rows[:k]]


_SEARCHERS = {"duckduckgo": search_duckduckgo, "tavily": search_tavily, "brave": search_brave}


def resolve_provider(provider: str) -> str:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown search provider {provider!r}; choose from {', '.join(PROVIDERS)}")
    if provider != "auto":
        return provider
    if os.environ.get("TAVILY_API_KEY"):
        return "tavily"
    if os.environ.get("BRAVE_API_KEY"):
        return "brave"
    return "duckduckgo"


def format_results(query: str, results: list[SearchResult]) -> str:
    if not results:
        return f"[no results for {query!r}]"
    lines = [f"Search results for {query!r}:"]
    for i, r in enumerate(results, 1):
        lines.append(f"\n{i}. {r.title}\n   {r.url}" + (f"\n   {r.snippet}" if r.snippet else ""))
    return "\n".join(lines)


def make_search(provider: str = "auto", k: int = 8) -> Callable[[str], str]:
    """A ``search(query) -> text`` function for ``ToolExecutor``. Repeated queries are cached."""
    name = resolve_provider(provider)
    fn = _SEARCHERS[name]
    cache: dict[str, str] = {}

    def search(query: str) -> str:
        q = " ".join(query.split())
        if q in cache:
            return cache[q]
        try:
            text = format_results(q, fn(q, k))
        except (WebError, ValueError, json.JSONDecodeError) as e:
            return f"[web_search error ({name}): {e}]"
        cache[q] = text
        return text

    search.provider = name  # type: ignore[attr-defined]
    return search


def make_fetch(max_chars: int = 40_000) -> Callable[[str], str]:
    return lambda url: fetch_url(url, max_chars=max_chars)
