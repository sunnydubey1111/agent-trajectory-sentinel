"""WS1 — real tools (first increment: no API key required).

Built on the WS0.5 contract (derail.harness.tools.Tool): each exposes a
name/description/parameters and a run(**args) -> str, so it drops straight
into a ToolRegistry and its calls render as the "[name(args) -> result]"
step-text the adapter's x channel already parses.

This increment ships the three that need no credentials:

  PythonREPL       execute a Python snippet (replaces the mock calculator;
                   enables the code-generation / execute-Python tasks)
  WikipediaSearch  real Wikipedia search API (RAG / knowledge tasks)
  ArxivSearch      real arXiv Atom API (the "latest papers" task)

Key-gated tools (Tavily, GitHub, SQL, Qdrant, browser, MCP) are later WS1
increments. HTTP tools take an injectable `get` fetcher so their parsing is
unit-tested offline; the default fetcher is stdlib urllib (no new deps).

CONTAINMENT
-------------------------------------------------------
Every tool here is driven by untrusted model output, so:

  * `default_registry()` never includes the host-code / navigation / process
    spawning tools (`python`, `browser_browse`, `mcp_call`).  A caller must
    name them in an allowlist - normally `build_registry(task.tools)`, which
    gives each task exactly the tools its prompt needs and nothing else.
  * PythonREPL runs the snippet in a subprocess under isolated mode (`-I`)
    with a wall-clock timeout, a throwaway working directory, an environment
    scrubbed of every credential-shaped variable, and an in-process network
    guard.  This is process-level containment, NOT an OS sandbox: see
    derail.harness.sandbox.REQUIRES_CONTAINER before pointing it at an
    adversarial model.
  * BrowserAutomation refuses non-HTTP(S) schemes and any host resolving to a
    non-global address, and accepts a per-call host allowlist.
  * MCPClientTool only starts servers that the *operator* configured by name;
    a model-supplied command line is refused.
  * The filesystem tools refuse credential-shaped paths, and every tool result
    is passed through sandbox.redact_secrets before it can reach a model, a
    trace or a cassette.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

from derail.harness.sandbox import (DANGEROUS_TOOLS, UrlRefused, check_url,
                                    guarded_code, is_sensitive_path,
                                    redact_secrets, scrubbed_env)

_UA = os.environ.get("AGENTWATCH_USER_AGENT",
                     "derail-harness/1.0 (research; contact via repo)")
# Output caps. Every one of these changes what the agent SEES, so a corpus is
# only comparable with another collected under the same values; they are named
# and overridable here rather than buried at the call sites so a deployment can
# state its own and a reader can find them all in one place.
_MAX_RESULT_CHARS = int(os.environ.get("AGENTWATCH_MAX_RESULT_CHARS", "600"))
_MAX_FILE_CHARS = int(os.environ.get("AGENTWATCH_MAX_FILE_CHARS", "4000"))
# Hard ceiling on bytes read from any remote body / subprocess pipe. Applied at
# the source, before parsing, so a hostile or runaway response cannot be
# materialised in memory first.
_MAX_FETCH_BYTES = int(os.environ.get("AGENTWATCH_MAX_FETCH_BYTES",
                                      str(2_000_000)))


_TRUSTSTORE_READY = False


def _ensure_tls() -> None:
    """Use the OS trust store once (this machine's AV intercepts TLS, so the
    bundled certifi roots fail — same fix the collectors use)."""
    global _TRUSTSTORE_READY
    if _TRUSTSTORE_READY:
        return
    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:  # noqa: BLE001 — absent/unneeded elsewhere; try anyway
        pass
    _TRUSTSTORE_READY = True


def _http_get(url: str, timeout: float = 15.0,
              max_bytes: int = _MAX_FETCH_BYTES) -> bytes:
    """GET a vetted URL, reading at most `max_bytes`."""
    _ensure_tls()
    req = urllib.request.Request(check_url(url), headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return resp.read(max_bytes)


def _http_post(url: str, json_data: dict, timeout: float = 15.0,
               max_bytes: int = _MAX_FETCH_BYTES) -> bytes:
    _ensure_tls()
    req = urllib.request.Request(
        check_url(url),
        data=json.dumps(json_data).encode("utf-8"),
        headers={"User-Agent": _UA, "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return resp.read(max_bytes)



# ------------------------------------------------------------- Python REPL
class PythonREPL:
    """Run a Python 3 snippet in an isolated subprocess; return stdout."""

    name = "python"
    description = ("Execute a Python 3 snippet and return whatever it prints "
                   "to stdout. Use print() to return values. A fresh "
                   "interpreter each call (no state persists); no network "
                   "access; times out.")
    parameters = {"code": "Python source to run. print() what you want back."}

    def __init__(self, timeout_s: float = 10.0, allow_network: bool = False,
                 max_output_bytes: int = 1_000_000) -> None:
        self.timeout_s = float(timeout_s)
        self.allow_network = bool(allow_network)
        self.max_output_bytes = int(max_output_bytes)

    def run(self, code: str) -> str:
        if not isinstance(code, str) or not code.strip():
            return "Error: empty code"
        # Isolated interpreter, throwaway cwd, credential-free environment and
        # a network guard around the model's own source.
        source = guarded_code(code, allow_network=self.allow_network)
        with tempfile.TemporaryDirectory() as cwd:
            # Spool to temp FILES rather than pipes, and cap by seeking rather
            # than by slicing what came back. `capture_output=True` buffers the
            # whole stream in this process first and only then truncates, so a
            # snippet printing in a loop is bounded by the timeout, not by
            # `max_output_bytes` — it can exhaust memory here well before the
            # timeout fires. A file spool bounds the resident cost to what is
            # read back.
            with tempfile.TemporaryFile() as out_f, \
                    tempfile.TemporaryFile() as err_f:
                try:
                    proc = subprocess.run(
                        [sys.executable, "-I", "-c", source],
                        stdout=out_f, stderr=err_f, timeout=self.timeout_s,
                        cwd=cwd, env=scrubbed_env(), stdin=subprocess.DEVNULL,
                    )
                except subprocess.TimeoutExpired:
                    return f"Error: timed out after {self.timeout_s:g}s"
                out_f.seek(0)
                err_f.seek(0)
                stdout = out_f.read(self.max_output_bytes).decode(
                    "utf-8", "replace")
                stderr = err_f.read(self.max_output_bytes).decode(
                    "utf-8", "replace")
        if proc.returncode != 0:
            err = stderr.strip().splitlines()
            return redact_secrets("Error: " + (err[-1] if err else "non-zero exit"))
        out = stdout.strip()
        return (redact_secrets(out)[:_MAX_RESULT_CHARS] if out
                else "(no output; did you print?)")


# ------------------------------------------------------------- Wikipedia
class WikipediaSearch:
    """Search Wikipedia; return the top titles with snippets."""

    name = "wikipedia_search"
    description = ("Search English Wikipedia and return the top matching "
                   "article titles with a short snippet from each.")
    parameters = {"query": "What to search for on Wikipedia."}

    def __init__(self, get: Callable[[str], bytes] = _http_get,
                 max_results: int = 3) -> None:
        self.get = get
        self.max_results = int(max_results)

    def run(self, query: str) -> str:
        if not str(query).strip():
            return "Error: empty query"
        url = ("https://en.wikipedia.org/w/api.php?action=query&list=search"
               "&format=json&srlimit=" + str(self.max_results)
               + "&srsearch=" + urllib.parse.quote(str(query)))
        data = json.loads(self.get(url))
        hits = data.get("query", {}).get("search", [])
        if not hits:
            return f"No Wikipedia results for {query!r}."
        lines = []
        for h in hits[:self.max_results]:
            snippet = (h.get("snippet", "")
                       .replace('<span class="searchmatch">', "")
                       .replace("</span>", "").replace("&quot;", '"'))
            lines.append(f"{h.get('title', '?')}: {snippet}")
        return " | ".join(lines)[:_MAX_RESULT_CHARS]


# ------------------------------------------------------------- arXiv
class ArxivSearch:
    """Search arXiv; return the top paper titles and authors."""

    name = "arxiv_search"
    description = ("Search arXiv for papers and return the top matches "
                   "(title, first author, arXiv id), most relevant first.")
    parameters = {"query": "Topic or keywords to search arXiv for."}

    def __init__(self, get: Callable[[str], bytes] = _http_get,
                 max_results: int = 3) -> None:
        self.get = get
        self.max_results = int(max_results)

    def run(self, query: str) -> str:
        if not str(query).strip():
            return "Error: empty query"
        # HTTPS: plaintext transport let a network attacker rewrite retrieved
        # content, which is then fed straight to the model.
        url = ("https://export.arxiv.org/api/query?sortBy=relevance"
               "&sortOrder=descending&max_results=" + str(self.max_results)
               + "&search_query=all:" + urllib.parse.quote(str(query)))
        root = ET.fromstring(self.get(url))
        ns = {"a": "http://www.w3.org/2005/Atom"}
        out = []
        for entry in root.findall("a:entry", ns)[:self.max_results]:
            title = " ".join((entry.findtext("a:title", "", ns)).split())
            aid = (entry.findtext("a:id", "", ns) or "").rsplit("/", 1)[-1]
            author = entry.findtext("a:author/a:name", "", ns)
            out.append(f"{title} ({author}, {aid})")
        return (" | ".join(out) or f"No arXiv results for {query!r}.")[
            :_MAX_RESULT_CHARS]


# ------------------------------------------------------------- DuckDuckGo
class DuckDuckGoSearch:
    """DuckDuckGo Instant-Answer API (keyless). Returns the abstract plus a
    few related topics — lighter than a full web index, but no credentials."""

    name = "web_search"
    description = ("Search the web (DuckDuckGo) and return a short abstract "
                   "and a few related result snippets.")
    parameters = {"query": "What to search the web for."}

    def __init__(self, get: Callable[[str], bytes] = _http_get,
                 max_results: int = 4) -> None:
        self.get = get
        self.max_results = int(max_results)

    def run(self, query: str) -> str:
        if not str(query).strip():
            return "Error: empty query"
        url = ("https://api.duckduckgo.com/?no_html=1&skip_disambig=1"
               "&format=json&q=" + urllib.parse.quote(str(query)))
        data = json.loads(self.get(url))
        parts = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"])
        for topic in data.get("RelatedTopics", []):
            if len(parts) >= self.max_results:      # was '>' - returned one extra
                break
            text = topic.get("Text") if isinstance(topic, dict) else None
            if text:
                parts.append(text)
        if not parts:
            return f"No web results for {query!r}."
        return " | ".join(parts)[:_MAX_RESULT_CHARS]


# ------------------------------------------------------------- filesystem
class _Sandboxed:
    """Base for filesystem tools confined to a workspace root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _resolve(self, path: str) -> Path | None:
        p = (self.root / str(path)).resolve()
        return p if p == self.root or p.is_relative_to(self.root) else None


class ReadFile(_Sandboxed):
    name = "read_file"
    description = ("Read a UTF-8 text file inside the workspace and return "
                   "its contents (truncated). Credential files and version "
                   "control internals are not readable.")
    parameters = {"path": "File path relative to the workspace root."}

    def run(self, path: str) -> str:
        p = self._resolve(path)
        if p is None:
            return "Error: path escapes the workspace"
        # Refuse before reading: a .env or .git/config inside the workspace
        # would otherwise be shipped to the model and stored in a plaintext
        # cassette.
        if is_sensitive_path(path) or is_sensitive_path(p):
            return "Error: refused - credential or version-control path"
        if not p.is_file():
            return f"Error: no such file {path!r}"
        text = p.read_text("utf-8", errors="replace")[:_MAX_FILE_CHARS]
        return redact_secrets(text)


class ListDir(_Sandboxed):
    name = "list_dir"
    description = "List the entries of a directory inside the workspace."
    parameters = {"path": "Directory path relative to the workspace root "
                          "(use '.' for the root)."}

    def run(self, path: str = ".") -> str:
        p = self._resolve(path)
        if p is None:
            return "Error: path escapes the workspace"
        if is_sensitive_path(path) or is_sensitive_path(p):
            return "Error: refused - credential or version-control path"
        if not p.is_dir():
            return f"Error: not a directory {path!r}"
        # Credential material is not advertised either; read_file would refuse
        # it anyway, and naming it only invites the model to try.
        entries = sorted(x.name + ("/" if x.is_dir() else "")
                         for x in p.iterdir() if not is_sensitive_path(x.name))
        return (", ".join(entries) or "(empty)")[:_MAX_RESULT_CHARS]


# ------------------------------------------------------------- Tavily Search
class TavilySearch:
    """Search the web using Tavily Search API (requires TAVILY_API_KEY)."""

    name = "tavily_search"
    description = ("Search the web using Tavily Search API and return titles, "
                   "URLs, and content snippets.")
    parameters = {"query": "The search query."}

    def __init__(self, max_results: int = 3) -> None:
        self.max_results = int(max_results)

    def run(self, query: str) -> str:
        from derail.config import get_api_key
        api_key = get_api_key("TAVILY_API_KEY")
        if not api_key:
            return "Error: TAVILY_API_KEY is not configured in derail.config / environment."
        if not str(query).strip():
            return "Error: empty query"
        payload = {
            "api_key": api_key,
            "query": str(query),
            "max_results": self.max_results
        }
        try:
            url = "https://api.tavily.com/search"
            data = json.loads(_http_post(url, payload).decode("utf-8"))
            results = data.get("results", [])
            if not results:
                return f"No Tavily results for {query!r}."
            lines = []
            for r in results[:self.max_results]:
                lines.append(f"{r.get('title', '?')} ({r.get('url', '')}): {r.get('content', '')}")
            return " | ".join(lines)[:_MAX_RESULT_CHARS]
        except Exception as exc:
            return f"Error: Tavily search failed: {exc}"


# ------------------------------------------------------------- Open-Meteo Weather
class OpenMeteoWeather:
    """Get the current weather for a city using the Open-Meteo API."""

    name = "get_weather"
    description = "Get the current weather (temperature, condition) for a city."
    parameters = {"city": "The city name to look up weather for."}

    def run(self, city: str) -> str:
        if not str(city).strip():
            return "Error: empty city name"
        try:
            # Geocode city
            geo_url = ("https://geocoding-api.open-meteo.com/v1/search?name="
                       + urllib.parse.quote(str(city)) + "&count=1&format=json")
            geo_data = json.loads(_http_get(geo_url).decode("utf-8"))
            results = geo_data.get("results", [])
            if not results:
                return f"Error: City {city!r} not found."
            lat = results[0]["latitude"]
            lon = results[0]["longitude"]
            name = results[0]["name"]
            country = results[0].get("country", "")

            # Fetch weather
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            w_data = json.loads(_http_get(weather_url).decode("utf-8"))
            curr = w_data.get("current_weather", {})
            temp = curr.get("temperature", "?")
            wind = curr.get("windspeed", "?")
            wcode = curr.get("weathercode", 0)
            code_desc = {
                0: "clear sky",
                1: "mainly clear", 2: "partly cloudy", 3: "overcast",
                45: "foggy", 48: "depositing rime fog",
                51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
                61: "slight rain", 63: "moderate rain", 65: "heavy rain",
                71: "slight snow", 73: "moderate snow", 75: "heavy snow",
                77: "snow grains",
                80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
                85: "slight snow showers", 86: "heavy snow showers",
                95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail"
            }
            desc = code_desc.get(wcode, "unknown condition")
            return f"Weather in {name}, {country}: {temp}°C, {desc}, wind speed {wind} km/h."
        except Exception as exc:
            return f"Error: Failed to get weather for {city}: {exc}"


# ------------------------------------------------------------- GitHub API Tool
class GitHubTool:
    """Interact with GitHub repositories (requires GITHUB_TOKEN)."""

    name = "github_tool"
    description = (
        "Interact with GitHub: search repositories, list contents, or read files. "
        "Actions: 'search_repos' (query), 'list_files' (repo_name, path), 'read_file' (repo_name, path)."
    )
    parameters = {
        "action": "Action to perform ('search_repos', 'list_files', 'read_file').",
        "query": "Search query for repositories (used with action='search_repos').",
        "repo_name": "Full repository name, e.g. 'octocat/Hello-World' (used with list_files, read_file).",
        "path": "File or folder path inside the repository (used with list_files, read_file)."
    }

    def run(self, action: str, query: str = "", repo_name: str = "", path: str = "") -> str:
        from derail.config import get_api_key
        token = get_api_key("GITHUB_TOKEN") or get_api_key("GITHUB_API_KEY")
        if not token:
            return "Error: GITHUB_TOKEN / GITHUB_API_KEY is not configured."
        try:
            from github import Github
            g = Github(token)
            if action == "search_repos":
                if not query:
                    return "Error: query parameter is required for search_repos."
                repos = g.search_repositories(query=query)
                results = []
                for repo in repos[:3]:
                    results.append(f"{repo.full_name}: {repo.description or 'No description'} (stars: {repo.stargazers_count})")
                return " | ".join(results) or "No repositories found."
            elif action == "list_files":
                if not repo_name:
                    return "Error: repo_name parameter is required."
                repo = g.get_repo(repo_name)
                contents = repo.get_contents(path)
                if isinstance(contents, list):
                    items = [f"{c.name}/" if c.type == "dir" else c.name for c in contents]
                    return ", ".join(items)[:_MAX_RESULT_CHARS]
                else:
                    return f"{contents.name} (file, size: {contents.size})"
            elif action == "read_file":
                if not repo_name or not path:
                    return "Error: repo_name and path parameters are required."
                repo = g.get_repo(repo_name)
                file_content = repo.get_contents(path)
                if isinstance(file_content, list):
                    return "Error: path is a directory, not a file."
                return file_content.decoded_content.decode("utf-8", errors="replace")[:_MAX_FILE_CHARS]
            else:
                return f"Error: unknown action {action!r}"
        except Exception as exc:
            return f"Error: GitHub API error: {exc}"


# ------------------------------------------------------------- Browser automation
class BrowserAutomation:
    """Browse websites and extract text content using Playwright."""

    name = "browser_browse"
    description = ("Navigate to a public http(s) URL and return its text "
                   "content. Private and internal addresses are refused.")
    parameters = {"url": "The website URL to browse."}

    def __init__(self, allow_hosts: tuple[str, ...] | None = None,
                 timeout_ms: int = 15000) -> None:
        # `allow_hosts=None` still refuses every non-public address; a task
        # that knows the sites it needs should pass them explicitly.
        self.allow_hosts = tuple(allow_hosts) if allow_hosts else None
        self.timeout_ms = int(timeout_ms)

    def run(self, url: str) -> str:
        if not str(url).strip():
            return "Error: empty URL"
        try:
            safe_url = check_url(str(url), allow_hosts=self.allow_hosts)
        except UrlRefused as exc:
            return f"Error: refused - {exc}"
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                try:
                    page = browser.new_page()
                    page.goto(safe_url, timeout=self.timeout_ms)
                    body_text = page.locator("body").inner_text()
                finally:
                    browser.close()
            return redact_secrets(body_text[:_MAX_FILE_CHARS])
        except Exception as exc:
            return f"Error: Browser navigation failed: {exc}"


# ------------------------------------------------------------- SQL Database Tool
class SQLDatabaseTool:
    """Execute SQL queries against a local SQLite database (SQLAlchemy)."""

    name = "sql_query"
    description = (
        "Execute a read-only SELECT SQL query against the e-commerce database. "
        "Schema contains: 'products' (id, name, category, price, stock, description) "
        "and 'orders' (id, product_id, quantity, order_date, status)."
    )
    parameters = {"query": "The SELECT SQL query to execute."}

    #: The fixture's source of truth: plain SQL, committed and diffable.
    SEED_SQL = Path(__file__).resolve().parent / "fixtures" / "ecommerce_seed.sql"

    MAX_ROWS = 15

    def __init__(self, db_path: str | Path | None = None,
                 max_rows: int | None = None) -> None:
        self.db_path = (Path(db_path) if db_path is not None
                        else self._default_db())
        self.max_rows = int(max_rows) if max_rows is not None else self.MAX_ROWS

    @staticmethod
    def _default_db() -> Path:
        """Built under the runtime root, never inside the committed corpus.

        `traces/` is frozen research data that BASELINE_MANIFEST.json hashes
        and the published Hugging Face dataset ships, so a tool fixture does
        not belong there. This one is built on demand from `SEED_SQL`, which
        is committed, diffable and hashed with the code.
        """
        from derail.harness.record_replay import runtime_root
        return runtime_root() / "fixtures" / "ecommerce.db"

    def _ensure_db(self) -> None:
        """Build the fixture from the committed seed if it is not there yet.

        Deterministic: same seed, same rows. Built into a temporary file and
        renamed, so two concurrent agents cannot observe a half-written
        database.
        """
        if self.db_path.exists():
            return
        if not self.SEED_SQL.exists():
            raise FileNotFoundError(f"missing SQL seed {self.SEED_SQL}")
        import sqlite3
        import tempfile
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.db_path.parent, suffix=".part")
        os.close(fd)
        try:
            con = sqlite3.connect(tmp)
            try:
                con.executescript(self.SEED_SQL.read_text("utf-8"))
                con.commit()
            finally:
                con.close()
            os.replace(tmp, self.db_path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @staticmethod
    def _is_single_read_statement(query: str) -> bool:
        """Lexical read-only check: one statement, starting SELECT or WITH.

        Defence in depth only - the connection itself is opened read-only and
        PRAGMA query_only is set, which is what actually enforces the policy.
        """
        stripped = re.sub(r"--[^\n]*", " ", str(query))
        stripped = re.sub(r"/\*.*?\*/", " ", stripped, flags=re.S).strip()
        if not stripped:
            return False
        if stripped.endswith(";"):
            stripped = stripped[:-1].rstrip()
        if ";" in stripped:                      # a second statement
            return False
        return bool(re.match(r"(?is)^\s*(select|with)\b", stripped))

    def run(self, query: str) -> str:
        if not str(query).strip():
            return "Error: empty query"
        if not self._is_single_read_statement(query):
            return "Error: Only a single SELECT/WITH query is permitted."
        try:
            self._ensure_db()
        except (OSError, FileNotFoundError) as exc:
            return f"Error: database fixture unavailable: {exc}"
        if not self.db_path.exists():
            return "Error: database file not found"
        try:
            from sqlalchemy import create_engine, event, text
            # mode=ro: SQLite refuses writes at the file level; query_only adds
            # the same guarantee for anything the URI misses.
            uri = f"file:{self.db_path.as_posix()}?mode=ro&uri=true"
            engine = create_engine("sqlite:///" + uri)

            @event.listens_for(engine, "connect")
            def _set_query_only(dbapi_conn, _record):  # noqa: ANN001
                dbapi_conn.execute("PRAGMA query_only = ON")

            try:
                with engine.connect() as conn:
                    result = conn.execute(text(query))
                    # Bounded fetch: fetchall() materialised the whole result
                    # set before slicing it.
                    rows = result.fetchmany(self.max_rows)
                    headers = list(result.keys())
                    result.close()
            finally:
                # A pooled connection keeps the file open, which on Windows
                # blocks rebuilding the fixture and leaks a handle per query.
                engine.dispose()
            if not rows:
                return "(no rows returned)"
            lines = [", ".join(headers)]
            lines += [", ".join(str(val) for val in r) for r in rows]
            return redact_secrets("\n".join(lines))[:_MAX_RESULT_CHARS]
        except Exception as exc:
            return f"Error: SQL execution failed: {exc}"


# ------------------------------------------------------------- Vector DB / RAG
_DOCS = [
    {"id": 1, "text": "Qdrant is a vector similarity search engine. It provides a production-ready service with a convenient API to store, search, and manage vectors along with additional payloads."},
    {"id": 2, "text": "To configure the derailment monitor, set GEMINI_API_KEY using python -m derail.config set-key GEMINI_API_KEY. The monitor standardizes features using a Standardizer class."},
    {"id": 3, "text": "Echo State Networks (ESNs) are a type of recurrent neural network. They consist of a random, sparse reservoir and a trainable linear readout layer."},
    {"id": 4, "text": "CUSUM (Cumulative Sum) is a sequential analysis technique. It is typically used for monitoring change detection in data streams and signals."},
    {"id": 5, "text": "The adapter parses step text to extract per-step features including reasoning depth, error flags, latency logs, and tool call success rates."}
]


class VectorSearchTool:
    """Retrieve documentation by BM25 lexical relevance.

    The previous implementation hashed each whole string into an independent
    pseudo-random positive vector, so two texts sharing words had NO shared
    representation and the "similarity" was random ranking (a CUSUM query put
    the CUSUM document fourth) - and it silently returned an error string when
    its backend failed to initialise. This uses a real BM25 score
    over the document corpus: shared content words produce shared relevance,
    it is deterministic and offline (no external service, no embedding model),
    and it always works. Sentence-transformer embeddings remain an opt-in
    upgrade, never enabled implicitly.
    """

    name = "vector_search"
    description = ("Search system documentation and research articles by "
                   "lexical (BM25) relevance to the query.")
    parameters = {"query": "The search query."}

    _K1 = 1.5
    _B = 0.75
    _TOKEN = re.compile(r"[a-z0-9]+")

    def __init__(self, docs: "list[dict] | None" = None) -> None:
        self._docs = docs if docs is not None else _DOCS
        self._doc_tokens = [self._tokenize(d["text"]) for d in self._docs]
        self._doc_len = [len(t) for t in self._doc_tokens]
        self._avg_len = (sum(self._doc_len) / len(self._doc_len)
                         if self._doc_len else 0.0)
        # document frequency per term
        self._df: dict[str, int] = {}
        for toks in self._doc_tokens:
            for term in set(toks):
                self._df[term] = self._df.get(term, 0) + 1

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return cls._TOKEN.findall(str(text).lower())

    def _bm25(self, q_terms: list[str], doc_idx: int) -> float:
        import math
        toks = self._doc_tokens[doc_idx]
        if not toks:
            return 0.0
        n = len(self._docs)
        dl = self._doc_len[doc_idx]
        score = 0.0
        for term in set(q_terms):
            df = self._df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            tf = toks.count(term)
            denom = tf + self._K1 * (1 - self._B + self._B * dl / max(self._avg_len, 1e-9))
            score += idf * (tf * (self._K1 + 1)) / max(denom, 1e-9)
        return score

    def run(self, query: str, limit: int = 2) -> str:
        if not str(query).strip():
            return "Error: empty query"
        q_terms = self._tokenize(query)
        scored = sorted(
            ((self._bm25(q_terms, i), d) for i, d in enumerate(self._docs)),
            key=lambda sd: sd[0], reverse=True)
        hits = [(s, d) for s, d in scored if s > 0][:limit]
        if not hits:
            return f"No documentation results for {query!r}."
        out = [f"[Score: {s:.2f}] {d['text']}" for s, d in hits]
        return " | ".join(out)[:_MAX_RESULT_CHARS]


# ------------------------------------------------------------- MCP Client Tool
class MCPClientTool:
    """Invoke a tool on an MCP server that the *operator* configured.

    The model names a server by identifier; it never supplies a command line.
    A model-supplied `server_cmd` was arbitrary process execution,
    and the old implementation also ignored the protocol error flag, split the
    command with `str.split`, and broke inside a running event loop.
    """

    name = "mcp_call"
    description = ("Invoke a tool hosted by a preconfigured Model Context "
                   "Protocol (MCP) server, named by its configured identifier.")
    parameters = {
        "server": "Identifier of a configured MCP server.",
        "tool_name": "The name of the tool on that server to run.",
        "arguments": "JSON string of arguments to pass to the tool."
    }

    def __init__(self, servers: dict[str, list[str]] | None = None,
                 timeout_s: float = 30.0) -> None:
        # {identifier: argv}. Empty by default: with no operator configuration
        # the tool can do nothing at all.
        self.servers = {str(k): [str(a) for a in v]
                        for k, v in (servers or {}).items()}
        self.timeout_s = float(timeout_s)

    def run(self, server: str, tool_name: str, arguments: str = "{}") -> str:
        if not server or not tool_name:
            return "Error: server and tool_name are required."
        argv = self.servers.get(str(server))
        if argv is None:
            known = sorted(self.servers) or ["(none configured)"]
            return (f"Error: refused - unknown MCP server {server!r}. "
                    f"Configured servers: {known}")
        try:
            args_dict = json.loads(arguments) if arguments else {}
        except Exception:
            return "Error: arguments must be a valid JSON string."
        if not isinstance(args_dict, dict):
            return "Error: arguments must be a JSON object."

        import asyncio

        async def _call() -> str:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(command=argv[0], args=list(argv[1:]))
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await asyncio.wait_for(session.initialize(), self.timeout_s)
                    listing = await asyncio.wait_for(session.list_tools(),
                                                     self.timeout_s)
                    available = [t.name for t in listing.tools]
                    if tool_name not in available:
                        return (f"Error: Tool {tool_name!r} not found on MCP "
                                f"server. Available: {available}")
                    result = await asyncio.wait_for(
                        session.call_tool(tool_name, arguments=args_dict),
                        self.timeout_s)
                    texts = [c.text for c in result.content
                             if hasattr(c, "text")]
                    body = "\n".join(texts)
                    # The protocol reports tool failures in-band; surfacing
                    # them as success corrupted the error channel.
                    if getattr(result, "isError", False):
                        return f"Error: MCP tool reported failure: {body}"
                    return body

        def _run_blocking() -> str:
            return asyncio.run(asyncio.wait_for(_call(), self.timeout_s * 3))

        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                out = _run_blocking()            # no loop here: run directly
            else:
                # Inside a running loop asyncio.run() would raise; hand the
                # call to a worker thread with its own loop.
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    out = pool.submit(_run_blocking).result(
                        timeout=self.timeout_s * 4)
            return redact_secrets(str(out))[:_MAX_RESULT_CHARS]
        except TimeoutError:
            return f"Error: MCP call timed out after {self.timeout_s:g}s"
        except Exception as exc:
            return f"Error: MCP call failed: {type(exc).__name__}: {exc}"


def _tool_factories(fs_root: str | Path | None) -> dict[str, Callable[[], object]]:
    """name -> zero-argument constructor for every tool this module ships."""
    factories: dict[str, Callable[[], object]] = {
        "python": PythonREPL,
        "wikipedia_search": WikipediaSearch,
        "arxiv_search": ArxivSearch,
        "web_search": DuckDuckGoSearch,
        "get_weather": OpenMeteoWeather,
        "tavily_search": TavilySearch,
        "github_tool": GitHubTool,
        "browser_browse": BrowserAutomation,
        "sql_query": SQLDatabaseTool,
        "vector_search": VectorSearchTool,
        "mcp_call": MCPClientTool,
    }
    if fs_root is not None:
        factories["read_file"] = lambda: ReadFile(fs_root)
        factories["list_dir"] = lambda: ListDir(fs_root)
    return factories


def available_tool_names(fs_root: str | Path | None = None) -> list[str]:
    return sorted(_tool_factories(fs_root))


def build_registry(allow: "list[str] | tuple[str, ...]",
                   fs_root: str | Path | None = None):
    """A ToolRegistry containing exactly the named tools.

    This is the entry point collectors should use: pass `task.tools`, so an
    agent is only ever offered the capabilities its own prompt requires, and
    the dangerous three are granted deliberately or not at all.
    """
    from derail.harness.tools import ToolRegistry
    factories = _tool_factories(fs_root)
    unknown = [n for n in allow if n not in factories]
    if unknown:
        raise ValueError(
            f"unknown tool(s) {unknown}; available: {sorted(factories)}"
            + ("" if fs_root is not None else
               " (filesystem tools need fs_root=)"))
    return ToolRegistry([factories[n]() for n in allow])


def default_registry(fs_root: str | Path | None = None,
                     allow: "list[str] | tuple[str, ...] | None" = None):
    """Registry of the tools that are safe to expose without a per-task decision.

    Excludes `python`, `browser_browse` and `mcp_call`: host code execution,
    arbitrary navigation and process spawning are never handed to a model
    implicitly.  Callers that genuinely need them pass `allow=`.
    """
    factories = _tool_factories(fs_root)
    names = (list(allow) if allow is not None
             else [n for n in factories if n not in DANGEROUS_TOOLS])
    return build_registry(names, fs_root=fs_root)



# --------------------------------------------------------------- smoke test
if __name__ == "__main__":
    from derail.harness.tools import ToolRegistry

    # --- Python REPL: real subprocess, fully offline & deterministic ---
    repl = PythonREPL(timeout_s=10)
    assert repl.run("print(6 * 7)") == "42"
    assert repl.run("print(sum(range(101)))") == "5050"
    assert repl.run("x = 1").startswith("(no output")
    assert repl.run("1/0").startswith("Error:")           # traceback -> error
    assert repl.run("   ").startswith("Error: empty")
    assert repl.run("import time; time.sleep(30)").startswith("Error: timed out")

    # --- HTTP tools: parse canned payloads (no network) ---
    wiki_json = json.dumps({"query": {"search": [
        {"title": "Echo state network",
         "snippet": 'a type of <span class="searchmatch">reservoir</span> computer'},
        {"title": "Reservoir computing", "snippet": "framework for computation"},
    ]}}).encode()
    wiki = WikipediaSearch(get=lambda url: wiki_json)
    r = wiki.run("reservoir computing")
    assert "Echo state network" in r and "reservoir computer" in r
    assert "searchmatch" not in r, "HTML not stripped"

    arxiv_xml = (
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        '<entry><title>Online Derailment Detection</title>'
        '<id>http://arxiv.org/abs/2601.01234v1</id>'
        '<author><name>A. Researcher</name></author></entry>'
        '</feed>').encode()
    arx = ArxivSearch(get=lambda url: arxiv_xml)
    ra = arx.run("agent monitoring")
    assert "Online Derailment Detection" in ra and "2601.01234v1" in ra

    # --- DuckDuckGo: parse a canned instant-answer payload ---
    ddg_json = json.dumps({
        "AbstractText": "An echo state network is a reservoir computer.",
        "RelatedTopics": [{"Text": "Reservoir computing paradigm"},
                          {"Text": "Recurrent neural networks"}],
    }).encode()
    ddg = DuckDuckGoSearch(get=lambda url: ddg_json)
    rd = ddg.run("echo state network")
    assert "reservoir computer" in rd and "Recurrent neural" in rd

    # --- sandboxed filesystem: read/list inside root, reject escapes ---
    with tempfile.TemporaryDirectory() as ws:
        (Path(ws) / "notes.txt").write_text("hello world", "utf-8")
        (Path(ws) / "sub").mkdir()
        rf, ld = ReadFile(ws), ListDir(ws)
        assert rf.run("notes.txt") == "hello world"
        assert "notes.txt" in ld.run(".") and "sub/" in ld.run(".")
        assert rf.run("../secret").startswith("Error: path escapes")
        assert rf.run("nope.txt").startswith("Error: no such file")

    # --- new tools smoke tests ---
    # Weather
    w = OpenMeteoWeather()
    w_res = w.run("Prague")
    assert "Weather in Prague" in w_res, f"Unexpected weather result: {w_res}"
    assert "°C" in w_res, f"Expected °C in weather result: {w_res}"

    # SQL DB query
    sql = SQLDatabaseTool()
    sql_res = sql.run("SELECT name, price FROM products WHERE category = 'Electronics' LIMIT 1")
    assert "name, price" in sql_res, f"Expected headers in: {sql_res}"
    assert "Wireless Noise-Canceling Headphones" in sql_res, f"Expected product in: {sql_res}"

    # SQL query safety
    sql_bad = sql.run("INSERT INTO products (name) VALUES ('dangerous')")
    assert sql_bad.startswith("Error: Only a single SELECT"), \
        f"Expected safety block: {sql_bad}"

    # Vector search
    vs = VectorSearchTool()
    vs_res = vs.run("CUSUM")
    assert "[Score: " in vs_res, f"Unexpected vector search result: {vs_res}"

    # Tavily Search: graceful error with OR without a configured key — the
    # empty query trips before any network call, so the smoke test stays
    # offline and deterministic on machines that do have TAVILY_API_KEY set.
    tav = TavilySearch()
    tav_res = tav.run("")
    assert tav_res.startswith("Error:"), f"Expected graceful error: {tav_res}"

    # GitHub Tool: same pattern — missing query errors before any API call.
    git = GitHubTool()
    git_res = git.run("search_repos", query="")
    assert git_res.startswith("Error:"), f"Expected graceful error: {git_res}"

    # Browser automation: an internal/loopback host is refused BEFORE launching
    # a browser (SSRF containment), so a localhost URL returns the
    # refusal rather than a navigation error.
    browser = BrowserAutomation()
    b_err = browser.run("http://localhost:9999/nonexistent")
    assert b_err.startswith("Error: refused"), f"Expected refusal: {b_err}"

    # MCP client: a model-supplied command line is refused; only operator-
    # configured server identifiers are accepted.
    mcp_t = MCPClientTool()
    mcp_res = mcp_t.run("py fake_server.py", "fake_tool")
    assert mcp_res.startswith("Error: refused"), f"Expected refusal: {mcp_res}"

    # --- contract: they register and their calls carry telemetry ---
    reg = ToolRegistry([PythonREPL(), wiki, arx, ddg, w, sql, vs])
    res = reg.call("python", {"code": "print('hi')"})
    assert res.content == "hi" and not res.is_error
    assert res.step_bit().startswith("[python(")

    # --- best-effort live probe (skips cleanly offline) ---
    try:
        live = WikipediaSearch().run("echo state network")
        note = f"live wiki OK: {live[:60]}..."
    except Exception as exc:  # noqa: BLE001 — offline / rate-limited is fine
        note = f"live probe skipped ({type(exc).__name__})"

    print("PASS real_tools.py smoke test |", note)

