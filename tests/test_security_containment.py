"""Containment of tools driven by untrusted model output.

Covers the validity rules this module enforces.
Every test states the behaviour the review asked for, not the implementation,
so a future refactor that reopens the hole fails here.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from derail.harness import real_tools, record_replay, sandbox
from derail.harness.record_replay import (Cassette, price_call, request_key)
from derail.harness.tasks import REAL_TASKS
from derail.harness.tools import SimpleTool, ToolRegistry


# ---------------------------------------------------------------------
def test_serving_paths_cannot_write_into_the_committed_corpus():
    """`traces/` is hashed by BASELINE_MANIFEST.json; a demo must not add to it.

    Collectors legitimately write there — their recordings ARE the dataset —
    so this pins the split rather than banning writes outright: every serving
    call site passes `serving=True`, which redirects recording under
    `runtime_root()`.
    """
    import inspect

    from derail.harness import agent_loop, demo_real
    from derail.harness.record_replay import runtime_root

    root = runtime_root().resolve()
    traces = (Path(__file__).resolve().parents[1] / "traces").resolve()
    assert not str(root).startswith(str(traces)), \
        f"runtime root {root} is inside the committed corpus"

    src = inspect.getsource(demo_real)
    assert "_cassette(serving=True)" in src, \
        "the demo_real serving path no longer marks its cassette as serving"
    assert "cassette=_cassette()" in src, \
        "the collector should keep writing its recordings into the dataset"
    assert "serving=True" in inspect.getsource(agent_loop._run_live)


def test_a_serving_cassette_records_outside_the_source_directory(tmp_path,
                                                                 monkeypatch):
    from derail.harness.record_replay import RUNTIME_DIR_ENV

    src, rt = tmp_path / "corpus", tmp_path / "rt"
    src.mkdir()
    monkeypatch.setenv(RUNTIME_DIR_ENV, str(rt))
    key = request_key("m", [{"role": "user", "text": "hi"}], {})

    serving = Cassette(src, mode="auto", serving=True)
    serving.call(key, lambda: {"text": "x"})
    assert list(src.iterdir()) == [], "serving wrote into the source corpus"
    assert list(rt.rglob("*.json")), "serving recorded nothing at all"


def test_default_registry_excludes_host_code_and_navigation_tools():
    names = set(real_tools.default_registry().names())
    assert not (names & sandbox.DANGEROUS_TOOLS), (
        f"default registry still exposes {names & sandbox.DANGEROUS_TOOLS}")
    assert "wikipedia_search" in names, "safe tools must still be available"


def test_dangerous_tools_require_an_explicit_allowlist():
    reg = real_tools.build_registry(("python", "wikipedia_search"))
    assert set(reg.names()) == {"python", "wikipedia_search"}


def test_build_registry_rejects_unknown_tools():
    with pytest.raises(ValueError, match="unknown tool"):
        real_tools.build_registry(("definitely_not_a_tool",))


def test_filesystem_tools_need_an_explicit_root():
    assert "read_file" not in real_tools.available_tool_names()
    assert "read_file" in real_tools.available_tool_names(fs_root=".")


def _tools_required_by_success_criteria() -> dict[str, set[str]]:
    """{task name: tool names its success_fn checks for} - read from the AST.

    The success criteria are the ground truth for what a task must be able to
    call; the allowlist has to cover them exactly.
    """
    import ast
    src = Path(real_tools.__file__).resolve().parents[0] / "tasks.py"
    tree = ast.parse(src.read_text("utf-8"))
    required: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "RealTask"):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        name = ast.literal_eval(kwargs["name"])
        names: set[str] = set()
        for sub in ast.walk(kwargs["success_fn"]):
            if (isinstance(sub, ast.Call)
                    and getattr(sub.func, "id", "") == "_tool_called"):
                names.add(ast.literal_eval(sub.args[1]))
        required[name] = names
    return required


def test_every_real_task_declares_a_minimal_allowlist():
    required = _tools_required_by_success_criteria()
    assert len(required) == len(REAL_TASKS)
    for task in REAL_TASKS:
        assert task.tools, f"{task.name} has no tool allowlist"
        reg = real_tools.build_registry(task.tools, fs_root=".")
        assert set(reg.names()) == set(task.tools)
        missing = required[task.name] - set(task.tools)
        assert not missing, f"{task.name} cannot succeed without {missing}"
        # Nothing dangerous is granted unless the task genuinely needs it.
        gratuitous = (set(task.tools) & sandbox.DANGEROUS_TOOLS) - required[task.name]
        assert not gratuitous, f"{task.name} grants {gratuitous} without needing it"


# ---------------------------------------------------------------------
def test_python_repl_cannot_open_sockets():
    repl = real_tools.PythonREPL(timeout_s=20)
    out = repl.run("import socket; s = socket.socket(); print('OPENED', s)")
    assert "OPENED" not in out
    assert "network access is disabled" in out or out.startswith("Error:"), out


def test_python_repl_environment_has_no_credentials(monkeypatch):
    monkeypatch.setenv("DERAIL_TEST_API_KEY", "AIzaSyTOTALLYFAKEKEYVALUE1234567890")
    repl = real_tools.PythonREPL(timeout_s=20)
    out = repl.run("import os; print(os.environ.get('DERAIL_TEST_API_KEY', 'ABSENT'))")
    assert "ABSENT" in out, out
    assert "AIza" not in out


def test_read_file_refuses_credential_paths():
    with tempfile.TemporaryDirectory() as ws:
        root = Path(ws)
        (root / ".env").write_text("GEMINI_API_KEY=AIzaSyFAKEFAKEFAKEFAKEFAKE12345\n", "utf-8")
        (root / ".git").mkdir()
        (root / ".git" / "config").write_text("[remote]\n url = https://x\n", "utf-8")
        (root / "notes.txt").write_text("plain text", "utf-8")

        rf, ld = real_tools.ReadFile(root), real_tools.ListDir(root)
        assert rf.run(".env").startswith("Error: refused")
        assert rf.run(".git/config").startswith("Error: refused")
        assert rf.run("notes.txt") == "plain text"        # normal reads still work
        listing = ld.run(".")
        assert "notes.txt" in listing
        assert ".env" not in listing and ".git" not in listing


def test_redaction_masks_known_credential_shapes():
    samples = [
        ("key AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456", "AIzaSyABCDEF"),
        ("sk-abcdefghijklmnopqrstuvwxyz0123", "sk-abcdefgh"),
        ("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345", "ghp_ABCDEF"),
        ("GEMINI_API_KEY=AIzaSy-not-a-real-key-value", "not-a-real-key"),
    ]
    for raw, secret_fragment in samples:
        assert secret_fragment not in sandbox.redact_secrets(raw), raw


def test_registry_redacts_before_recording_or_returning():
    leaky = SimpleTool("leaky", "leaks", {"x": "anything"},
                       lambda x: "token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345 here")
    reg = ToolRegistry([leaky])
    with tempfile.TemporaryDirectory() as d:
        cas = Cassette(d, mode="auto")
        res = reg.call("leaky", {"x": "1"}, cassette=cas)
        assert "ghp_ABCDEFGHIJ" not in res.content
        stored = "".join(p.read_text("utf-8") for p in Path(d).glob("*.json"))
        assert "ghp_ABCDEFGHIJ" not in stored, "secret written to a cassette"


# ------------------------------------------------------ (SSRF paths)
@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/x",
    "http://127.0.0.1:8765/state",
    "http://localhost/admin",
    "http://169.254.169.254/latest/meta-data/",   # cloud metadata
    "http://10.0.0.5/internal",
    "http://192.168.1.1/",
    "http://user:pw@example.com/",
])
def test_check_url_refuses_dangerous_targets(url):
    with pytest.raises(sandbox.UrlRefused):
        sandbox.check_url(url)


def test_check_url_host_allowlist():
    assert sandbox.check_url("https://en.wikipedia.org/wiki/Main_Page",
                             allow_hosts=("wikipedia.org",), resolve=False)
    with pytest.raises(sandbox.UrlRefused):
        sandbox.check_url("https://evil.example.com/x",
                          allow_hosts=("wikipedia.org",), resolve=False)


def test_browser_refuses_internal_targets_without_launching():
    out = real_tools.BrowserAutomation().run("http://127.0.0.1:9/")
    assert out.startswith("Error: refused"), out


# ---------------------------------------------------------------------
def test_mcp_refuses_model_supplied_commands():
    tool = real_tools.MCPClientTool()
    assert "server_cmd" not in tool.parameters, (
        "the model must not be able to supply a command line")
    out = tool.run(server="py -c \"print(1)\"", tool_name="anything")
    assert out.startswith("Error: refused"), out


def test_mcp_only_accepts_configured_identifiers():
    tool = real_tools.MCPClientTool(servers={"docs": ["py", "server.py"]})
    assert tool.run(server="other", tool_name="x").startswith("Error: refused")


# ---------------------------------------------------------------------
def test_sql_tool_builds_its_fixture_outside_the_committed_corpus():
    """The fixture is a tool asset, not research data, and not a binary.

    `traces/` is frozen research data that the manifest hashes and the
    published dataset ships, so a fixture must not live there, must be
    rebuildable, and must be hashed with the code it belongs to.
    """
    from derail.harness.record_replay import runtime_root

    tool = real_tools.SQLDatabaseTool()
    repo = Path(__file__).resolve().parents[1]
    assert tool.SEED_SQL.exists(), "the SQL seed must be committed"
    assert repo in tool.SEED_SQL.parents, tool.SEED_SQL
    assert runtime_root().resolve() in tool.db_path.resolve().parents, (
        f"fixture {tool.db_path} is not under the runtime root")
    assert (repo / "traces") not in tool.db_path.resolve().parents, (
        "the fixture is back inside the committed corpus")
    assert not list((repo / "traces").glob("*.db")), \
        "a database file reappeared in the trace corpus"


def test_the_sql_fixture_is_rebuilt_deterministically_when_missing(tmp_path):
    import hashlib

    db = tmp_path / "fixture.db"
    tool = real_tools.SQLDatabaseTool(db_path=db)
    assert "Wireless Noise-Canceling Headphones" in tool.run(
        "SELECT name FROM products WHERE id = 1")
    first = hashlib.sha256(db.read_bytes()).hexdigest()

    db.unlink()                       # a wiped runtime dir must self-heal
    assert tool.run("SELECT COUNT(*) FROM orders").endswith("6")
    assert hashlib.sha256(db.read_bytes()).hexdigest() == first, \
        "rebuilding the fixture from the same seed is not deterministic"


@pytest.mark.parametrize("query", [
    "UPDATE products SET price = 0",
    "DROP TABLE products",
    "SELECT 1; DROP TABLE products",
    "-- comment\nDELETE FROM products",
    "/* hide */ INSERT INTO products VALUES (1)",
    "PRAGMA writable_schema = 1",
])
def test_sql_tool_refuses_writes(query):
    assert real_tools.SQLDatabaseTool().run(query).startswith("Error:"), query


def test_sql_tool_returns_bounded_rows():
    tool = real_tools.SQLDatabaseTool(max_rows=2)
    out = tool.run("SELECT id, name FROM products")
    assert out.count("\n") == 2, out          # header + 2 rows


def test_sql_connection_is_read_only_even_if_the_lexer_is_fooled():
    tool = real_tools.SQLDatabaseTool()
    # A SELECT that a lexical check accepts but which mutates via a CTE-style
    # trick must still fail: the connection itself is read-only.
    out = tool.run("WITH x AS (SELECT 1) SELECT * FROM x")
    assert not out.startswith("Error:"), out   # legitimate read still works
    import sqlalchemy
    engine = sqlalchemy.create_engine(
        "sqlite:///" + f"file:{tool.db_path.as_posix()}?mode=ro&uri=true")
    with engine.connect() as conn, pytest.raises(Exception):
        conn.execute(sqlalchemy.text("UPDATE products SET price = 0"))


# ---------------------------------------------------------------------
def test_arxiv_uses_https():
    seen = {}

    def fake_get(url: str) -> bytes:
        seen["url"] = url
        return b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'

    real_tools.ArxivSearch(get=fake_get).run("echo state networks")
    assert seen["url"].startswith("https://"), seen["url"]


# ---------------------------------------------------------------------
@pytest.mark.parametrize("key", [
    "../escape", "..\\escape", "a/b", "a\\b", "/abs", "", ".", "..",
    "x" * 200,
])
def test_cassette_rejects_unsafe_keys(key):
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            Cassette(d)._file(key)


# ---------------------------------------------------------------------
def test_cassette_does_not_cache_errors():
    calls = {"n": 0}

    def failing():
        calls["n"] += 1
        return {"content": "Error: rate limited", "is_error": True}

    with tempfile.TemporaryDirectory() as d:
        cas = Cassette(d, mode="auto")
        key = request_key("tool", "x", {}, namespace="test")
        cas.call(key, failing, is_error=lambda r: r["is_error"])
        cas.call(key, failing, is_error=lambda r: r["is_error"])
        assert calls["n"] == 2, "a transient failure was replayed as a result"
        assert not list(Path(d).glob("*.json"))


def test_cassette_honours_ttl():
    with tempfile.TemporaryDirectory() as d:
        key = request_key("weather", "Lisbon", namespace="test")
        Cassette(d, mode="auto").call(key, lambda: {"t": 1})
        fresh = Cassette(d, mode="auto", ttl_s=3600).call(key, lambda: {"t": 2})
        assert fresh == {"t": 1}
        stale = Cassette(d, mode="auto", ttl_s=0.0).call(key, lambda: {"t": 3})
        assert stale == {"t": 3}, "an expired recording was replayed"


def test_recorded_at_is_never_in_the_future():
    """The stamp must not be later than the moment of writing.

    `round(t, 3)` goes to the nearest millisecond, so it stamped recordings up
    to half a millisecond ahead. A read inside that window computes a negative
    age, and a zero TTL then replays an expired record. That is why the TTL
    test passed here and failed on CI: the window is real, and a fast machine
    lands inside it.
    """
    # Tested on the stamp itself rather than through a written file: a write
    # takes longer than the half-millisecond window on a slow machine, which
    # is exactly why the defect was invisible locally and fired on CI. One
    # sample is also not enough, since rounding to nearest goes up only about
    # half the time.
    ahead = []
    for _ in range(500):
        stamp = record_replay.recorded_at_stamp()
        now = time.time()
        if stamp > now:
            ahead.append(stamp - now)

    assert not ahead, (
        f"{len(ahead)} of 500 stamps were in the future, by up to "
        f"{max(ahead) * 1000:.3f} ms — a read inside that window computes a "
        f"negative age and replays an expired recording")


def test_zero_ttl_expires_a_future_dated_recording():
    """Even a stamp from the future must not read as fresh.

    Clock steps backwards happen, and a record that looks newer than now would
    otherwise be replayed forever regardless of the TTL asked for.
    """
    with tempfile.TemporaryDirectory() as d:
        key = request_key("weather", "Faro", namespace="test")
        Cassette(d, mode="auto").call(key, lambda: {"t": 1})

        recording = next(Path(d).glob("*.json"))
        payload = json.loads(recording.read_text("utf-8"))
        payload["recorded_at"] = time.time() + 60.0      # a minute ahead
        recording.write_text(json.dumps(payload), encoding="utf-8")

        served = Cassette(d, mode="auto", ttl_s=0.0).call(key, lambda: {"t": 2})
        assert served == {"t": 2}, "a future-dated recording was replayed"


def test_cassette_zero_ttl_expires_a_same_tick_recording(monkeypatch):
    """A zero TTL must expire a record read in the clock tick it was written.

    Read and write can land on one `time.time()` value, giving an age of
    exactly 0.0, which a `>` comparison calls fresh. Whether the TTL held then
    depended on how fast the machine was: this passed locally and failed on CI
    on the same commit, intermittently.

    The clock is frozen rather than raced, so age is exactly zero every run.
    Freezing it also gives the test teeth --- with the boundary written as `>`
    it fails, which a wall-clock version does not, because a few microseconds
    of real elapsed time hide the defect.
    """
    with tempfile.TemporaryDirectory() as d:
        key = request_key("weather", "Porto", namespace="test")
        Cassette(d, mode="auto").call(key, lambda: {"t": 1})

        recording = next(Path(d).glob("*.json"))
        frozen = json.loads(recording.read_text("utf-8"))["recorded_at"]
        monkeypatch.setattr(record_replay.time, "time", lambda: frozen)

        served = Cassette(d, mode="auto", ttl_s=0.0).call(key, lambda: {"t": 2})
        assert served == {"t": 2}, "a zero TTL replayed a same-tick recording"


def test_cassette_migrates_legacy_keys_instead_of_re_paying():
    calls = {"n": 0}

    def live():
        calls["n"] += 1
        return {"content": "expensive"}

    with tempfile.TemporaryDirectory() as d:
        legacy = request_key("tool", "search", {"q": "x"})
        new = request_key("tool", "search", {"q": "x"}, "fingerprint",
                          namespace="tool/v2")
        cas = Cassette(d, mode="auto")
        cas.call(legacy, live)                     # the old recording
        assert calls["n"] == 1
        out = cas.call(new, live, legacy_keys=(legacy,))
        assert out == {"content": "expensive"} and calls["n"] == 1
        assert (Path(d) / f"{new}.json").exists(), "not migrated to the new key"


def test_cassette_key_includes_tool_configuration():
    a = ToolRegistry([real_tools.SQLDatabaseTool(max_rows=5)])
    b = ToolRegistry([real_tools.SQLDatabaseTool(max_rows=50)])
    from derail.harness.tools import tool_fingerprint
    assert tool_fingerprint(a._tools["sql_query"]) != tool_fingerprint(b._tools["sql_query"])


def test_cassette_records_are_valid_json_after_write():
    with tempfile.TemporaryDirectory() as d:
        cas = Cassette(d, mode="auto")
        key = request_key("x", namespace="test")
        cas.call(key, lambda: {"a": 1})
        payload = json.loads((Path(d) / f"{key}.json").read_text("utf-8"))
        assert payload["response"] == {"a": 1} and "recorded_at" in payload
        assert not list(Path(d).glob("*.tmp")), "temp file left behind"


# ---------------------------------------------------------------------
def test_pro_long_context_tier_keys_on_prompt_tokens():
    # 150k prompt + 100k output: the prompt is under the 200k tier boundary,
    # so the cheap tier applies even though the total exceeds it.
    lo = price_call("gemini-2.5-pro", 150_000, 100_000)
    manual = (150_000 * 1.25 + 100_000 * 10.0) / 1e6
    assert abs(lo - manual) < 1e-9, (lo, manual)
    hi = price_call("gemini-2.5-pro", 250_000, 1_000)
    assert hi > (250_000 * 1.25 + 1_000 * 10.0) / 1e6


@pytest.mark.parametrize("args", [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (5, 0, 10)])
def test_pricing_rejects_impossible_token_counts(args):
    with pytest.raises(ValueError):
        price_call("gemini-2.5-flash", *args)
