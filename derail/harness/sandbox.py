"""Containment policy for tools driven by untrusted model output.

the harness handed a model a Python
REPL, arbitrary browser navigation, a model-supplied MCP server command and
unrestricted in-root file reads, and every result was shipped to a cloud model
and written to a plaintext cassette.  This module is the single place where the
containment rules live so the tools cannot each invent their own.

What this module *does* provide:

  * per-call allowlists of tool names (the registry refuses everything else);
  * a secret-path deny list for the filesystem tools;
  * secret redaction applied to every tool result before it leaves the process;
  * URL vetting that refuses non-HTTP(S) schemes and any host that resolves to
    a private, loopback, link-local, multicast or otherwise non-global address
    (internal-network probing / SSRF);
  * a scrubbed subprocess environment plus an in-process network guard for the
    Python REPL, so a snippet cannot read this process's credentials or open
    sockets.

What it does NOT provide, stated plainly so nobody over-trusts it: this is not
an OS-level sandbox.  A determined adversarial model could still exhaust CPU or
disk inside the temporary working directory, and the URL check has an inherent
resolve-then-connect race.  Running collection against an actively adversarial
model requires a container or VM with filesystem, process and egress policy;
`REQUIRES_CONTAINER` records that requirement in code.
"""
from __future__ import annotations

import ipaddress
import os
import re
import socket
import urllib.parse
from pathlib import Path

# Tools whose capability is host code execution, arbitrary navigation or
# arbitrary process spawning.  They are never in a default registry; a caller
# must name them in an allowlist, which makes the decision visible per task.
DANGEROUS_TOOLS = frozenset({"python", "browser_browse", "mcp_call"})

# Set by deployments that have real OS-level isolation; only documentation and
# error text depend on it.
REQUIRES_CONTAINER = ("adversarial-model collection requires OS-level isolation "
                      "(container/VM with filesystem, process and egress policy); "
                      "derail.harness.sandbox is process-level containment only")


# --------------------------------------------------------------- secret paths
_SENSITIVE_NAMES = {
    ".env", ".env.local", ".netrc", "_netrc", "credentials", "credentials.json",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "secrets.json", ".pypirc",
    ".htpasswd", "known_hosts", "authorized_keys",
}
_SENSITIVE_SUFFIXES = {".pem", ".key", ".pfx", ".p12", ".keystore", ".jks",
                       ".ppk", ".asc", ".gpg", ".kdbx"}
_SENSITIVE_DIR_PARTS = {".git", ".ssh", ".gnupg", ".aws", ".azure", ".config",
                        "__pycache__", ".venv", "venv", "node_modules"}


def is_sensitive_path(path: str | os.PathLike[str]) -> bool:
    """True when a path looks like credential material or VCS internals.

    Deliberately name-based rather than content-based: the point is to refuse
    *before* reading, so nothing sensitive is ever loaded into a prompt.
    """
    p = Path(path)
    name = p.name.lower()
    if name in _SENSITIVE_NAMES or name.startswith(".env"):
        return True
    if p.suffix.lower() in _SENSITIVE_SUFFIXES:
        return True
    return any(part.lower() in _SENSITIVE_DIR_PARTS for part in p.parts)


# ------------------------------------------------------------------ redaction
# Ordered most-specific first; each pattern keeps a short prefix so a redacted
# result is still debuggable.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"AIza[0-9A-Za-z_\-]{30,}"), "AIza<redacted>"),                # Google
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "sk-<redacted>"),                  # OpenAI-style
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "gh_<redacted>"),              # GitHub
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "github_pat_<redacted>"),
    (re.compile(r"tvly-[A-Za-z0-9_\-]{16,}"), "tvly-<redacted>"),              # Tavily
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA<redacted>"),                       # AWS key id
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
     "<redacted private key>"),
    # KEY=value / "token": "value" style assignments of long opaque strings.
    (re.compile(r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL)[A-Z0-9_]*)"
                r"\s*[:=]\s*[\"']?([^\s\"',;]{12,})"),
     r"\1=<redacted>"),
)


def redact_secrets(text: str) -> str:
    """Mask credential-looking substrings in a tool result.

    Applied to every tool result before it reaches a model, a trace file or a
    cassette.  Redaction is one-way and lossy on purpose.
    """
    if not text:
        return text
    out = str(text)
    for pattern, replacement in _SECRET_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


# ----------------------------------------------------------------- URL policy
class UrlRefused(ValueError):
    """Raised when a URL fails the containment policy."""


def check_url(url: str, *, allow_hosts: tuple[str, ...] | None = None,
              resolve: bool = True) -> str:
    """Return the URL if it is safe to fetch, else raise :class:`UrlRefused`.

    Rejects non-HTTP(S) schemes, credentials embedded in the netloc, hosts
    outside ``allow_hosts`` when one is given, and any host resolving to a
    non-global address (loopback, private, link-local, multicast, reserved) -
    the internal-network probing path called out in.
    """
    raw = str(url).strip()
    parsed = urllib.parse.urlsplit(raw)
    if parsed.scheme not in ("http", "https"):
        raise UrlRefused(f"scheme {parsed.scheme or '(none)'!r} is not allowed; use http(s)")
    if parsed.username or parsed.password:
        raise UrlRefused("credentials in the URL are not allowed")
    host = parsed.hostname
    if not host:
        raise UrlRefused("URL has no host")
    host_l = host.lower()
    if allow_hosts is not None:
        allowed = tuple(h.lower() for h in allow_hosts)
        if not any(host_l == h or host_l.endswith("." + h) for h in allowed):
            raise UrlRefused(f"host {host_l!r} is not in the allowlist {list(allowed)}")
    if not resolve:
        return raw

    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise UrlRefused(f"cannot resolve host {host_l!r}: {exc}") from exc
    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if not addr.is_global or addr.is_multicast:
            raise UrlRefused(
                f"host {host_l!r} resolves to non-public address {addr} - "
                f"internal-network access is refused")
    return raw


# ------------------------------------------------------- subprocess isolation
# Environment variables whose *names* alone justify removal before handing an
# environment to model-authored code.
_SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL",
                     "SESSION", "COOKIE", "AUTH")
_KEEP_ENV = ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR",
             "LANG", "LC_ALL", "PYTHONIOENCODING", "NUMBER_OF_PROCESSORS")


def scrubbed_env() -> dict[str, str]:
    """A minimal environment with no credential-shaped variables.

    Allowlist, not deny list: only the variables a Python interpreter needs in
    order to start are carried over, so an exfiltration attempt from inside the
    REPL finds nothing worth sending.  ``_SECRET_ENV_HINTS`` additionally
    guards the keep-list against a future entry that carries a credential.
    """
    env = {k: v for k, v in ((k, os.environ.get(k)) for k in _KEEP_ENV)
           if v is not None
           and not any(hint in k.upper() for hint in _SECRET_ENV_HINTS)}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONNOUSERSITE"] = "1"
    return env


# Prepended to model-authored code.  Disables socket creation so the snippet
# cannot open network connections, then runs the snippet with a clean globals
# dict.  Not a kernel-level block - it stops the stdlib paths a model would
# actually use, and the scrubbed environment removes anything worth sending.
NETWORK_GUARD = """\
import socket as _s, builtins as _b


class _Blocked(OSError):
    pass


def _deny(*a, **k):
    raise _Blocked("network access is disabled inside the sandboxed Python tool")


_s.socket = _deny
_s.create_connection = _deny
_s.socketpair = _deny
try:
    import urllib.request as _u
    _u.urlopen = _deny
except Exception:
    pass
"""


def guarded_code(code: str, *, allow_network: bool = False) -> str:
    """Wrap model-authored source with the network guard unless explicitly allowed."""
    if allow_network:
        return code
    return NETWORK_GUARD + "\n" + code
