"""Audit committed trace corpora for label and collection defects.

found that episodes are labelled by *intent* rather than
by evidence: an injected episode counts as a positive even when the stochastic
injector never fired, and a "healthy" episode counts as a negative
even when the task failed.  This tool decides those questions from
the trace bytes, so a corpus can be quantified before anything is re-collected
and so a collector can refuse a bad episode at write time.

    py -m devtools.trace_audit                     # every corpus, summary
    py -m devtools.trace_audit --corpus real_research7b --verbose
    py -m devtools.trace_audit --json audit.json

Exit status is 1 when any defect is found, so it can gate collection in CI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from dataclasses import asdict, dataclass, field

from derail.harness.inject import _DECOYS, _ERROR_TEMPLATES
from derail.telemetry.events import parse_step_events

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRACES = REPO_ROOT / "traces"

#: Evidence that a given injected class actually mutated a tool result.
#: Matched against the recorded result text of post-onset tool calls.
_MARKERS: dict[str, tuple[str, ...]] = {
    "rate_limit": ("429 Too Many Requests",),
    "timeout": ("timed out after 30s",),
    "sql_timeout": ("SQL statement timeout",),
    "mcp_unavailable": ("unavailable (connection refused)",),
    "browser_fail": ("net::ERR_CONNECTION_TIMED_OUT",),
    "tool_cascade": ("service unavailable (HTTP 503)",),
    "looping": ("Temporary data inconsistency detected",),
    "wrong_document": tuple(d[:40] for d in _DECOYS),
    # Short on purpose: legacy traces truncated results at 100 characters, so
    # the full hijack sentence is often cut off mid-payload.
    "goal_drift": ("[SYSTEM UPDATE:",),
}
#: Classes whose evidence is structural rather than a fixed string.
_STRUCTURAL = {"malformed_json", "context_corruption"}
#: `context_corruption` appends " $<int>" to the shuffled result.
_CORRUPTION_SUFFIX = re.compile(r"\$\d{1,3}\s*$")

#: Study-level classes the real collectors map their own classes onto; the
#: original class is what the marker table knows about.
_MAPPED_BACK = {
    "tool_cascade": ("tool_cascade", "rate_limit", "malformed_json",
                     "sql_timeout", "mcp_unavailable", "browser_fail",
                     "timeout"),
    "grounding_loss": ("wrong_document",),
    "looping": ("looping",),
    "goal_drift": ("goal_drift",),
    "context_corruption": ("context_corruption",),
}


@dataclass
class EpisodeAudit:
    corpus: str
    episode_id: str
    label: str | None
    tau: int | None
    T: int
    defects: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


def _looks_malformed(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        return True
    return False


def _post_onset_results(steps: list[dict], tau: int) -> list[str]:
    out = []
    for step in steps[tau:]:
        for call in parse_step_events(step)[0]:
            if call.result:
                out.append(call.result)
    return out


def _mutation_evidence(label: str, results: list[str],
                       steps: list[dict], tau: int) -> tuple[str, str]:
    """(verdict, explanation) for one injected episode.

    Verdicts: ``confirmed`` (a mutation is visible), ``no_op`` (results were
    recorded and none of them was mutated), ``unverifiable`` (the corpus
    records no tool results after the onset, so the question cannot be
    answered from the file - the corpora).
    """
    candidates = _MAPPED_BACK.get(label, (label,))
    for candidate in candidates:
        for marker in _MARKERS.get(candidate, ()):
            if any(marker in r for r in results):
                return "confirmed", f"{candidate}: {marker[:40]!r}"
        if candidate == "malformed_json" and any(_looks_malformed(r)
                                                 for r in results):
            return "confirmed", "malformed_json: unparseable JSON result"
    if "context_corruption" in candidates:
        # Corruption shuffles the words and appends " $<int>"; that suffix is
        # the only fixed trace it leaves, and it is decisive (verified against
        # all 22 real_research7b corruption episodes).
        if any(_CORRUPTION_SUFFIX.search(r) for r in results):
            return "confirmed", "context_corruption: shuffled result + $<int>"
    # An error flag raised after the onset is itself evidence for the error
    # classes (the collector sets it from the injector).
    if any(step.get("error") for step in steps[tau:]):
        return "confirmed", "post-onset error flag"
    if not results:
        return "unverifiable", ("no tool results recorded after the onset - "
                                "this corpus predates structured telemetry")
    return "no_op", (f"{len(results)} post-onset result(s) recorded, none "
                     f"carries a {label} marker")


def _recorded_evidence(entry: dict, path: pathlib.Path) -> tuple[bool, str]:
    """(usable, why) for the collector's own record of what it injected.

    Some injections are history-layer - a rewritten task, corrupted stored
    results - and leave no marker in the step's own tool results, so scanning
    the trace cannot see them.  The v5 collector records what it actually did,
    and the record is bound to these exact bytes by `trace_sha256`, so a stale
    claim about different data cannot pass.  Without that binding the record is
    ignored and the trace has the last word.
    """
    facts = entry.get("injection") or {}
    digest = entry.get("trace_sha256")
    if not facts or not digest:
        return False, "no checksum-bound injection record"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        return False, "injection record does not match these trace bytes"
    if not facts.get("applied_count"):
        return False, "record says the injection never applied"
    return True, (f"collector recorded {facts['applied_count']} mutation(s) "
                  f"from step {facts.get('first_applied_t')} via "
                  f"{facts.get('applied_tools')}")


def _episode_evidence(entry: dict, steps: list[dict], label: str, tau: int,
                      path: pathlib.Path) -> tuple[str, str]:
    """Evidence verdict for one labelled episode, record first then trace."""
    ok, why = _recorded_evidence(entry, path)
    if ok:
        return "confirmed", why
    results = _post_onset_results(steps, tau)
    verdict, trace_why = _mutation_evidence(label, results, steps, tau)
    if verdict == "confirmed":
        return verdict, trace_why
    return verdict, f"{trace_why} ({why})"


def audit_corpus(corpus_dir: pathlib.Path) -> list[EpisodeAudit]:
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text("utf-8"))
    audits: list[EpisodeAudit] = []
    for entry in manifest:
        path = corpus_dir / entry.get("file", f"{entry['episode_id']}.jsonl")
        if not path.exists():
            audits.append(EpisodeAudit(corpus_dir.name, entry["episode_id"],
                                       entry.get("failure_class"),
                                       entry.get("tau"), 0,
                                       ["missing_trace_file"]))
            continue
        steps = [json.loads(line) for line in
                 path.read_text("utf-8").splitlines() if line.strip()]
        audit = EpisodeAudit(corpus_dir.name, entry["episode_id"],
                             entry.get("failure_class"), entry.get("tau"),
                             len(steps))

        if entry.get("T") is not None and int(entry["T"]) != len(steps):
            audit.defects.append("manifest_T_mismatch")
            audit.detail["manifest_T"] = entry["T"]

        has_lp = any(step.get("token_logprobs") for step in steps)
        if entry.get("has_logprobs") is not None and bool(entry["has_logprobs"]) != has_lp:
            audit.defects.append("logprob_flag_mismatch")

        label, tau = entry.get("failure_class"), entry.get("tau")
        if label:
            if tau is None:
                audit.defects.append("labelled_without_tau")
            elif tau >= len(steps) - 1:
                audit.defects.append("no_post_onset_steps")
            else:
                verdict, what = _episode_evidence(entry, steps, label, int(tau),
                                                  path)
                if verdict != "confirmed":
                    audit.defects.append(f"{verdict}_positive")
                    audit.detail["why"] = what
        else:
            if entry.get("success") is False:
                audit.defects.append("unsuccessful_healthy")

        if not any(step.get("tool_events") for step in steps):
            audit.defects.append("legacy_text_only_telemetry")
        if not any(step.get("task") for step in steps):
            audit.defects.append("no_task_recorded")
        audits.append(audit)
    return audits


def audit_all(corpora: list[str] | None = None) -> list[EpisodeAudit]:
    out: list[EpisodeAudit] = []
    for directory in sorted(TRACES.iterdir()):
        if not directory.is_dir() or directory.name.startswith("_"):
            continue
        if corpora and directory.name not in corpora:
            continue
        out.extend(audit_corpus(directory))
    return out


#: Defects that make an episode unusable as labelled; the rest are provenance
#: gaps that Phase 3 re-collection closes.
BLOCKING = {"no_op_positive", "unverifiable_positive", "unsuccessful_healthy",
            "no_post_onset_steps", "labelled_without_tau", "missing_trace_file",
            "manifest_T_mismatch"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append", help="restrict to a corpus")
    ap.add_argument("--verbose", action="store_true", help="list every episode")
    ap.add_argument("--json", type=pathlib.Path, help="write the full report")
    args = ap.parse_args(argv)

    audits = audit_all(args.corpus)
    by_corpus: dict[str, list[EpisodeAudit]] = {}
    for audit in audits:
        by_corpus.setdefault(audit.corpus, []).append(audit)

    print(f"{'corpus':24s} {'eps':>5} {'labelled':>8} {'no-op':>6} "
          f"{'unverif':>8} {'unsucc':>7} {'other':>6}")
    blocking_total = 0
    for corpus, items in sorted(by_corpus.items()):
        labelled = sum(1 for a in items if a.label)
        no_op = sum(1 for a in items if "no_op_positive" in a.defects)
        unver = sum(1 for a in items if "unverifiable_positive" in a.defects)
        unsucc = sum(1 for a in items if "unsuccessful_healthy" in a.defects)
        rest = BLOCKING - {"no_op_positive", "unverifiable_positive",
                           "unsuccessful_healthy"}
        other = sum(1 for a in items if set(a.defects) & rest)
        blocking_total += sum(1 for a in items if set(a.defects) & BLOCKING)
        print(f"{corpus:24s} {len(items):5d} {labelled:8d} {no_op:6d} "
              f"{unver:8d} {unsucc:7d} {other:6d}")

    if args.verbose:
        for audit in audits:
            if set(audit.defects) & BLOCKING:
                print(f"  {audit.corpus}/{audit.episode_id}: "
                      f"{sorted(set(audit.defects) & BLOCKING)} {audit.detail}")

    print(f"\n{blocking_total} episode(s) cannot be used as labelled.")
    if args.json:
        args.json.write_text(
            json.dumps([asdict(a) for a in audits], indent=1), "utf-8")
        print(f"full report -> {args.json}")
    return 1 if blocking_total else 0


if __name__ == "__main__":
    sys.exit(main())
