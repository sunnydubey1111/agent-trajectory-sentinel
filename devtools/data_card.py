"""Generate the corpus data card from the manifests themselves.

`DATA_CARD.md` describes every trace corpus in the repository: how many
episodes, which model produced them, whether failures were injected at a known
onset or arose organically, episode lengths, and whether the uncertainty channel
is populated. Deriving it from `manifest.json` rather than writing it by hand
means the card cannot describe a corpus that is no longer there, or miss one
that was added.

    py -m devtools.data_card --write        # regenerate DATA_CARD.md
    py -m devtools.data_card --check        # fail if the file is out of date
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import statistics
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRACES = REPO_ROOT / "traces"
CARD_PATH = REPO_ROOT / "DATA_CARD.md"

#: One line of provenance per corpus: what it was collected for. Corpora are
#: read from disk, so a corpus missing from this table still appears in the
#: card -- with its purpose blank, which is the signal to describe it.
PURPOSE: dict[str, str] = {
    "autogen": "AutoGen loop, qwen2.5:3b -- small-model operating-envelope evidence",
    "autogen7b": "AutoGen loop at 7b -- the envelope finding confirmed causally",
    "demo7b": "Live-demo calibration corpus, superseded by the task-scoped rebuild",
    "demo7b_scoped": "Live-demo healthy null under the task-scoped toolset",
    "demo_real": "Demo agent, real-tool suite -- FIXED task shape (all T=7); "
                 "superseded as a healthy null by demo_real_varied",
    "demo_real_varied": "Demo-agent healthy null with VARIED task shape and "
                        "length (T=7-10); the fixed-shape corpus collapsed the "
                        "healthy spread and made the demo false-alarm",
    "real_research7b_long_drift": "Long-runway real goal_drift: the only corpus "
                                  "where drift has >=9 post-onset steps (24/24), "
                                  "collected to test the conceptor mechanism",
    "langgraph": "LangGraph StateGraph agent, qwen2.5:3b",
    "langgraph7b": "LangGraph StateGraph agent at 7b",
    "ollama": "Native loop on Ollama, qwen2.5:3b",
    "ollama7b": "Native loop on Ollama at 7b",
    "ollama_llama8b": "Cross-family transfer arm: llama3.1:8b on the 7b task plan",
    "organic7b": "First organic (non-injected) failure set, research task",
    "organic_demo7b": "Pre-registered organic hallucination study, temperature 0.9",
    "organic_demo7b_cold": "Serving-temperature arm (0.2), seed-paired with the 0.9 arm",
    "organic_demo7b_ext": "Extension of the organic set for fabrication base rate",
    "organic_demo7b_holdout": "Held-out corpus at disjoint task seeds (40000+)",
    "organic_demo7b_provoked": "Transient tool failures provoke fabrication into testable range",
    "organic_llama8b": "llama3.1:8b organic arm at temperature 0.9",
    "organic_llama8b_cold": "llama3.1:8b organic arm at the served temperature",
    "real": "First live Gemini corpus on the real-tool task suite",
    "real_gemini_long": "Lengthened Gemini corpus with real post-onset horizon",
    "real_ollama7b": "Real-tool research task on local qwen2.5:7b",
    "real_research3b": "Real-tool research task at 3b -- model-transfer arm",
    "real_research7b": "Primary real-tool research corpus",
    "real_research7b_long": "Lengthened research corpus for horizon analysis",
    "real_research7b_long_ext": "Extension of the lengthened research corpus",
}


def _corpora() -> list[tuple[str, list[dict]]]:
    out = []
    for manifest in sorted(TRACES.glob("*/manifest.json")):
        # A leading underscore marks a directory that is not one of our
        # corpora: scratch output, or a corpus imported from another project.
        # Counting those here would misstate what this repository collected.
        if manifest.parent.name.startswith("_"):
            continue
        entries = json.loads(manifest.read_text("utf-8"))
        if entries:
            out.append((manifest.parent.name, entries))
    return out


def _root_corpus_size() -> int:
    """Episodes in `traces/manifest.json`, which `_corpora()` cannot see.

    `_corpora()` globs `*/manifest.json`, so the top-level manifest is matched
    by nothing and its episodes appear in no total on this card. Counting them
    here keeps the omission stated rather than silent.
    """
    manifest = TRACES / "manifest.json"
    if not manifest.exists():
        return 0
    return len(json.loads(manifest.read_text("utf-8")))


#: Gemini corpora that DO sit in subdirectories, so this card can see them.
_GEMINI_SUBDIRS = ("real", "real_gemini_long")


def _gemini_in_card() -> int:
    """Gemini episodes inside this card's total."""
    total = 0
    for name in _GEMINI_SUBDIRS:
        manifest = TRACES / name / "manifest.json"
        if manifest.exists():
            total += len(json.loads(manifest.read_text("utf-8")))
    return total


def _gemini_breakdown() -> str:
    """`corpora \\`real\\` (18) and \\`real_gemini_long\\` (125)`, from the manifests."""
    parts = []
    for name in _GEMINI_SUBDIRS:
        manifest = TRACES / name / "manifest.json"
        if manifest.exists():
            n = len(json.loads(manifest.read_text("utf-8")))
            parts.append(f"`{name}` ({n})")
    return "corpora " + " and ".join(parts)


def _gemini_total() -> int:
    """Every Gemini episode in the repository, the scope of the notice.

    Computed rather than written down: the 330/143/187 split is exactly the
    kind of number that goes stale in prose while the manifests move on.
    """
    return _gemini_in_card() + _root_corpus_size()


def _rejections() -> tuple[list[tuple[str, int, int, float]], dict[str, dict[str, int]]]:
    """Per-corpus discard rates and the healthy/injected split of each rule.

    Reasons carry episode-specific numbers ("too short: T=3 < 4"), so digits
    are collapsed to N and any parenthetical detail dropped; otherwise every
    episode would look like its own rule.
    """
    per_corpus: list[tuple[str, int, int, float]] = []
    per_rule: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"healthy": 0, "injected": 0})
    for rejected in sorted(TRACES.glob("*/rejected.json")):
        if rejected.parent.name.startswith("_"):
            continue
        records = json.loads(rejected.read_text("utf-8"))
        manifest = rejected.parent / "manifest.json"
        accepted = len(json.loads(manifest.read_text("utf-8"))) if manifest.exists() else 0
        attempted = accepted + len(records)
        if not attempted:
            continue
        for record in records:
            reason = re.sub(r"\d+(\.\d+)?", "N", record["reason"]).split(" (")[0].strip()
            kind = "healthy" if record.get("requested_class") is None else "injected"
            per_rule[reason][kind] += 1
        per_corpus.append((rejected.parent.name, attempted, len(records),
                           100.0 * len(records) / attempted))
    per_corpus.sort(key=lambda row: row[3])
    return per_corpus, dict(per_rule)


def _summarise(entries: list[dict]) -> dict[str, object]:
    lengths = [e.get("T") for e in entries if isinstance(e.get("T"), int)]
    classes = collections.Counter(e.get("failure_class") for e in entries)
    injected = sum(n for cls, n in classes.items() if cls)
    temps = {e["temperature"] for e in entries if e.get("temperature") is not None}
    return {
        "n": len(entries),
        "models": sorted({str(e.get("model", "?")) for e in entries}),
        "injected": injected,
        "healthy": classes.get(None, 0),
        "classes": sorted(c for c in classes if c),
        "median_T": int(statistics.median(lengths)) if lengths else 0,
        "min_T": min(lengths) if lengths else 0,
        "max_T": max(lengths) if lengths else 0,
        "logprobs": sum(1 for e in entries if e.get("has_logprobs")),
        "temperatures": sorted(temps),
        "labelled_tau": sum(1 for e in entries if e.get("tau") is not None),
    }


def render() -> str:
    corpora = _corpora()
    rejections, rule_split = _rejections()
    rej_attempted = sum(row[1] for row in rejections)
    rej_total = sum(row[2] for row in rejections)
    total = sum(len(e) for _, e in corpora)
    real_tool = sum(len(e) for name, e in corpora if name.startswith("real"))
    models = collections.Counter()
    for _, entries in corpora:
        for e in entries:
            models[str(e.get("model", "?"))] += 1

    lines = [
        "# Data card",
        "",
        "Generated by `py -m devtools.data_card --write` from the corpus",
        "manifests, so it cannot drift from what is committed.",
        "",
        "## What is here",
        "",
        f"- **{total:,} agent episodes** across **{len(corpora)} corpora**, every",
        "  trace committed as JSONL under `traces/`.",
        f"- **{real_tool:,}** of those episodes use *real* tools (arXiv, Wikipedia,",
        "  web fetch, SQL, Python); the rest use a deterministic mock-tool suite.",
        "- Episodes per agent model:",
        "",
        "| model | episodes |",
        "|---|---:|",
    ]
    for model, n in models.most_common():
        lines.append(f"| `{model}` | {n:,} |")

    lines += [
        "",
        "## How episodes were produced",
        "",
        "Two collection regimes, and the distinction matters for every claim made",
        "from them.",
        "",
        "**Injected.** A tool-layer injector fires at a known step tau, so ground",
        "truth is exact. The failure that follows is real model behaviour; only",
        "the trigger is controlled. Classes: `goal_drift` (the task text is",
        "silently rewritten mid-run), `looping`, `tool_cascade`,",
        "`context_corruption` (earlier tool results are garbled),",
        "`grounding_loss`, plus `rate_limit`, `timeout`, `malformed_json` and",
        "`wrong_document` on the research collectors.",
        "",
        "**What an injected label asserts, exactly.** It records FAULT",
        "EXPOSURE, not confirmed task failure: the acceptance gate requires",
        "that the injector really mutated a result and that at least one step",
        "followed, so there is something to detect. Whether the agent then",
        "went wrong is a separate question, and one these labels do not",
        "answer -- an agent that noticed the fault and recovered carries the",
        "same label as one that derailed. Detection rates on injected corpora",
        "are therefore rates of *detecting an exposed fault*, and a claim that",
        "reads them as \"caught a failed task\" claims more than the label",
        "supports. The organic corpora below carry that stronger claim,",
        "because their labels are graded against a computable ground truth.",
        "",
        "**Organic.** Nothing is injected. Episodes are labelled after the fact,",
        "objectively and by script, from each run's own tool results against a",
        "computable ground truth -- `healthy`, `arithmetic_error`, `hallucinated`,",
        "`incomplete`, `other`. These are the sets that carry the deployment",
        "claims, because they contain the failures the model produces on its own.",
        "",
        "## Corpora",
        "",
        "`T` is episode length in steps. `tau` counts episodes with a ground-truth",
        "onset. `u` counts episodes carrying token logprobs, which populate the",
        "uncertainty channel.",
        "",
        "| corpus | n | model(s) | healthy | injected | tau | u | T (min/med/max) | purpose |",
        "|---|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for name, entries in corpora:
        s = _summarise(entries)
        models_txt = ", ".join(f"`{m}`" for m in s["models"])
        lines.append(
            f"| `{name}` | {s['n']} | {models_txt} | {s['healthy']} | "
            f"{s['injected']} | {s['labelled_tau']} | {s['logprobs']} | "
            f"{s['min_T']}/{s['median_T']}/{s['max_T']} | "
            f"{PURPOSE.get(name, '')} |")

    lines += [
        "",
        "## Collection settings",
        "",
        "- **Local models** are served by Ollama. The demo and the serving-arm",
        "  corpora sample at temperature 0.2; the failure-provoking organic arms",
        "  sample at 0.9. A monitor calibrated at one temperature does not",
        "  transfer to the other, which is why the arms are seed-paired.",
        "- **Gemini corpora** use `gemini-2.5-flash`. The free tier gates",
        "  `response_logprobs`, so those corpora carry the `e+m` channels only and",
        "  the collector records that rather than fabricating a `u` channel.",
        "- **Task seeds** are recorded per episode. The held-out corpus uses seeds",
        "  40000+, disjoint from every corpus the checks were designed against.",
        "- **Rejections are recorded, never silent.** An episode that fails the",
        "  acceptance gate is written to the corpus's `rejected.json` with its",
        "  reason, so the acceptance rate stays visible rather than being hidden",
        "  behind a larger N.",
        "",
        "## What the acceptance gate discarded",
        "",
        "Twelve corpora record rejections. The discard rate is not uniform and",
        "the reader should not assume it is:",
        "",
        "| corpus | attempted | rejected | discard |",
        "|---|---|---|---|",
        *[f"| `{name}` | {att} | {rej} | {pct:.1f}% |"
          for name, att, rej, pct in rejections],
        f"| **total** | **{rej_attempted}** | **{rej_total}** | "
        f"**{100.0 * rej_total / rej_attempted:.1f}%** |",
        "",
        f"The range is {rejections[0][3]:.1f}% (`{rejections[0][0]}`) to "
        f"{rejections[-1][3]:.1f}% (`{rejections[-1][0]}`), so a per-corpus N in",
        "the table above is not a fixed fraction of what was attempted.",
        "",
        "The rules are also asymmetric, which matters more than the rate. Only",
        "the length rule can reject a healthy episode; every other rule fires",
        "solely on injected ones, because it checks that the injection actually",
        "landed. Positives are therefore filtered on a condition negatives are",
        "never tested against:",
        "",
        "| rejection rule | healthy | injected |",
        "|---|---|---|",
        *[f"| {reason} | {counts['healthy']} | {counts['injected']} |"
          for reason, counts in sorted(
              rule_split.items(),
              key=lambda kv: -(kv[1]["healthy"] + kv[1]["injected"]))],
        "",
        "An injected episode whose mutation never applied is not a failure",
        "episode, so keeping it would mislabel the data; dropping it still",
        "means the surviving positives are the ones injection succeeded on.",
        "",
        "## One corpus this card does not count",
        "",
        f"**Gemini episodes, once: {_root_corpus_size() + 143} collected, 143 "
        "exported in this dataset, and the published channel-max AUC of 0.840 "
        f"computed on the other {_root_corpus_size()}.**",
        "",
        "`traces/manifest.json` — the top level, not a subdirectory — lists",
        f"**{_root_corpus_size()} `gemini-2.5-flash` episodes** that are",
        "committed in the repository but appear in none of the totals above,",
        "and are not part of this dataset. Every count on",
        "this page enumerates corpora by globbing `traces/*/manifest.json`,",
        "which matches subdirectories only, so this set is invisible to the",
        "card, to the claims ledger and to the Hugging Face export.",
        "",
        "It is not abandoned data: it is the corpus behind the published",
        "channel-max AUC of 0.84 on 187 live Gemini episodes. It is recorded",
        "here so that a reader who counts the `.jsonl` files does not find more",
        "episodes than the card admits to, and so that the Gemini terms in",
        "`traces/NOTICE_gemini.md` are understood to cover it. The totals are",
        "left as they are rather than restated, because every published number",
        "was computed against the corpora the glob finds.",
        "",
        "## Provenance and integrity",
        "",
        "Every committed file is hashed in `BASELINE_MANIFEST.json`",
        "(`py -m devtools.artifact_manifest --check`), so an accidental edit to a",
        "trace cannot pass unnoticed. Corpora with a `trace_sha256` field also",
        "carry a per-episode hash in the manifest itself.",
        "",
        # These runs read a real workspace, and that workspace was this
        # repository mid-development. The recordings are published unedited,
        # so they hold file contents and listings that do not match the
        # released tree. Readers are told that here rather than left to
        # discover it as an inconsistency.
        "**What the `workspace_file_qa` runs recorded.** The",
        "`workspace_file_qa` runs in `traces/real` recorded an agent reading",
        "files in a real software workspace — this project's own repository at",
        "an earlier revision. The recorded file contents and directory listings",
        "are therefore development-time snapshots and may differ from the",
        "published repository. They are published unedited, as recorded: a",
        "trace is a record of what the agent actually saw, so it is not",
        "rewritten to match the current tree.",
        "",
        "## Licensing",
        "",
        "Apache-2.0 covers what this project wrote: the source, the trace format and",
        "schema, the result tables, and this card. It cannot cover what the",
        "project only recorded or called, because a licence cannot grant",
        "rights the licensor never held. Recorded model output carries the",
        "terms of the model that produced it:",
        "",
        "- **`qwen2.5:7b`, `qwen2.5:3b`** (2,247 episodes) are Apache-2.0, which",
        "  places no condition on redistributing their output.",
        "- **`llama3.1:8b`** (433 episodes) is under the Llama 3.1 Community",
        "  License. This corpus is **Built with Llama**; the licence is at",
        "  <https://llama.meta.com/llama3_1/license/> and the Acceptable Use",
        "  Policy it carries forward is at",
        "  <https://llama.meta.com/llama3_1/use-policy/>. Reuse of these",
        "  episodes is bound by both.",
        "- **`gemini-2.5-flash`** was called through the Gemini API on the",
        "  unpaid tier. Google's terms bar using the Services to develop",
        "  competing models, and a condition on reuse follows from that. The",
        "  clause is quoted and the condition stated in",
        "  `traces/NOTICE_gemini.md`, which anyone reusing this output should",
        "  read first.",
        "",
        f"  That condition covers **{_gemini_total()} episodes**, of which only",
        f"  **{_gemini_in_card()}** appear in this card: {_gemini_breakdown()}.",
        "  The other",
        f"  **{_root_corpus_size()}** are listed in the top-level",
        "  `traces/manifest.json` rather than in a corpus subdirectory, and this",
        "  card enumerates corpora by globbing `traces/*/manifest.json`, so they",
        "  fall outside its total above. They are committed all the same and the",
        "  notice covers them; the three counts are reconciled in a table there.",
        "",
        "External benchmark corpora (AFTraj-2K, ATBench) are **not**",
        "redistributed here. They download into gitignored directories and are",
        "never committed; only our measurements of them are, and those are",
        "Apache-2.0.",
        "Anyone importing them is bound by the terms at the source.",
        "",
        "Cassette-replayed tool results come from public services and keep",
        "their own terms. Most are uncopyrightable facts - arXiv titles and",
        "identifiers, repository names and star counts. Two carry a licence:",
        "",
        "- Current-weather records are from **Open-Meteo**",
        "  (<https://open-meteo.com>), CC BY 4.0.",
        "- Search snippets are from **Wikipedia**, CC BY-SA 4.0, and are stored",
        "  as the short excerpts the search API returns, not article text.",
        "",
        "## Intended use and limits",
        "",
        "These traces exist to evaluate monitors, not to train agents. Three",
        "limits bear directly on what can be concluded from them:",
        "",
        "1. **Task diversity is narrow.** A mock booking task and a real-tool",
        "   research task. Per-class detection numbers are conditional on these.",
        "2. **Injected onsets are not organic onsets.** Injected classes give",
        "   exact tau and are what most detection numbers are measured on; the",
        "   organic sets are smaller and their failure mix is whatever the model",
        "   actually produced.",
        "3. **No personal data.** Tasks are synthetic or public-source (arXiv,",
        "   Wikipedia, public repositories). Tool results from live services are",
        "   recorded as cassettes for deterministic replay.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="py -m devtools.data_card")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if not (args.write or args.check):
        parser.error("pass --write or --check")

    card = render()
    if args.check:
        current = CARD_PATH.read_text("utf-8") if CARD_PATH.exists() else ""
        if current.replace("\r\n", "\n") != card:
            print("DATA_CARD.md is out of date; run "
                  "`py -m devtools.data_card --write`", file=sys.stderr)
            return 1
        print("DATA_CARD.md matches the committed corpora")
    if args.write:
        CARD_PATH.write_text(card, encoding="utf-8", newline="\n")
        print(f"wrote {CARD_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
