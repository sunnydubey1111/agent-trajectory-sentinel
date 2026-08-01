"""Import AFTraj-2K into the trace-corpus format the monitors already read.

AFTraj-2K (Zhang et al., arXiv:2605.08715, CC-BY-4.0) is an external corpus of
multi-agent trajectories with an annotated earliest-decisive-error step. It is
the closest published benchmark to this project's question, so running against
it is the only way to say anything about how these monitors compare with an
LLM auditor rather than with our own baselines.

Nothing about the monitors changes. `derail/telemetry/adapter.py` already
states that external validation is an adapter problem, and this module is that
adapter: it converts each trajectory into the same per-step records
`load_trace_jsonl` consumes, writes a `manifest.json` beside them, and stops.
Evaluation is then `run_hybrid_study --datasets aftraj`, unmodified.

    py -m derail.experiments.import_aftraj            # download and convert
    py -m derail.experiments.import_aftraj --from DIR # convert local parquet

Output goes to `traces/_aftraj/`. The leading underscore marks it as an
imported corpus rather than one of ours: it is not committed, not counted in
DATA_CARD.md, and not hashed into BASELINE_MANIFEST.json, because it is
someone else's data and folding it into our totals would misstate both.

MAPPING, and what it costs
--------------------------
A trajectory is a list of turns with `role`, `content`, `action`, `thought`.
Turns whose role is `user` (the task) or `environment` (tool results) are not
agent steps; every other turn is. An environment turn is folded into the
preceding agent step as that step's tool results, matched by call id, which is
how a step and its results already travel together in our own traces.

`mistake_step` indexes the turn list. It usually lands on an agent turn, and
then tau is that turn's position among agent steps. In 34 of the 1,114 unsafe
rows it lands on an `environment` turn instead - the decisive error is a tool
result, not a model utterance - and tau is then the step that issued the call,
because that is the step whose record carries the result. Getting this wrong
would silently shift every lead-time number, so the turn-to-step owner map is
built explicitly and a tau that cannot be placed raises rather than defaults.

Two channels are unavailable and are recorded as unavailable rather than
faked:

  u — AFTraj carries no token logprobs, so every step sets
      `logprobs_available: false` and the surprisal dims take MISSING_SURPRISAL.
      This is the same e+m-only path the Gemini corpora already run on.
  m — no per-step timings, so latency is left at its default and that dim is
      constant across the corpus. A constant dim is degenerate, not
      informative; the standardizer's degenerate-dim handling covers it.

The honest reading is that this corpus exercises the embedding and action
channels and cannot exercise the uncertainty channel at all.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

TRACES_DIR = pathlib.Path(__file__).resolve().parents[2] / "traces"
CORPUS_DIR = TRACES_DIR / "_aftraj"
HF_DATASET = "ZBox008003/AFTraj"
SPLITS = ("safe", "unsafe")

#: Roles that are not agent steps: the task statement and the tool channel.
NON_AGENT_ROLES = {"user", "environment"}
MIN_STEPS = 4          # matches run_hybrid_study.MIN_T


def _parquet_urls(split: str) -> list[str]:
    api = (f"https://huggingface.co/api/datasets/{HF_DATASET}"
           f"/parquet/default/{split}")
    with urllib.request.urlopen(api, timeout=120) as response:
        return json.loads(response.read())


def _download(dest: pathlib.Path) -> pathlib.Path:
    """Fetch both splits as parquet into `dest`, skipping what is present."""
    dest.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for i, url in enumerate(_parquet_urls(split)):
            path = dest / f"{split}-{i}.parquet"
            if not path.exists():
                print(f"[aftraj] downloading {split} -> {path.name}")
                urllib.request.urlretrieve(url, path)
    return dest


def _load_split(source: pathlib.Path, split: str) -> list[dict]:
    import pyarrow.parquet as pq

    files = sorted(source.glob(f"{split}-*.parquet"))
    if not files:
        raise SystemExit(f"no {split}-*.parquet under {source}")
    rows: list[dict] = []
    for path in files:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def _calls(turn: dict) -> list[dict]:
    """Tool calls issued by one agent turn, or [] when it issued none."""
    raw = (turn.get("action") or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [c for c in parsed if isinstance(c, dict)] if isinstance(parsed, list) else []


def _results(turn: dict | None) -> dict[str, dict]:
    """Tool results from an environment turn, keyed by call id."""
    if turn is None or turn.get("role") != "environment":
        return {}
    try:
        parsed = json.loads((turn.get("content") or "").strip() or "[]")
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, list):
        return {}
    return {str(r.get("call_id", "")): r for r in parsed if isinstance(r, dict)}


def _is_error(result: str) -> bool:
    head = result.strip()[:40].lower()
    return head.startswith(("error", "exception", "traceback", "failed"))


def _convert(row: dict) -> tuple[list[dict], int | None]:
    """One trajectory -> (step records, tau).

    tau is None for a safe trajectory. For an unsafe one it is the position,
    among agent steps, of the turn `mistake_step` points at.
    """
    turns = row["turns"]
    mistake = row.get("mistake_step")
    steps: list[dict] = []
    # turn index -> the step that owns it. An agent turn owns itself; an
    # environment turn belongs to the step that issued the call, since that
    # step's record is what carries the result.
    owner: dict[int, int] = {}

    for index, turn in enumerate(turns):
        if turn.get("role") in NON_AGENT_ROLES:
            if steps:
                owner[index] = len(steps) - 1
            continue
        owner[index] = len(steps)
        calls = _calls(turn)
        results = _results(turns[index + 1] if index + 1 < len(turns) else None)

        events = []
        errored = False
        for call in calls:
            call_id = str(call.get("id", ""))
            result = str(results.get(call_id, {}).get("result", ""))
            is_error = _is_error(result)
            errored = errored or is_error
            args = call.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            events.append({"id": call_id, "name": str(call.get("name", "tool")),
                           "args": args if isinstance(args, dict) else {},
                           "result": result, "result_chars": len(result),
                           "result_truncated": False, "is_error": is_error,
                           "latency_s": None})

        # The model's own words. Tool calls reach the features through
        # tool_events, which parse_step_events prefers over anything the text
        # could claim, so they are deliberately not re-rendered into the text.
        text = " ".join(p for p in (turn.get("thought") or "",
                                    turn.get("content") or "") if p.strip())
        steps.append({
            "text": text,
            "action": "tool_call" if events else "synthesis",
            "output_tokens": max(len(text.split()), 1),
            "error": errored,
            "logprobs_available": False,
            "tool_events": events,
        })

    if mistake is None:
        return steps, None
    tau = owner.get(mistake)
    if tau is None:
        raise ValueError(
            f"{row['conv_id']}: mistake_step {mistake} (role "
            f"{turns[mistake]['role']!r}) precedes every agent step; tau "
            f"cannot be placed")
    return steps, tau


def convert(source: pathlib.Path, out_dir: pathlib.Path) -> dict:
    """Write the corpus and its manifest; return a summary of what was kept."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    kept = {"safe": 0, "unsafe": 0}
    dropped = {"safe": 0, "unsafe": 0}

    for split in SPLITS:
        for row in _load_split(source, split):
            steps, tau = _convert(row)
            # Episode requires 0 < tau < T: an onset at the first step leaves
            # no pre-onset history to alarm from, and an onset past the end is
            # not observable at all.
            if len(steps) < MIN_STEPS or (tau is not None
                                          and not 0 < tau < len(steps)):
                dropped[split] += 1
                continue
            episode_id = str(row["conv_id"])
            name = f"{episode_id}.jsonl"
            (out_dir / name).write_text(
                "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in steps),
                encoding="utf-8")
            manifest.append({
                "episode_id": episode_id,
                "file": name,
                # The failure class is AFTraj's own provenance split: an
                # injected corruption is a different object from a failure
                # their judges diagnosed in an uncorrupted run, and pooling
                # them would hide which kind the monitor catches.
                # `external` records that the mechanism is AFTraj's to name,
                # not ours: their failures do not map onto this project's
                # taxonomy and forcing them into one of its classes would
                # assert a mechanism nobody measured. `source` below is their
                # own label, and per-class numbers are grouped on that.
                "failure_class": None if tau is None else "external",
                "source": None if tau is None else str(row["unsafe_source"]),
                "tau": tau,
                "T": len(steps),
                "has_logprobs": False,
                "model": "aftraj-2k",
                "domain": str(row["domain"]),
            })
            kept[split] += 1

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return {"kept": kept, "dropped": dropped, "episodes": len(manifest)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="py -m derail.experiments.import_aftraj",
        description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="source", default=None,
                        help="directory holding {safe,unsafe}-*.parquet; "
                             "downloaded from Hugging Face when omitted")
    parser.add_argument("--out", default=str(CORPUS_DIR))
    args = parser.parse_args(argv)

    source = pathlib.Path(args.source) if args.source else _download(
        CORPUS_DIR / "_parquet")
    summary = convert(source, pathlib.Path(args.out))
    print(f"[aftraj] kept {summary['kept']['safe']} safe + "
          f"{summary['kept']['unsafe']} unsafe = {summary['episodes']} episodes")
    print(f"[aftraj] dropped {summary['dropped']['safe']} safe + "
          f"{summary['dropped']['unsafe']} unsafe (fewer than {MIN_STEPS} "
          f"agent steps)")
    print(f"[aftraj] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
