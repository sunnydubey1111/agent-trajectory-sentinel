"""Assemble the Hugging Face Space: a replay of real runs and real scores.

Constraints, in the order they bind:

1. The live demo needs a local Ollama and a 7B model. A Space cannot have one.
2. Hugging Face charges for gradio and Docker Spaces even on free CPU; only
   STATIC Spaces are free. So the Space cannot run Python at all.

What survives both is this: the monitor is fitted and run *here*, over real
committed episodes, and the resulting per-step scores are shipped as data to a
single self-contained page that replays them. The scores are the real monitor's
output, computed by `build()` from the same corpus and the same fitting
protocol the study uses - but they are computed ahead of time, and the page
says so rather than implying a live model.

    py -m devtools.hf_space --build              # write build/space/ and stop
    py -m devtools.hf_space --build --push       # ... and upload
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import sys

import numpy as np

from derail.common import Standardizer, rng_for
from derail.evaluation.metrics import pick_threshold
from derail.monitor.esn import ChannelMaxESNMonitor
from derail.telemetry.adapter import load_trace_jsonl

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build" / "space"
DEFAULT_REPO_ID = "sunnydubey1111/agent-trajectory-sentinel-demo"

#: Real-tool research corpus: 120 healthy runs for the null and 171 injected
#: across eight failure classes, so every run offered is a real recording.
CORPUS = "real_research7b"
PER_CLASS = 2          # injected episodes offered per failure class
N_HEALTHY_SHOWN = 6    # healthy runs offered, drawn from the held-out split
FA_BUDGET = 0.05
CHANNELS = ("e", "m")  # this corpus records no usable logprobs


#: What the agent was asked to do. This corpus predates the per-step `task`
#: field, so the instruction itself is not recoverable from the traces; this is
#: the task definition the collector used, recorded here so the page can say
#: what the runs are instead of showing scores with no context.
TASK_DESCRIPTION = (
    "Research a topic and report what you find. The agent has four real tools "
    "— arXiv search, Wikipedia, web search and a Python interpreter — and must "
    "gather evidence with them before answering."
)

_QUERY = re.compile(r'"query"\s*:\s*"([^"]{3,120})"')


def _topic(steps: list[dict]) -> str | None:
    """The run's subject, taken from the first query it issues.

    Derived, not recorded: this corpus stores no task prompt, so the page
    labels it as inferred rather than presenting it as the instruction given.
    """
    for step in steps:
        match = _QUERY.search(str(step.get("text", "")))
        if match:
            return match.group(1)
    return None


def _load(corpus_dir: pathlib.Path, entry: dict):
    return load_trace_jsonl(corpus_dir / entry["file"],
                            episode_id=entry["episode_id"], tau=entry["tau"],
                            failure_class=entry["failure_class"],
                            extended=True)


def _select(manifest: list[dict]) -> list[dict]:
    """Two injected runs per class, longest post-onset horizon first.

    Longest first because a behavioural monitor needs steps after the onset to
    accumulate evidence; leading with the shortest would misrepresent what the
    method claims rather than demonstrate it.
    """
    chosen: list[dict] = []
    seen: dict[str, int] = {}
    for entry in sorted(manifest,
                        key=lambda e: -(e["T"] - 1 - (e["tau"] or 0))):
        cls = entry["failure_class"]
        if entry["tau"] is None or seen.get(cls, 0) >= PER_CLASS:
            continue
        seen[cls] = seen.get(cls, 0) + 1
        chosen.append(entry)
    return chosen


def compute(corpus_dir: pathlib.Path) -> dict:
    """Fit the monitor and score the runs the page will replay."""
    manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
    healthy_entries = [e for e in manifest if e["tau"] is None]
    healthy = [_load(corpus_dir, e) for e in healthy_entries]

    # Same discipline as the study: the alarm line is chosen on healthy runs
    # the monitor was not fitted to.
    perm = rng_for(0, "space-split").permutation(len(healthy))
    split = int(round(0.75 * len(healthy)))
    train = [healthy[i] for i in perm[:split]]
    val = [healthy[i] for i in perm[split:]]

    monitor = ChannelMaxESNMonitor(Standardizer().fit(train),
                                   channels=CHANNELS, K=8, cusum=True,
                                   seed=1300)
    monitor.fit(train)
    theta = float(pick_threshold([monitor.score_episode(e) for e in val],
                                 FA_BUDGET))

    val_ids = {e.episode_id for e in val}
    shown = [e for e in healthy_entries
             if e["episode_id"] in val_ids][:N_HEALTHY_SHOWN]
    shown += _select(manifest)

    runs = []
    for entry in shown:
        episode = _load(corpus_dir, entry)
        scores = np.asarray(monitor.score_episode(episode), dtype=float)
        steps = []
        for line in (corpus_dir / entry["file"]).read_text("utf-8").splitlines():
            if not line.strip():
                continue
            step = json.loads(line)
            text = str(step.get("text", "")).strip() or "(no text)"
            steps.append({"action": str(step.get("action", "?")),
                          "text": text[:700]})
        fired = np.where(scores > theta)[0]
        runs.append({
            "id": entry["episode_id"],
            "cls": entry["failure_class"],
            "topic": _topic(steps),
            "tau": entry["tau"],
            "scores": [round(float(s), 4) for s in scores],
            "alarm": int(fired[0]) if fired.size else None,
            "steps": steps,
        })

    return {"theta": round(theta, 4), "corpus": CORPUS,
            "task": TASK_DESCRIPTION, "model": "qwen2.5:7b",
            "n_train": len(train), "n_val": len(val),
            "fa_budget": FA_BUDGET, "runs": runs}


def build(out_dir: pathlib.Path, repo_id: str = DEFAULT_REPO_ID) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    data = compute(REPO_ROOT / "traces" / CORPUS)
    (out_dir / "data.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8", newline="\n")
    page = (pathlib.Path(__file__).parent / "space_page.html").read_text("utf-8")
    (out_dir / "index.html").write_text(page, encoding="utf-8", newline="\n")
    (out_dir / "README.md").write_text(_card(repo_id, data),
                                       encoding="utf-8", newline="\n")

    caught = sum(1 for r in data["runs"]
                 if r["tau"] is not None and r["alarm"] is not None)
    failing = sum(1 for r in data["runs"] if r["tau"] is not None)
    false_alarms = sum(1 for r in data["runs"]
                       if r["tau"] is None and r["alarm"] is not None)
    return {"runs": len(data["runs"]), "failing": failing, "caught": caught,
            "false_alarms": false_alarms, "theta": data["theta"]}


def _card(repo_id: str, data: dict) -> str:
    owner = repo_id.split("/")[0]
    failing = [r for r in data["runs"] if r["tau"] is not None]
    caught = sum(1 for r in failing if r["alarm"] is not None)
    return "\n".join([
        "---",
        "title: AgentTrajectorySentinel Live",
        "emoji: 📉",
        "colorFrom: indigo",
        "colorTo: red",
        "sdk: static",
        "app_file: index.html",
        "pinned: false",
        "license: apache-2.0",
        # The hub caps this at 60 characters and rejects the upload otherwise.
        "short_description: Watch a monitor catch an LLM agent derailing",
        "---",
        "",
        "# AgentTrajectorySentinel Live",
        "",
        "A one-class behavioural monitor scoring real agent runs, one step at",
        "a time. Pick a run, step through it, watch the score move.",
        "",
        "## What is real here, and what is not",
        "",
        "**Real:** the runs are committed recordings of a live qwen2.5:7b agent",
        "on real tools, with a known failure-onset step. The scores are this",
        f"project's echo-state-network monitor, fitted on {data['n_train']}",
        "healthy runs from the same corpus. The alarm line",
        f"({data['theta']}) is the threshold that spends a "
        f"{int(data['fa_budget'] * 100)}% false-alarm budget on",
        f"{data['n_val']} *held-out* healthy runs — not a number picked to make",
        "the demo look good. The monitor never sees the failure label.",
        "",
        "**Not real:** the scoring happened when this page was built, not when",
        "you clicked. Hugging Face charges for Spaces that run Python, so this",
        "one is static. Every number came out of the code in the repository",
        "below and can be recomputed with `py -m devtools.hf_space --build`.",
        "",
        "## What to look for",
        "",
        f"- Of the {len(failing)} failing runs shipped here, **{caught} are",
        f"  caught and {len(failing) - caught} are missed**. The misses are",
        "  included on purpose: a demo showing only successes is an",
        "  advertisement.",
        "- On a healthy run the score should stay flat and under the line.",
        "- Short runs are hard. This monitor needs a few steps after the onset",
        "  to accumulate evidence, which is measured in the paper and visible",
        "  here.",
        "",
        "Paper: <https://arxiv.org/abs/2608.02464>",
        "",
        f"Code and data: <https://github.com/{owner}/"
        "agent-trajectory-sentinel>",
        "",
    ])


def push(out_dir: pathlib.Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="space", space_sdk="static",
                    private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=repo_id,
                      repo_type="space",
                      commit_message="Publish the monitor replay")
    print(f"[space] pushed to https://huggingface.co/spaces/{repo_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="py -m devtools.hf_space")
    parser.add_argument("--build", action="store_true", required=True)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--out", default=str(BUILD_DIR))
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.out)
    summary = build(out_dir, args.repo_id)
    print(f"[space] {summary['runs']} runs, alarm line {summary['theta']}: "
          f"{summary['caught']}/{summary['failing']} failures caught, "
          f"{summary['false_alarms']} false alarm(s)")
    print(f"[space] built {out_dir}")
    if args.push:
        push(out_dir, args.repo_id, args.private)
    else:
        print("[space] not pushed (pass --push)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
