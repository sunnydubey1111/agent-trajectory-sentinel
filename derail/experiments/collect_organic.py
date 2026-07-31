"""T5 — collect ORGANIC (non-injected) episodes for validity checking.

Every labeled failure in the study so far comes from the tool-layer
injector with a known tau. This collector produces the missing evidence:
episodes run at Ollama's natural/high temperature (0.9 vs the curated
0.2), with NO injection — whatever fails, fails organically (the 0.2
default exists precisely because high temperature makes small models emit
junk-token bursts and leak raw tool-call syntax in a sizable fraction of
runs). Labels are assigned afterwards by human/manual review against a
documented rubric (organic_labels.csv), never by the monitors themselves.

Run:  py -m derail.experiments.collect_organic [--n 30]
Then: review traces/organic7b/*.jsonl -> organic_labels.csv
      py -m derail.experiments.score_organic
"""

from __future__ import annotations

import argparse

from derail.harness.collection import CorpusInUse, guard_output_dir
import json
from pathlib import Path

from derail.experiments.collect_traces import OllamaBackend
from derail.harness.agent_loop import run_real_episode
from derail.harness.collect_real import (RESEARCH_TASK_TOOLS, _TOPICS,
                                         _default_task)
from derail.harness.real_tools import _ensure_tls, build_registry
from derail.harness.record_replay import Cassette

TRACES_DIR = Path(__file__).resolve().parents[2] / "traces" / "organic7b"


class HotOllamaBackend(OllamaBackend):
    """OllamaBackend at a configurable temperature (default 0.9).

    The parent pins 0.2 to keep the HEALTHY baseline stable; here the
    point is the opposite — let the model fail naturally.
    """

    temperature = 0.9

    def _chat(self, want_logprobs: bool) -> dict:
        body = {"model": self.model, "messages": self.history,
                "tools": self._tools, "stream": False,
                "options": {"num_predict": 512,
                            "temperature": self.temperature}}
        if want_logprobs:
            body["logprobs"] = True
        r = self._httpx.post(f"{self.base}/api/chat", json=body,
                             timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()


def main(argv: list[str] | None = None) -> None:
    global TRACES_DIR
    parser = argparse.ArgumentParser(
        prog="py -m derail.experiments.collect_organic")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--out-dir", default=None,
                        help=f"corpus directory (default: {TRACES_DIR})")
    parser.add_argument("--allow-existing", action="store_true",
                        help="collect into a corpus that already holds episodes")
    args = parser.parse_args(argv)

    if args.out_dir:
        TRACES_DIR = Path(args.out_dir)
    try:
        guard_output_dir(TRACES_DIR, allow_existing=args.allow_existing)
    except CorpusInUse as exc:
        raise SystemExit(f"[organic] {exc}")

    _ensure_tls()
    # Capability allowlist for the shared research task.
    registry = build_registry(RESEARCH_TASK_TOOLS)
    # The cassette follows the corpus: a scratch run must not deposit
    # recordings into the shared committed cassette directory.
    cassette = Cassette(str(TRACES_DIR.parent / "_cassettes" / TRACES_DIR.name),
                        mode="auto")
    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in range(args.n):
        eid = f"organic-{i:03d}"
        path = TRACES_DIR / f"{eid}.jsonl"
        if path.exists():
            steps = [json.loads(x) for x in
                     path.read_text("utf-8").splitlines() if x]
            print(f"  [{eid}] resumed (T={len(steps)})")
        else:
            backend = HotOllamaBackend(args.model,
                                       tool_specs=registry.specs())
            backend.temperature = args.temperature
            # stride the topic index so tasks differ from the healthy set's
            # low-temperature runs over the shared topic list
            steps = run_real_episode(backend, registry,
                                     _default_task(7 * i + 3),
                                     max_steps=args.max_steps,
                                     cassette=cassette)
            path.write_text("\n".join(json.dumps(s) for s in steps), "utf-8")
            print(f"  [{eid}] T={len(steps)}")
        manifest.append({"episode_id": eid, "file": path.name,
                         "failure_class": None, "tau": None,
                         "T": len(steps),
                         "has_logprobs": any(bool(s.get("token_logprobs"))
                                             for s in steps),
                         "model": args.model,
                         "temperature": args.temperature})
    (TRACES_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), "utf-8")
    print(f"[organic] {len(manifest)} episodes -> {TRACES_DIR}; now review "
          "and fill organic_labels.csv (see score_organic)")


if __name__ == "__main__":
    main()
