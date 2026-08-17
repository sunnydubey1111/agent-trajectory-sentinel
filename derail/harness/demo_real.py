"""Real-tools demo engine — the mock booking demo, rebuilt on real tools.

Same idea as experiments/demo.py (a live agent scored by the ESN channel-max
monitor, with failure-injection buttons), but the agent drives REAL tools
(arXiv / Wikipedia / web / python) on a fixed research task, injection uses
the WS4 ToolInjector, and everything runs through the cassette so a stage run
is REAL yet deterministic and free (Ollama, live u channel).

This module is the engine: collect a healthy calibration set for the fixed
demo task, cross-fit the monitor + threshold on it, and run/score one healthy
or injected episode. The web UI is wired on top of this (next increment); the
headless `--demo` mode proves detection works on real-tool injections first.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from derail.common import (
    IDX_LATENCY_LOG,
    IDX_TOOL_LATENCY,
    Standardizer,
    rng_for,
)
from derail.evaluation.metrics import pick_threshold
from derail.experiments.demo import StreamingChannelMax
from derail.monitor.esn import _WASHOUT
from derail.harness.agent_loop import run_real_episode
from derail.harness.inject import ToolInjector
from derail.harness.record_replay import Cassette
from derail.telemetry.adapter import episode_from_trace, load_trace_jsonl
from derail.verify.checks import RESEARCH_SPEC, required_coverage

TRACES = Path(__file__).resolve().parents[2] / "traces" / "demo_real_varied"
#: The original fixed-shape corpus. Retained as a historical record; it is NOT
#: a valid null for what this demo now serves, because every one of its 48
#: episodes runs the identical 6-call structure (see `demo_task`).
LEGACY_TRACES = Path(__file__).resolve().parents[2] / "traces" / "demo_real"
MODEL = "qwen2.5:7b"
FA_BUDGET = 0.10

# The TOPIC varies per episode, and so does the SHAPE.
#
# Varying only the topic was not enough. A null of 48 episodes that all run the
# same 6 calls in the same order has almost no spread in step count, action
# pattern or per-step cost, so the healthy residual stds collapse and any live
# run lands far outside it -- healthy runs scored ~10^4 against a threshold of
# 191 and the demo false-alarmed on clean episodes. Diversity in the null is
# not cosmetic here; it is what makes the threshold mean anything.
#
# So each episode now draws a topic AND a shape: the order of the required
# calls is permuted, and zero to three OPTIONAL extra searches are appended,
# giving episode lengths of roughly 7-11 steps instead of a constant 7.
# Every variant still satisfies RESEARCH_SPEC (arxiv x2, wikipedia x2,
# web x1, python x1), so the task-completeness policy stays checkable and the
# healthy/injected labels keep their meaning.
_DEMO_TOPICS = (
    ("echo state networks", "transformers", "time-series anomaly detection"),
    ("isolation forests", "autoencoders", "network intrusion detection"),
    ("graph neural networks", "LSTMs", "financial fraud detection"),
    ("one-class SVM", "variational autoencoders", "sensor fault detection"),
    ("Gaussian processes", "convolutional networks", "ECG anomaly detection"),
    ("hidden Markov models", "diffusion models", "industrial process monitoring"),
)

#: Optional extra steps, appended in varying number. They add length and
#: action-pattern spread without touching the required-call contract.
_OPTIONAL_STEPS = (
    "web_search for a benchmark dataset used in {app}",
    "arxiv_search for a survey of {app}",
    "wikipedia_search for '{app}'",
)


# Exactly the tools demo_task names below.
DEMO_TASK_TOOLS = ("arxiv_search", "wikipedia_search", "web_search", "python")


def demo_task(seed: int) -> str:
    """Research task for episode `seed`: varied topic AND varied shape.

    Deterministic in `seed`, so healthy episode k and injected episode k still
    run the identical task and remain counterfactual pairs.
    """
    a, b, app = _DEMO_TOPICS[seed % len(_DEMO_TOPICS)]
    rng = rng_for(seed, "demo-real-task")

    required = [
        f"arxiv_search for recent {a} papers",
        f"arxiv_search for {b} papers",
        f"wikipedia_search for '{a}'",
        f"wikipedia_search for '{b}'",
        "web_search for a recent comparison of the two",
        "use the python tool to print how many of the two approaches you "
        "found papers for",
    ]
    # Permute the gathering calls; the python summarisation stays last of the
    # required set, since it counts what the searches found.
    gather = required[:-1]
    order = rng.permutation(len(gather))
    steps = [gather[i] for i in order] + [required[-1]]

    n_extra = int(rng.integers(0, len(_OPTIONAL_STEPS) + 1))
    if n_extra:
        picks = rng.permutation(len(_OPTIONAL_STEPS))[:n_extra]
        steps += [_OPTIONAL_STEPS[i].format(app=app) for i in picks]

    numbered = " ".join(f"({i + 1}) {s}," for i, s in enumerate(steps)).rstrip(",")
    return (f"You are a research assistant comparing {a} and {b} for {app}. "
            f"Do all of the following, one tool call per step: {numbered}. "
            f"Finish with a one-line comparison.")


def _registry():
    from derail.harness.real_tools import _ensure_tls, build_registry
    _ensure_tls()
    # The demo task names its tools explicitly.
    return build_registry(DEMO_TASK_TOOLS)


def _backend():
    from derail.experiments.collect_traces import OllamaBackend
    return OllamaBackend(MODEL, tool_specs=_registry().specs())


#: Failure classes this demo injects. Committing episodes for each is what
#: gives the path an OFFLINE regression test: without them the injected side
#: exists only as live runs, so a regression in detection can only be caught
#: by someone happening to run the demo against a live model.
DEMO_CLASSES = ("looping", "rate_limit", "wrong_document", "malformed_json")


def _cassette(serving: bool = False, corpus: Path | None = None) -> Cassette:
    """Cassette named after the corpus it records, so a repointed TRACES
    cannot quietly keep replaying the previous corpus's recordings.

    `serving=True` for the demo: it replays the committed recordings but
    records any new one under `runs/`, so watching the demo cannot append to
    the dataset the published numbers are computed from.
    """
    return Cassette(f"traces/_cassettes/{(corpus or TRACES).name}",
                    mode="auto", serving=serving)


def collect_healthy(n: int, n_inject: int = 0,
                    out: Path | None = None) -> None:
    """Collect n healthy (and optionally n_inject per class) demo episodes.

    `out` writes to a sibling corpus instead of TRACES. Growing a corpus that
    published numbers are already computed from would move those numbers, so a
    corpus that needs more episodes gets a sibling and the frozen one is left
    alone; the cassette follows the corpus for the reason `_cassette` records.
    """
    from derail.harness.collect_real import collect_dataset

    target = out or TRACES

    # This task has no computable ground-truth answer, so nothing can verify a
    # run at collection time. The null is filtered instead at fit time, where
    # `fit_monitor` drops any run that skipped a call `RESEARCH_SPEC` requires
    # (DESIGN.md Amendment 7). Stated here so "unverified" is a recorded
    # decision rather than a default nobody chose.
    collect_dataset(target, lambda s: _backend(), _registry(),
                    n_healthy=n, n_inject_per_class=n_inject,
                    classes=DEMO_CLASSES if n_inject else (), tau=2,
                    task_fn=demo_task, model=MODEL,
                    cassette=_cassette(corpus=target),
                    allow_unverified_healthy=True)


#: Wall-clock latency dims. On a local box these measure the MACHINE, not the
#: agent, and the calibration corpus was recorded against a live model while
#: the demo replays through a cassette — so the served tool latency collapses
#: to a value the healthy null never contains. Measured on live healthy runs:
#: IDX_TOOL_LATENCY arrives at about -17 standardized units, and the fused
#: score is ~230,000 against a 10%-FA threshold of 191, i.e. the monitor
#: false-alarms on a clean run because of replay timing. Zeroing these two
#: dims SYMMETRICALLY (calibration and serving) drops that to ~10,800 — a 21x
#: reduction — and leaves every agent-behaviour feature untouched. This is the
#: same policy experiments/demo.py already declares; demo_real simply lacked
#: it. Cloud/API deployments, where latency is stationary infrastructure and
#: therefore real agent signal, keep these features.
NUISANCE_DIMS = (IDX_LATENCY_LOG, IDX_TOOL_LATENCY)


def _drop_machine_nuisance(X: np.ndarray) -> np.ndarray:
    """Zero the wall-clock latency dims of one episode matrix, in place."""
    for d in NUISANCE_DIMS:
        if X.shape[-1] > d:
            X[..., d] = 0.0
    return X


#: An episode shorter than this scores nothing at all. The monitor emits 0.0
#: for the first `_WASHOUT` steps, so a run with T <= _WASHOUT produces an
#: all-zero stream: not a quiet healthy run, but a run that was never scored.
#: Measured live, 3 of 8 healthy demo runs ended at T=2 and returned exactly
#: 0.0 — counting those as "healthy, no false alarm" would report the washout
#: back as evidence. `_WASHOUT + 1` is the shortest episode that yields even
#: one scored step.
MIN_SCOREABLE_T = _WASHOUT + 1


def _is_scoreable(T: int) -> bool:
    """Did this episode last long enough for the monitor to score any step?"""
    return T >= MIN_SCOREABLE_T


def fit_monitor(src: Path = TRACES) -> tuple[StreamingChannelMax, float, float]:
    """Cross-fit the channel-max monitor + threshold on healthy demo traces."""
    manifest = json.loads((src / "manifest.json").read_text("utf-8"))
    entries = [e for e in manifest
               if e["failure_class"] is None and _is_scoreable(e["T"])]
    n_vacuous = sum(1 for e in manifest
                    if e["failure_class"] is None and not _is_scoreable(e["T"]))
    if n_vacuous:
        print(f"[demo_real] vacuous-episode policy: {n_vacuous} healthy run(s) "
              f"with T < {MIN_SCOREABLE_T} excluded — shorter than the washout, "
              f"so no step was ever scored and their 0.0 is not evidence.")

    # A healthy null must hold runs that DID the task: a run omitting required
    # calls is strongly anomalous to the monitor, and leaving it in inflates
    # the null and lifts the threshold above where real failures live
    # (DESIGN.md Amendment 7). Unlike the booking demo this task has no
    # computable ground-truth answer, so completeness is checkable here and
    # correctness is not -- that half of the policy is a stated limitation.
    def _complete(path: Path) -> bool:
        steps = [json.loads(x) for x in
                 path.read_text("utf-8").splitlines() if x.strip()]
        return not required_coverage(steps, RESEARCH_SPEC)

    done = [e for e in entries if _complete(src / e["file"])]
    if len(done) >= 10 and len(done) < len(entries):
        print(f"[demo_real] task-completeness policy: "
              f"{len(entries) - len(done)}/{len(entries)} runs excluded from "
              f"the healthy null (required calls missing).")
        entries = done

    def _is_clean(path: Path) -> bool:
        # Exclude REAL glitch episodes (qwen unicode/CJK bursts, empty steps)
        # that inflate theta — a raw-telemetry filter, not score-peeking.
        for line in path.read_text("utf-8").splitlines():
            txt = json.loads(line).get("text", "")
            if not txt.strip():
                return False
            non_ascii = sum(1 for ch in txt if ord(ch) > 127)
            if non_ascii > max(8, 0.2 * len(txt)):
                return False
        return True

    clean = [e for e in entries if _is_clean(src / e["file"])]
    if len(clean) >= 10:
        entries = clean
    healthy = [load_trace_jsonl(src / e["file"], episode_id=e["episode_id"],
                                use_sentence_transformers=False, extended=True)
               for e in entries]
    for ep in healthy:                    # symmetric with the live loop
        _drop_machine_nuisance(ep.X)
    if len(healthy) < 5:
        raise SystemExit(f"only {len(healthy)} healthy demo episodes — "
                         f"collect more: py -m derail.harness.demo_real "
                         f"--collect-healthy 30")

    perm = rng_for(0, "demo-real-split").permutation(len(healthy))
    n_folds = 5 if len(healthy) >= 15 else 3
    folds = [perm[k::n_folds] for k in range(n_folds)]
    val_scores = []
    for k in range(n_folds):
        rest = [healthy[i] for j in range(n_folds) if j != k for i in folds[j]]
        std_k = Standardizer().fit(rest)
        mon_k = StreamingChannelMax(std_k)
        mon_k.fit(rest)
        val_scores += [mon_k.score_episode(healthy[i]) for i in folds[k]]
    std = Standardizer().fit(healthy)
    mon = StreamingChannelMax(std)
    mon.fit(healthy)
    theta = float(pick_threshold(val_scores, fa_budget=FA_BUDGET))
    theta5 = float(pick_threshold(val_scores, fa_budget=0.05))
    print(f"[demo_real] machine-nuisance policy: latency dims "
          f"{list(NUISANCE_DIMS)} zeroed at calibration and serving "
          f"(local wall-clock timing measures the box, and the null was "
          f"recorded live while the demo replays from a cassette).")
    print(f"[demo_real] fitted on {len(healthy)} healthy demo episodes; "
          f"theta(10% FA)={theta:.2f}, theta(5% FA)={theta5:.2f}")
    return mon, theta, theta5


def run_and_score(monitor: StreamingChannelMax, theta: float, registry,
                  backend, *, inject_class: str | None = None, tau: int = 2,
                  seed: int = 0, cassette: Cassette | None = None) -> dict:
    """Run one healthy/injected demo episode; return steps + scores + alarm."""
    injector = (ToolInjector(inject_class, tau=tau, seed=7)
                if inject_class else None)
    steps = run_real_episode(backend, registry, demo_task(seed), max_steps=12,
                             cassette=cassette, injector=injector)
    # An injected run can end at or before its own onset — the agent gives up,
    # or the injected fault halts it, on the very step the fault lands. Such a
    # run has NO post-onset horizon, so it is not an example of "the monitor
    # missed a failure"; there was nothing after the onset to detect. Episode
    # asserts 0 < tau < T, so labelling it would crash. Report it as
    # unscoreable instead of crashing or, worse, counting it as a miss.
    if inject_class is not None and len(steps) <= tau + 1:
        return {"steps": steps, "scores": [], "alarm_step": None,
                "inject_class": inject_class, "inject_tau": tau,
                "T": len(steps), "unscoreable": True}
    ep = episode_from_trace(
        steps, "demo", tau=(tau if inject_class else None),
        failure_class=(inject_class or None),
        severity=(0.5 if inject_class else None),
        use_sentence_transformers=False, extended=True)
    _drop_machine_nuisance(ep.X)          # symmetric with fit_monitor
    scores = monitor.score_episode(ep)
    alarm = next((t for t, s in enumerate(scores) if s > theta), None)
    return {"steps": steps, "scores": [float(s) for s in scores],
            "alarm_step": alarm, "inject_class": inject_class,
            "inject_tau": tau if inject_class else None, "T": ep.T,
            "unscoreable": False}


# --------------------------------------------------------------- smoke test
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="py -m derail.harness.demo_real")
    parser.add_argument("--collect-healthy", type=int, default=0)
    parser.add_argument("--collect-injected", type=int, default=0,
                        help="episodes per failure class")
    parser.add_argument("--out", type=Path, default=None,
                        help="collect into this corpus directory instead of "
                             f"{TRACES.name} (use for a sibling corpus so the "
                             "frozen one keeps its published numbers)")
    parser.add_argument("--demo", action="store_true",
                        help="fit on collected healthy + run healthy and each "
                             "injected class, print detection")
    args = parser.parse_args()

    if args.collect_healthy or args.collect_injected:
        collect_healthy(args.collect_healthy, args.collect_injected,
                        out=args.out)
    elif args.demo:
        registry = _registry()
        mon, theta, theta5 = fit_monitor()
        cas = _cassette(serving=True)
        print(f"\n{'scenario':<18} {'alarm':>6} {'max_score':>12}  {'T':>2}  "
              f"verdict")
        n_fa = n_missed = 0
        for i, fc in enumerate((None, "looping", "rate_limit", "wrong_document",
                                "malformed_json")):
            r = run_and_score(mon, theta, registry, _backend(),
                              inject_class=fc, tau=2, seed=100 + i, cassette=cas)
            mx = max(r["scores"]) if r["scores"] else 0.0
            alarmed = r["alarm_step"] is not None
            # The healthy control is the point of the table: say out loud
            # whether it alarmed, rather than printing a number to read past.
            if r.get("unscoreable"):
                verdict = (f"UNSCOREABLE (ended at T={r['T']} <= tau+1; "
                           f"no post-onset horizon)")
            elif fc is None:
                verdict = "FALSE ALARM" if alarmed else "clean"
                n_fa += alarmed
            elif not alarmed:
                verdict = "MISSED"
                n_missed += 1
            else:
                verdict = f"detected @{r['alarm_step']} (tau={r['inject_tau']})"
            print(f"{fc or 'healthy':<18} {str(r['alarm_step']):>6} "
                  f"{mx:>12.1f}  {r['T']:>2}  {verdict}")
        print(f"\ntheta(10% FA)={theta:.1f}  ->  "
              f"{n_missed} missed, {n_fa} healthy false alarm(s)")
        if n_fa:
            # A healthy control that alarms invalidates the whole table: if the
            # clean run scores in the same range as the injected ones, the
            # alarms above are the monitor reporting "this run is unlike my
            # calibration data", not "this run derailed". Say so here rather
            # than let a reader count four detections as evidence.
            print(
                "\n  WARNING: the healthy control ALARMED. The detections in\n"
                "  this table are NOT evidence of discrimination — healthy and\n"
                "  injected runs are scoring in the same range, which means the\n"
                "  healthy null does not cover the distribution actually being\n"
                "  served. The null is 48 episodes of one fixed task structure;\n"
                "  its residual spread collapses, so every live run is far from\n"
                "  it. Fix by collecting a larger and more varied healthy null\n"
                "  under the SAME conditions the demo serves:\n"
                "      py -m derail.harness.demo_real --collect-healthy 60")
    else:
        # Offline: fit + score on SCRIPTED healthy episodes (no Ollama/network).
        import tempfile
        from derail.harness.real_tools import ArxivSearch
        from derail.harness.tools import ToolRegistry

        class _Scripted:
            def __init__(self, n=5):
                self.n = n
            def reset(self, task): pass
            def add_tool_results(self, results): pass
            def step(self, t):
                if t < self.n:
                    return {"stop_reason": "tool_use", "text": "searching",
                            "tool_uses": [{"id": f"c{t}", "name": "arxiv_search",
                                           "input": {"query": f"q{t}"}}],
                            "output_tokens": 6, "token_logprobs": [-0.2] * 6}
                return {"stop_reason": "end_turn", "text": "done",
                        "tool_uses": [], "output_tokens": 5,
                        "token_logprobs": [-0.3] * 5}

        xml = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
               '<title>Paper</title><id>http://arxiv.org/abs/1v1</id>'
               '<author><name>A</name></author></entry></feed>').encode()
        reg = ToolRegistry([ArxivSearch(get=lambda url: xml)])

        # Build a handful of healthy episodes, fit, then score healthy vs injected.
        with tempfile.TemporaryDirectory() as d:
            src = Path(d)
            from derail.harness.collect_real import collect_dataset
            collect_dataset(src, lambda s: _Scripted(5), reg, n_healthy=8,
                            n_inject_per_class=0, classes=(),
                            task_fn=lambda s: "demo task", model="scripted",
                            verbose=False, allow_unverified_healthy=True)
            mon, theta, theta5 = fit_monitor(src)
            healthy = run_and_score(mon, theta, reg, _Scripted(5))
            injected = run_and_score(mon, theta, reg, _Scripted(5),
                                     inject_class="looping", tau=2)
            assert healthy["inject_class"] is None
            assert injected["inject_class"] == "looping"
            # Injected episode should score no lower than healthy at its peak.
            assert max(injected["scores"]) >= max(healthy["scores"]) - 1e-9
            assert all(np.isfinite(injected["scores"]))
        print("PASS demo_real.py offline self-test | "
              "--collect-healthy N then --demo for the real thing")
