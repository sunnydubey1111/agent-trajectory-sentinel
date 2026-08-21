# Framework x real-tool validation: LangGraph and AutoGen

Frozen monitor: `esn_cusum_max[e,m]`, theta_b5=40.08187009017411 (`results/framework_monitor_freeze.json`). Tasks: arxiv_paper_search, multi_city_weather. Model: qwen2.5:7b. Seed base: 52026.

The two frameworks disagree sharply (per-framework `healthy_fa_rate` below): this monitor was fit once, only on episodes collected through this project's own custom `OllamaBackend` harness (see `framework_monitor_freeze.py`), and never retrained per framework, by design (no recalibration after this study's episodes exist). A monitor's detection/false-alarm rate does not automatically zero-shot-transfer to a different agent orchestrator's step/latency/tool-call shape -- that is the population difference this study measures, not a defect in either corpus. The pooled row below states this population explicitly; treat it as descriptive, not as a single operating point, since it averages two frameworks whose rates differ by construction.

**LangGraph's `detection_rate=0.833` is not a meaningful detection result on its own** -- it sits beside `healthy_fa_rate=1.000`: the monitor alarms on every LangGraph episode collected, healthy or injected, past its washout, so it does not distinguish derailment from this framework's own step/latency shape at all. AutoGen's `detection_rate=0.833` at `healthy_fa_rate=0.667` false alarms is the more informative of the two per-framework results here, though a false-alarm rate that high is still not a usable operating point.

## langgraph7b_real (n=24)
- episode_auc (offline ranking): 0.549
- healthy_fa_rate: 1.000
- detection_rate: 0.833
- early_alarm_rate: 0.083
- median_delay: 0.5
- median_lead: 4.0
- per_class:
  - goal_drift: detection_rate=0.667
  - looping: detection_rate=1.000
  - tool_cascade: detection_rate=0.667
  - context_corruption: detection_rate=1.000

## autogen7b_real (n=24)
- episode_auc (offline ranking): 0.653
- healthy_fa_rate: 0.667
- detection_rate: 0.833
- early_alarm_rate: 0.083
- median_delay: 1.0
- median_lead: 5.0
- per_class:
  - goal_drift: detection_rate=0.667
  - looping: detection_rate=1.000
  - tool_cascade: detection_rate=0.667
  - context_corruption: detection_rate=1.000

## Pooled (both frameworks, n=48)
- episode_auc (offline ranking): 0.576
- healthy_fa_rate: 0.833
- detection_rate: 0.833
- early_alarm_rate: 0.083
- median_delay: 1.0
- median_lead: 4.0
- per_class:
  - goal_drift: detection_rate=0.667
  - looping: detection_rate=1.000
  - tool_cascade: detection_rate=0.667
  - context_corruption: detection_rate=1.000

## Diagnosis: why LangGraph false-alarms at 100%

Both harnesses measure `latency_s` identically -- LLM-call time only, tool execution excluded -- structurally verified and covered by a live cross-harness contract test (`tests/test_telemetry_contracts.py`, against real Ollama).

Verified against the frozen monitor's own per-channel CUSUM streams (`ChannelMaxESNMonitor.subs`, channels e and m scored separately): on every LangGraph and AutoGen healthy episode checked, the **meta channel (m)** -- action one-hot, log-latency, log-token-count, error flag -- crosses theta_b5 first and by a wide margin over the semantic channel (e). The dominant driver within m is log-latency: per-step LLM-call latency in this corpus averages **0.53s** (LangGraph) / **0.47s** (AutoGen) (excluding each episode's final synthesis step), against the frozen monitor's native-harness training mean of **4.22s** -- roughly an 8x gap, with per-step z-scores of -3 to -4 against the training distribution. The final synthesis step separately spikes log-token-count (z up to +4) on both frameworks, a real content difference (a longer final answer than the training distribution's typical step). Every other part of the scoring path (channel/feature indices, standardization against the frozen training statistics, per-episode CUSUM reset, threshold semantics, u-channel exclusion for the no-logprobs case) was checked against source and found correct.

This is calibration/generalization failure, not architecture failure or a scoring bug: LangGraph/AutoGen's underlying Ollama client issues and consumes each call differently enough that its LLM-call latency runs on a different scale than this project's native `OllamaBackend` harness, even though both measure the identical quantity. Root-causing exactly why the call itself is faster is future/secondary work, out of scope here. Confirming evidence, not merely consistent with it: scored IN-POPULATION -- fit and thresholded on the corpus's own mock-tool LangGraph/AutoGen traces (`traces/langgraph7b`, `traces/autogen7b`), the same per-corpus refit procedure `run_hybrid_study.py` uses -- `esn_cusum_max`'s healthy false-alarm rate is **0.000** for LangGraph and **0.083** for AutoGen (`results/tables/hybrid_benchmark.csv`), against the zero-shot rates above. The architecture instruments both frameworks fine; a threshold frozen on this project's own native harness does not transfer to a different orchestrator's step/latency shape. Framework COMPATIBILITY ("ATS can instrument LangGraph/AutoGen") and calibration GENERALIZATION ("one frozen threshold transfers across orchestrators") are separate claims; only the first holds here. Per-framework recalibration is future/secondary work, not a substitute for this frozen result.
