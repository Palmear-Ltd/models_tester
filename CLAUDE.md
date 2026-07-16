# CLAUDE.md — models_tester

Guidance for Claude Code working in this repo. Keep it current when conventions change.

## What this is

A Python/Tkinter desktop tool (`main.py` → `ModelsTesterApp`, launched via `launcher.py`) for testing Palmear's CNN audio classifier (palm-weevil infestation detection) live from a mic or a WAV file. Inference lives in `inference_utils.py` (TFLite + a `Scaler`).

Bolted onto it is the **Audio Signal Health Monitoring** subsystem in `app/health/` — a quality-assurance layer that judges whether the captured signal is *trustworthy* (sensor/cable/hardware health), independent of and **strictly additive to** the classifier. Full architecture proposal: `arch_update.md`. It was built in phases 0–6 (all complete; see Status).

## Status — Audio Signal Health Monitoring (Phases 0–6 COMPLETE)

Built on branch `features/audio_signal_health`. 259 tests pass. Per-phase specs in `docs/superpowers/specs/`, plans in `docs/superpowers/plans/`. The authoritative phase-by-phase log (what each delivered, file by file) is the memory note `audio-health-monitoring-phases.md` — **read it first** next session.

- **0** foundation + integration seam · **1** time-domain checks T001–T007 + fusion + live indicator · **2** frequency checks F001–F004 + per-check panel + config profiles · **3** calibration (generation CLI `calibrate.py`, evaluation, UI) · **4** stability checks S001–S003 + debounced `RuntimeMonitor` + health timeline plot · **5** startup validation engine + "Validate Acquisition" button · **6** anomaly detection (RMS z-distance) → confidence, surfaced in indicator/log.

Deferred (noted in specs): persisting startup/anomaly reports; full-covariance Mahalanobis (needs an extended profile format).

**Post-completion fix (2026-07-16):** `rootcause.py`'s SENSOR_LINK attribution (the engine behind the "Validate Acquisition" button and live "Likely cause" label) was found firing at confidence 1.00 on every recording tested — fault, clean, and infested alike — making it useless as a tester-facing signal. Root cause: T009's hardcoded click-count thresholds sat inside the noise floor of genuinely healthy piezo contact, and `assess_many` fired SENSOR_LINK on any positive summed score across a session with no baseline-relative floor. Fixed by recalibrating T009 (informed by `calibration_profiles/multiyear_healthy_v1.json`) and replacing the session-level rule with a mean-per-window-score vs. a persisted cutoff (`app/health/check_thresholds.json`, `app/health/rootcause_session_config.json` — same load-JSON-with-fallback idiom as `app/decision/threshold.py`). Along the way, found and worked around a real artifact: a session's first ~4 windows are built from a still-partly-zero rolling buffer (see `main.py:handle_audio_chunk`), and the zero→signal edge spuriously trips the click detector, inflating every session's score by a roughly constant floor — documented but *not* eliminated (see comment next to `DEFAULT_SESSION_CUTOFF` in `rootcause.py`). Spec/plan: `docs/superpowers/{specs,plans}/2026-07-16-rootcause-threshold-recalibration*.md`. Commits `a99777c`, `ef52724`, `35dbd14` (the last also adds a Run-bar "Refresh" devices button and a threshold-config transparency log line).

## `app/health/` map (the portable core)

`models.py` (HealthState, CheckStatus, CheckCategory, AudioWindow, Measurement, SignalCheckResult, HealthReport) · `checks/` (base, time_domain, frequency_domain, stability) · `manager.py` (runs checks, isolates failures) · `feature_prep.py` (shared FFT spectrum) · `fusion.py` (`decide`) · `calibration.py` + `calibration_eval.py` · `config.py` (REGISTRY + profiles + `pipeline_for_profile`) · `pipeline.py` (`HealthAnalysisPipeline.analyze`, 7 stages) · `monitoring.py` (RuntimeMonitor) · `startup.py` (validation) · `anomaly.py` (Mahalanobis/confidence).

## Hard conventions (violating these = rework)

- **`app/health/` imports ONLY stdlib + NumPy.** Never import tester/UI code (`main.py`, tkinter, matplotlib, librosa, tflite). It must stay portable to production (Flutter) later. Reports/profiles are duck-typed across module boundaries to avoid coupling.
- **Health is additive.** It runs alongside `run_inference` in `main.py:handle_audio_chunk` and must never block, slow, or change classification. Each window's pipeline must stay well under the 0.5 s hop.
- **Checks declare `category` as a class attribute ONLY.** The `SignalCheckManager` stamps `result.category`. Do NOT pass `category=` into `SignalCheckResult` in a check's `run`. Direct-call tests assert the *class* attr, not `result.category`. (Implementers keep re-introducing `category=self.category` — reject it.)
- **Frequency checks** read the shared spectrum from `feature_prep.prepare_features` (never recompute an FFT) and PASS on silence/non-finite (time-domain critical checks own catastrophic failure).
- **State vs confidence:** `fusion.decide(results, calibration_evaluation=None, anomaly_result=None)`. Calibration can only *escalate* state (more-severe-wins). Anomaly sets **confidence only, never state**. No profile loaded → unchanged behavior.
- `main.py` has a `SAMPLE_RATE = 44100` constant; the whole pipeline runs at that rate, mono, 2.5 s windows at 0.5 s hop.

## Workflow

- **Claude MAY commit locally** (per task/phase) — commit messages carry **NO `Co-Authored-By` trailer**. **Pushing and merging to `main` stay the owner's call.** (Superseded the old "owner commits everything manually / never `git add`" rule on 2026-07-06.)
- Build features through superpowers: **brainstorming → writing-plans → subagent-driven-development** (the owner consistently picks subagent-driven). Use **systematic-debugging** for bugs. TDD throughout (synthetic signals: silence, sine, noise, clipped, DC-offset, hum).
- Specs → `docs/superpowers/specs/YYYY-MM-DD-*.md`; plans → `docs/superpowers/plans/`.
- Big features ship in sub-phases (a/b/c): headless engine first, then UI. Each sub-phase leaves the tester working + one visible feature.

## Environment & testing

- **`.venv` is Python 3.13 with only `numpy` + `pytest` installed.** That's enough for everything in `app/health/` and all tests. The **GUI** (`launcher.py`) and any tflite/audio work need the full `requirements.txt` (tensorflow/librosa/tkinter/sounddevice) installed first — so manual GUI checks are the **owner's** job; Claude verifies headless.
- Run tests: `.venv/bin/python -m pytest tests/ -q`
- UI changes Claude can verify headless: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read())"` then `.venv/bin/python -c "import main"`, plus the unchanged test count.

## Gotchas

- **Pyright "import could not be resolved" / "not accessed" warnings are FALSE POSITIVES** (the IDE isn't pointed at `.venv`). Ignore them.
- **Sonnet review/implementer subagents hallucinate NumPy deprecations** (e.g. claim `np.ptp`/`np.hanning` were removed, or `np.hann` exists). All false — verify against the installed NumPy before acting.
- **Reviewer/implementer subagents often don't relay their final message** (they end with "Ready"/"Done"). Verify their work yourself (run the tests/grep), or parse their transcript JSONL under the session's `tasks/<agentId>.output`.
- **Trust the spec over a test assertion when they conflict.** Phase 6a example: a faulty `== 0.0` assertion in a plan led an implementer to replace the agreed *linear* confidence formula with a piecewise cliff — the formula was right, the test was wrong.
- `models/9_1_2/scaler.json` is the default scaler (a `.npz` there was once a corrupt HTML download; `Scaler.load` now surfaces real errors via `last_error`).

## Persistent memory

Project memory lives at `~/.claude/projects/-Users-bashar-workspace-palmear-models-tester/memory/` (index `MEMORY.md`). The phase log there is the fastest way to reload context — start there.
