# Audio Signal Health Monitoring — Phased Implementation Design

**Date:** 2026-06-23
**Status:** Approved (design); ready for implementation planning
**Source spec:** `arch_update.md` (full high-level architecture)

## 1. Purpose

`arch_update.md` defines a complete high-level architecture for an Audio Signal
Health Monitoring subsystem: a quality-assurance layer that evaluates whether the
captured audio signal is trustworthy before and during ML classification,
independent of the CNN classifier.

This document defines how to implement that architecture **gradually** in the
`models_tester` application. The work is split into phases. **At the end of every
phase the model tester still runs as before, plus one visible new feature.**

## 2. Context: existing code

The tester already implements the exact acquisition pipeline the spec assumes:

| Spec concept | Existing code |
| --- | --- |
| 0.5s chunks @ 44.1kHz mono | `mic_loop` / `file_loop` (`main.py`) via `sounddevice` InputStream, 0.5s blocks |
| Chunk delivery | `audio_queue` → `process_queue` → `handle_audio_chunk` |
| 2.5s circular buffer | `session_buffer` updated by `np.roll` (`main.py:683`) |
| 2.5s analysis window every 0.5s | sliding-window branch → `run_inference(session_buffer)` |
| Feature extraction | `app/audio/features.py` (`FeatureExtractor`) |
| Classifier | `app/model/inference.py` (`ModelInference`, TFLite) |

**Integration seam:** `handle_audio_chunk` updates `session_buffer`, then builds an
`AudioWindow` and runs the Health Analysis Pipeline on the *same* window, alongside
`run_inference`. The tester GUI is `ModelsTesterApp` (Tkinter) in `main.py`.

## 3. Key decisions

1. **Portable core.** The Health Analysis Pipeline lives in a new self-contained
   `app/health/` package built on **pure NumPy** (`numpy.fft`, hand-rolled mel/band
   math — no librosa/scipy in the core checks), so the logic can later be ported to
   the production (Flutter) app. The package **never imports tester or UI code**.
2. **UI grows per phase.** Surfacing starts as a status indicator + log, grows into a
   dedicated health panel, and ends with a health timeline plot.
3. **Calibration data is available.** Calibration is an early-middle phase (Phase 3),
   after the deterministic checks exist — calibration runs the same pipeline over
   healthy recordings to derive per-check statistics, so checks must exist first.
4. **Health is additive.** The subsystem never modifies, delays, or blocks the CNN
   prediction. It only produces a parallel Health Report.
5. **Real-time budget.** The pipeline must complete well under the 0.5s analysis
   interval. Checks stay computationally cheap; this is a hard constraint per phase.

## 4. Phasing strategy

**Capability-stacked**: each phase adds one coherent layer from the spec, with a
**Phase 0** that stands up the full pipeline skeleton end-to-end first, so from
Phase 1 onward every check flows through real wiring with zero architectural churn.

Alternatives considered and rejected: a pure vertical thin-slice (front-loads
plumbing before broad value) and library-first/UI-last (violates the "visible
feature each phase" requirement).

Each phase is TDD-friendly using the spec's Chapter 12 approach — synthetic signals
(silence, sine, clipped, DC-offset, broadband noise) with known expected outputs.

## 5. Phases

### Phase 0 — Foundation & integration seam
- New `app/health/` package. Pure-data objects (spec Ch. 5): `AudioWindow`
  (immutable), `Measurement`, `SignalCheckResult`, `HealthReport`, `HealthState`
  enum.
- `HealthAnalysisPipeline` skeleton (the 7 stages as no-ops) + `SignalCheckManager`
  registry.
- Seam: `handle_audio_chunk` builds an `AudioWindow` from `session_buffer` and runs
  the pipeline alongside `run_inference`, never blocking or altering the prediction.
- **Visible feature:** a health state indicator (shows `UNKNOWN`).
- **Exit:** tester behaves identically; pipeline produces an empty report each window.

### Phase 1 — Time-domain checks + basic Decision Fusion
- T001–T007: flatline, signal energy, peak amplitude, clipping, crest factor, DC
  offset, zero-crossing rate — independent pure-numpy checks.
- Rule-based Decision Fusion (spec Ch. 10 rules) → OK/WARNING/FAULT + diagnostics.
  Manual thresholds via a minimal config.
- **Visible feature:** live OK/WARNING/FAULT indicator + diagnostic messages in the
  existing log panel.
- **Exit:** tester flags disconnected sensor / clipping / dead signal in real time.

### Phase 2 — Frequency-domain checks + shared Feature Preparation + config profiles
- Feature Preparation stage computes FFT / power spectrum / frequency bins once and
  shares them across checks.
- F001–F004: spectral shape, spectral flatness, band energy distribution, electrical
  hum detection.
- Formalize the config system (spec Ch. 4.7): global/category/check hierarchy,
  mandatory checks, profiles (development / production / diagnostic / minimal).
- **Visible feature:** dedicated **health panel** — per-check table with measurements,
  status, and overall state.
- **Exit:** spectral degradations detected; checks toggled by profile.

### Phase 3 — Calibration (Layer 2)
- Offline calibration tool that runs the *same* pipeline over healthy recordings
  (`test_data` / calibration set) → `CalibrationProfile` JSON with per-measurement
  statistics (mean/median/std/min/max/p5/p95, spec Ch. 6).
- Calibration Evaluation stage; checks switch from manual to calibration-derived
  thresholds.
- **Visible feature:** active-profile display + calibration evaluation column in the
  panel; a "Generate calibration profile" action.
- **Exit:** thresholds adapt to the real sensor instead of hand-tuned constants.

### Phase 4 — Stability checks + history + Runtime Monitoring
- Bounded history of `SignalCheckResult`s; S001–S003: energy stability, spectral
  stability, long-term noise floor.
- Runtime Monitoring (spec Ch. 8): monitoring events, fault persistence, recovery
  detection — distinguishing transient blips from real faults.
- **Visible feature:** **health timeline plot** alongside score/energy plots + an
  event log.
- **Exit:** gradual degradation and intermittent faults surface without false alarms
  on isolated windows.

### Phase 5 — Startup Validation
- System Validation (device / sample-rate / mono / profile checks) + 20s Signal
  Validation aggregating ~40 windows → PASS/WARNING/FAIL + Startup Health Report
  (spec Ch. 7). **Optional / toggleable** so the tester can still run ad-hoc.
- **Visible feature:** a startup-gate banner/dialog before runtime with the startup
  decision.
- **Exit:** the tester can refuse/warn before testing on a bad acquisition chain.

### Phase 6 — Anomaly Detection (Layer 3) + confidence + reporting/logging polish
- Statistical anomaly detection on the calibrated feature vector (e.g. Mahalanobis
  distance from the calibration profile — pure numpy, no ML dependency), confidence
  estimation refinement, structured diagnostic logging + per-check performance timing
  (spec Ch. 9–11).
- **Visible feature:** anomaly score + confidence in the panel; structured health log
  export.
- **Exit:** full spec coverage; subtle abnormalities caught beyond fixed thresholds.

## 6. Module layout (target)

```
app/health/
  __init__.py
  models.py            # AudioWindow, Measurement, SignalCheckResult, HealthReport, HealthState
  pipeline.py          # HealthAnalysisPipeline (7 stages)
  manager.py           # SignalCheckManager (registry, isolation, stats)
  config.py            # config hierarchy + profiles
  checks/
    base.py            # SignalHealthCheck interface
    time_domain.py     # T001–T007
    frequency_domain.py# F001–F004
    stability.py       # S001–S003
  calibration.py       # CalibrationProfile + offline generation (Phase 3)
  fusion.py            # Decision Fusion (Ch. 9–10)
  monitoring.py        # Runtime Monitoring: events, persistence, recovery (Phase 4)
  startup.py           # Startup Validation (Phase 5)
  anomaly.py           # Statistical anomaly detection (Phase 6)
```

The tester integrates only through `HealthAnalysisPipeline` and the data objects.

## 7. Cross-cutting constraints

- **Independence:** `app/health/` imports neither `main.py` nor any UI module.
- **Determinism:** identical input window → identical Health Report.
- **Performance:** total pipeline execution << 0.5s per window; per-check timing
  recorded from Phase 6.
- **Explainability:** every WARNING/FAULT traces to specific `SignalCheckResult`s.
- **No regression:** the CNN prediction path is never modified by health monitoring.

## 8. Out of scope (deferred / future)

Per spec Ch. 13, treated as future extensions, not part of this plan: ML-based
decision fusion, predictive/remaining-useful-life monitoring, multi-channel/multi-
sensor acquisition, fleet/cloud integration, deep-learning anomaly detection.
