# Phase 3b — Calibration Evaluation & Integration

**Date:** 2026-06-24
**Status:** Approved (design); ready for implementation planning
**Scope:** Headless calibration evaluation engine + pipeline/fusion integration. No UI (that is Phase 3c).

## 1. Purpose

Phase 3a generates a `CalibrationProfile` (per-check, per-measurement statistics) from healthy recordings but does nothing with it at runtime. Phase 3b makes the health pipeline *use* a profile: each analysis window's measurements are compared against the calibrated normal range, producing per-measurement deviation verdicts that Decision Fusion folds into the final health state. This adds a **sensor-relative** layer (drift / abnormal-for-this-rig) on top of the absolute time/frequency checks.

## 2. Key Decisions (from brainstorming)

- **Separate Calibration Evaluation stage**, not threshold injection into checks. The deterministic checks keep their manual thresholds as an always-on catastrophic safety net (flatline/clipping work with no profile loaded); calibration is an *added* adaptive layer.
- **Percentile-based deviation rule** (robust for skewed audio measurements).
- Additive fusion: with no profile, behavior is exactly as today.

## 3. Deviation Rule

For each measurement present in **both** the window's results and the profile (`MeasurementStats` with `p5`, `p95`, `min`, `max`):

- **PASS** if `warn_lo ≤ value ≤ warn_hi`, where the normal band `[p5, p95]` is widened by a small tolerance: `warn_lo = p5 − tol`, `warn_hi = p95 + tol`, `tol = warn_margin_frac · (p95 − p5)` (default `warn_margin_frac = 0.1`).
- **FAULT** if `value < fault_lo` or `value > fault_hi`, where `fault_lo = min − f·(max − min)`, `fault_hi = max + f·(max − min)` (default `f = 0.5`).
- **WARNING** otherwise (outside the 90% band but within the extended range).

Edge cases: if `max == min` (degenerate/constant measurement) the range is zero; apply a tiny absolute epsilon so float jitter doesn't FAULT. Measurements absent from the profile are skipped (not evaluated). All thresholds (`warn_margin_frac`, `f`, epsilon) are parameters with the stated defaults.

## 4. `CalibrationEvaluation` Object

Stored in `HealthReport.calibration_evaluation` (slot already exists). Contains:
- `deviations`: list of per-measurement records `(check_id, measurement, value, p5, p95, verdict)` for measurements that were evaluated.
- `warn_count`, `fault_count`: tallies of WARNING/FAULT verdicts.
- `summary`: short human-readable string naming the deviating measurements (empty when all PASS).

Lives in `app/health/calibration_eval.py` (NumPy/stdlib only) with `evaluate_calibration(results, profile, *, warn_margin_frac=0.1, fault_margin=0.5) -> CalibrationEvaluation`, where `results` is the window's `list[SignalCheckResult]` (each carries its `measurements`).

## 5. Pipeline Wiring

- `HealthAnalysisPipeline.__init__(manager=None, calibration_profile=None)` stores the profile.
- Stage 4 `_evaluate_calibration(results)`: returns `None` if no profile (today's behavior), else `evaluate_calibration(results, self.calibration_profile)`.
- `build_pipeline(config, calibration_profile=None)` and `pipeline_for_profile(name, calibration_profile=None)` gain the optional profile so callers (3c UI) can supply one.

## 6. Fusion Integration

`decide(results, calibration_evaluation=None) -> (HealthState, float, str)` (extended signature; default `None` preserves current behavior):
- Compute the **check-based** state/confidence/summary exactly as today.
- Compute a **calibration** state from the evaluation: `≥2 FAULT deviations → FAULT`; `exactly 1 FAULT or any WARNING → WARNING`; else `OK`. (A single FAULT deviation is treated as WARNING to avoid one-off content extremes faulting; this mirrors the existing "2+ major → FAULT" check rule.)
- **Final state = the more severe of the check state and the calibration state** (severity order OK < WARNING < FAULT; UNKNOWN only when no checks executed and no calibration signal).
- Merge diagnostics so the summary names deviating calibrated measurements.

`pipeline.analyze` Stage 6 calls `decide(results, calibration_evaluation)`.

## 7. Out of Scope (Phase 3c)

- Selecting/loading a profile in the UI, showing the active profile, a calibration column in the Signal Health panel, and a "Generate profile" button. (3b is exercised only via code/tests.)
- No change to the calibration *generation* (3a) or to what measurements exist.

## 8. Testing (headless)

- **Deviation rule:** synthetic `MeasurementStats` + values across PASS/WARNING/FAULT bands; degenerate (max==min) case.
- **`evaluate_calibration`:** results with measurements vs a small profile → correct verdicts, counts, and that unprofiled measurements are skipped.
- **Pipeline:** `analyze` with no profile → `calibration_evaluation is None` and unchanged state; with a profile → a `CalibrationEvaluation` is produced.
- **Fusion:** check-only behavior unchanged when `calibration_evaluation=None`; calibration WARNING/FAULT escalates the final state via "more severe"; a single calibration FAULT yields WARNING, ≥2 yields FAULT.
- Full suite stays green (no UI touched).
