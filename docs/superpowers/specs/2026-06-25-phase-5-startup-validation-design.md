# Phase 5 — Startup Validation

**Date:** 2026-06-25
**Status:** Approved (design); ready for implementation planning
**Scope:** A startup-validation engine + an optional "Validate Acquisition" action. Split into 5a (headless validator) and 5b (UI). No change to the health checks/fusion/calibration.

## 1. Purpose

Startup Validation (spec Ch. 7) verifies the acquisition chain is healthy enough for reliable analysis, by (1) checking system configuration and (2) aggregating health over a ~20 s capture into a single PASS / WARNING / FAIL decision. In the **tester** this is an **optional, on-demand action** (a "Validate Acquisition" button), never a forced gate before normal testing.

## 2. Decision (from brainstorming)

- **Optional button**, not a mandatory gate.
- **Decision rule:** FAIL if any System check errors **or** the FAULT-window fraction ≥ `fault_fraction` (default 0.25) **or** no audio captured; WARNING if some windows are WARNING/FAULT but below the fault fraction; PASS otherwise. Thresholds configurable.
- The validator reuses the existing health pipeline (the same per-window `HealthReport`s the runtime already produces).

## 3. Phase 5a — Headless Validator (`app/health/startup.py`)

Pure stdlib/NumPy, fully testable.

- `StartupDecision` enum: `PASS`, `WARNING`, `FAIL`.
- `SystemValidation(passed: bool, errors: list[str], warnings: list[str])` and
  `validate_system(*, sample_rate, input_ready, calibration_loaded, expected_sample_rate=44100) -> SystemValidation`:
  - error (→ `passed=False`) if `sample_rate != expected_sample_rate` or `not input_ready`;
  - warning (non-blocking) if `not calibration_loaded` ("no calibration profile loaded").
- `SignalAggregate(total, ok, warning, fault, check_failures: dict[str,int])` and
  `aggregate_signal(reports) -> SignalAggregate`: tally each report's `final_state` (OK/WARNING/FAULT; UNKNOWN counted as fault for safety) and, per check, count windows where its `status` is not PASS (`check_failures[check_id]`).
- `StartupResult(decision: StartupDecision, system: SystemValidation, signal: SignalAggregate, summary: str)` and
  `evaluate_startup(system, signal, *, fault_fraction=0.25) -> StartupResult` applying §2's rule; `summary` names the dominant failing checks and the window tally.

These compose into a convenience `run_validation(reports, *, sample_rate, input_ready, calibration_loaded) -> StartupResult`.

## 4. Phase 5b — UI (`main.py`)

- A **"Validate Acquisition"** button (run bar or Settings). On click it performs a **bounded ~20 s capture** from the selected input (reusing the existing single-shot-style bounded acquisition path — `single_shot_duration_sec`/`max_chunks` pattern), feeding each 2.5 s window through the current `health_pipeline.analyze` and collecting the `HealthReport`s.
- On completion it calls `run_validation(...)` (passing `sample_rate=44100`, `input_ready` from the input selection, `calibration_loaded = self.calibration_profile is not None`) and shows the **`StartupResult`** in a dialog/banner: the decision (color-coded), the window tally, and the top failing checks.
- It never blocks normal START TEST; it is a separate, deliberate action.

## 5. Out of Scope

- No forced gate, no auto-run on app launch.
- No change to checks, fusion, calibration, or runtime monitoring.
- Persisting the startup report to disk (possible later).

## 6. Testing

- **5a (headless):** `validate_system` (sample-rate mismatch / input-not-ready → not passed; no calibration → warning, still passed; all good → passed); `aggregate_signal` (counts + per-check failure tally from a list of stand-in reports); `evaluate_startup` (system error → FAIL; fault fraction ≥ threshold → FAIL; empty → FAIL; some warning → WARNING; all OK → PASS).
- **5b (UI):** `ast.parse` + `import main` + unchanged suite; manual GUI — press Validate with a good signal → PASS, with a disconnected/faulty input → FAIL, with a healthy signal but no calibration loaded → PASS/WARNING with a "no calibration" note; normal testing unaffected.
