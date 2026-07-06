# Phase 3b — Calibration Evaluation & Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pipeline *use* a `CalibrationProfile`: compare each window's measurements against the calibrated normal range (percentile-based), produce per-measurement deviation verdicts, and fold them into Decision Fusion — additively, so behavior is unchanged when no profile is loaded.

**Architecture:** A pure-NumPy `app/health/calibration_eval.py` compares `SignalCheckResult` measurements against a profile's `MeasurementStats` and returns a `CalibrationEvaluation`. The pipeline gains an optional `calibration_profile`; Stage 4 returns `None` (no profile) or the evaluation. `decide()` takes the evaluation and the final state is the more severe of the check-based state and the calibration state. No UI (Phase 3c).

**Tech Stack:** Python 3.13, NumPy, stdlib; pytest. `app/health/` stays NumPy-only.

> **Commit policy:** Owner commits manually. **Do NOT run git commit/add/push.** End at "tests pass."

> **Reference spec:** `docs/superpowers/specs/2026-06-24-phase-3b-calibration-evaluation-design.md`. Builds on 3a (`app/health/calibration.py`: `CalibrationProfile`, `MeasurementStats`, `generate_profile`).

---

## Current State

- `app/health/calibration.py`: `CalibrationProfile.statistics` is `{check_id: {measurement_name: MeasurementStats}}`; `MeasurementStats` has `count, mean, median, std, minimum, maximum, p5, p95`. `generate_profile(signals, sample_rate, *, profile_id, ...)` builds one.
- `app/health/models.py`: `SignalCheckResult` has `check_id`, `measurements` (list of `Measurement(name, value, unit)`), `status`, `category`, `executed`; `CheckStatus` (PASS/WARNING/FAIL/NOT_EXECUTED); `HealthState`; `HealthReport.calibration_evaluation` slot exists.
- `app/health/pipeline.py`: `HealthAnalysisPipeline.__init__(self, manager=None)`; Stage 4 `calibration_evaluation = self._evaluate_calibration(results)` (stub returns `None`); Stage 6 `final_state, confidence, summary = decide(results)`.
- `app/health/fusion.py`: `decide(results) -> (HealthState, float, str)`.
- `app/health/config.py`: `build_pipeline(config) -> HealthAnalysisPipeline`; `pipeline_for_profile(name) -> HealthAnalysisPipeline`.
- 88 tests pass.

## File Structure

**Create:** `app/health/calibration_eval.py` — `MeasurementDeviation`, `CalibrationEvaluation`, `evaluate_calibration`. Test: `tests/health/test_calibration_eval.py`.
**Modify:** `app/health/pipeline.py` (optional profile + Stage 4 + Stage 6 call), `app/health/config.py` (pass-through profile), `app/health/fusion.py` (fold calibration into `decide`). Tests: `tests/health/test_pipeline.py`, `tests/health/test_fusion.py`.

---

## Task 1: Calibration evaluation engine

**Files:** Create `app/health/calibration_eval.py`; Test `tests/health/test_calibration_eval.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/health/test_calibration_eval.py`:

```python
from app.health.calibration import CalibrationProfile, MeasurementStats
from app.health.calibration_eval import CalibrationEvaluation, evaluate_calibration
from app.health.models import CheckStatus, Measurement, SignalCheckResult


def _stats(p5, p95, mn, mx):
    return MeasurementStats(
        count=100, mean=(p5 + p95) / 2, median=(p5 + p95) / 2, std=1.0,
        minimum=mn, maximum=mx, p5=p5, p95=p95,
    )


def _profile():
    return CalibrationProfile(
        profile_id="p", statistics={"T002": {"rms": _stats(0.1, 0.3, 0.05, 0.4)}}
    )


def _result(check_id, **meas):
    return SignalCheckResult(
        check_id=check_id,
        check_name=check_id,
        measurements=[Measurement(n, v) for n, v in meas.items()],
    )


def test_pass_inside_band():
    ev = evaluate_calibration([_result("T002", rms=0.2)], _profile())
    assert ev.deviations == []
    assert ev.warn_count == 0 and ev.fault_count == 0


def test_warning_outside_band_within_range():
    # 0.35 > p95 (0.3) but < fault_hi (0.4 + 0.5*0.35 = 0.575) -> WARNING.
    ev = evaluate_calibration([_result("T002", rms=0.35)], _profile())
    assert ev.warn_count == 1 and ev.fault_count == 0
    assert ev.deviations[0].verdict is CheckStatus.WARNING


def test_fault_far_outside_range():
    ev = evaluate_calibration([_result("T002", rms=2.0)], _profile())
    assert ev.fault_count == 1
    assert ev.deviations[0].verdict is CheckStatus.FAIL


def test_unprofiled_measurements_and_checks_are_skipped():
    ev = evaluate_calibration(
        [_result("T002", rms=0.2, extra=99.0), _result("T999", foo=1.0)], _profile()
    )
    assert ev.deviations == []  # rms passes; extra/foo/T999 not in profile


def test_degenerate_constant_measurement():
    prof = CalibrationProfile(
        profile_id="p", statistics={"T006": {"dc_offset": _stats(0.0, 0.0, 0.0, 0.0)}}
    )
    assert evaluate_calibration([_result("T006", dc_offset=0.0)], prof).deviations == []
    assert evaluate_calibration([_result("T006", dc_offset=0.5)], prof).fault_count == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_calibration_eval.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.health.calibration_eval'`.

- [ ] **Step 3: Implement `app/health/calibration_eval.py`**

```python
"""Calibration Evaluation (spec §3.8): compare a window's measurements against a
CalibrationProfile's per-measurement normal range and flag deviations.

Percentile-based: PASS inside [p5, p95] (widened by a tolerance), FAULT well
outside the observed [min, max] range, WARNING in between. Pure stdlib + NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.health.models import CheckStatus


@dataclass
class MeasurementDeviation:
    check_id: str
    measurement: str
    value: float
    p5: float
    p95: float
    verdict: CheckStatus  # PASS / WARNING / FAIL


@dataclass
class CalibrationEvaluation:
    deviations: list = field(default_factory=list)  # list[MeasurementDeviation]
    warn_count: int = 0
    fault_count: int = 0
    summary: str = ""


def _verdict(value, stats, warn_margin_frac, fault_margin, eps) -> CheckStatus:
    span = stats.p95 - stats.p5
    warn_lo = stats.p5 - warn_margin_frac * span
    warn_hi = stats.p95 + warn_margin_frac * span
    rng = stats.maximum - stats.minimum
    fault_lo = stats.minimum - fault_margin * rng - eps
    fault_hi = stats.maximum + fault_margin * rng + eps
    if value < fault_lo or value > fault_hi:
        return CheckStatus.FAIL
    if value < warn_lo or value > warn_hi:
        return CheckStatus.WARNING
    return CheckStatus.PASS


def evaluate_calibration(
    results, profile, *, warn_margin_frac=0.1, fault_margin=0.5, eps=1e-9
) -> CalibrationEvaluation:
    deviations = []
    warn_count = 0
    fault_count = 0
    for result in results:
        check_stats = profile.statistics.get(result.check_id)
        if not check_stats:
            continue
        for m in result.measurements:
            stats = check_stats.get(m.name)
            if stats is None:
                continue
            verdict = _verdict(float(m.value), stats, warn_margin_frac, fault_margin, eps)
            if verdict is CheckStatus.PASS:
                continue
            deviations.append(
                MeasurementDeviation(
                    check_id=result.check_id,
                    measurement=m.name,
                    value=float(m.value),
                    p5=stats.p5,
                    p95=stats.p95,
                    verdict=verdict,
                )
            )
            if verdict is CheckStatus.FAIL:
                fault_count += 1
            else:
                warn_count += 1
    if deviations:
        parts = [
            f"{d.check_id}.{d.measurement}={d.value:.4g} (cal {d.p5:.4g}..{d.p95:.4g})"
            for d in deviations
        ]
        summary = "calibration deviations: " + "; ".join(parts)
    else:
        summary = ""
    return CalibrationEvaluation(
        deviations=deviations,
        warn_count=warn_count,
        fault_count=fault_count,
        summary=summary,
    )
```

(NumPy is not actually needed here; stdlib + `app.health.models` only. The module stays portable.)

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_calibration_eval.py -q`
Expected: PASS (5 passed).

---

## Task 2: Pipeline & config wiring (optional profile)

**Files:** Modify `app/health/pipeline.py`, `app/health/config.py`; Test `tests/health/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/health/test_pipeline.py`:

```python
import numpy as np  # noqa: E402

from app.health.calibration import generate_profile  # noqa: E402
from app.health.calibration_eval import CalibrationEvaluation  # noqa: E402
from app.health.config import pipeline_for_profile  # noqa: E402

SR_CAL = 44100


def _sine6():
    n = int(6.0 * SR_CAL)
    return (0.3 * np.sin(2 * np.pi * 1000.0 * np.arange(n) / SR_CAL)).astype(np.float32)


def _cal_window():
    return AudioWindow(samples=np.zeros(110250, dtype=np.float32), sample_rate=SR_CAL)


def test_pipeline_without_profile_has_no_calibration_eval():
    report = pipeline_for_profile("development").analyze(_cal_window())
    assert report.calibration_evaluation is None


def test_pipeline_with_profile_produces_calibration_eval():
    profile = generate_profile([_sine6()], SR_CAL, profile_id="p")
    sine_window = AudioWindow(samples=_sine6()[:110250], sample_rate=SR_CAL)
    report = pipeline_for_profile("development", calibration_profile=profile).analyze(sine_window)
    assert isinstance(report.calibration_evaluation, CalibrationEvaluation)
```

(`AudioWindow` and `HealthAnalysisPipeline` are already imported at the top of `test_pipeline.py` from Phase 0.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_pipeline.py -q`
Expected: FAIL — `pipeline_for_profile()` got an unexpected keyword argument `calibration_profile`.

- [ ] **Step 3: Add the profile to the pipeline**

In `app/health/pipeline.py`, add the import after `from app.health.fusion import decide`:

```python
from app.health.calibration_eval import evaluate_calibration
```

Change `__init__`:

```python
    def __init__(self, manager: Optional[SignalCheckManager] = None):
        self.manager = manager if manager is not None else SignalCheckManager()
```

to:

```python
    def __init__(self, manager: Optional[SignalCheckManager] = None, calibration_profile=None):
        self.manager = manager if manager is not None else SignalCheckManager()
        self.calibration_profile = calibration_profile
```

Change `_evaluate_calibration`:

```python
    def _evaluate_calibration(self, results: list) -> Optional[Any]:
        # Phase 0 no-op; Phase 3 compares results against the calibration profile.
        return None
```

to:

```python
    def _evaluate_calibration(self, results: list) -> Optional[Any]:
        if self.calibration_profile is None:
            return None
        return evaluate_calibration(results, self.calibration_profile)
```

- [ ] **Step 4: Thread the profile through config factories**

In `app/health/config.py`, change `build_pipeline`:

```python
def build_pipeline(config: HealthConfig) -> HealthAnalysisPipeline:
    return HealthAnalysisPipeline(manager=build_manager(config))
```

to:

```python
def build_pipeline(config: HealthConfig, calibration_profile=None) -> HealthAnalysisPipeline:
    return HealthAnalysisPipeline(
        manager=build_manager(config), calibration_profile=calibration_profile
    )
```

And `pipeline_for_profile`:

```python
def pipeline_for_profile(name: str) -> HealthAnalysisPipeline:
    return build_pipeline(config_for_profile(name))
```

to:

```python
def pipeline_for_profile(name: str, calibration_profile=None) -> HealthAnalysisPipeline:
    return build_pipeline(config_for_profile(name), calibration_profile=calibration_profile)
```

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_pipeline.py -q`
Expected: PASS (existing pipeline tests + 2 new).

---

## Task 3: Fold calibration into Decision Fusion

**Files:** Modify `app/health/fusion.py`, `app/health/pipeline.py`; Test `tests/health/test_fusion.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/health/test_fusion.py`:

```python
from app.health.calibration_eval import CalibrationEvaluation  # noqa: E402


def _cal(warn=0, fault=0):
    return CalibrationEvaluation(
        deviations=[], warn_count=warn, fault_count=fault,
        summary="cal deviation" if (warn or fault) else "",
    )


def test_calibration_none_preserves_check_decision():
    rs = [_res("T001", CheckStatus.PASS, CheckCategory.CRITICAL)]
    assert decide(rs, None)[0] is HealthState.OK


def test_calibration_warning_escalates_ok_to_warning():
    rs = [_res("T001", CheckStatus.PASS, CheckCategory.CRITICAL)]
    assert decide(rs, _cal(warn=1))[0] is HealthState.WARNING


def test_single_calibration_fault_is_warning():
    rs = [_res("T001", CheckStatus.PASS, CheckCategory.CRITICAL)]
    assert decide(rs, _cal(fault=1))[0] is HealthState.WARNING


def test_two_calibration_faults_is_fault():
    rs = [_res("T001", CheckStatus.PASS, CheckCategory.CRITICAL)]
    assert decide(rs, _cal(fault=2))[0] is HealthState.FAULT


def test_check_fault_not_downgraded_by_clean_calibration():
    rs = [_res("T001", CheckStatus.FAIL, CheckCategory.CRITICAL)]
    assert decide(rs, _cal())[0] is HealthState.FAULT
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_fusion.py -q`
Expected: FAIL — `decide()` takes 1 positional argument but 2 were given.

- [ ] **Step 3: Replace the `decide` function in `app/health/fusion.py`**

Replace the entire `decide` function with:

```python
_SEVERITY_RANK = {
    HealthState.UNKNOWN: -1,
    HealthState.OK: 0,
    HealthState.WARNING: 1,
    HealthState.FAULT: 2,
}


def _calibration_state(evaluation) -> HealthState:
    """Map a CalibrationEvaluation's deviation tallies to a health state."""
    if evaluation.fault_count >= 2:
        return HealthState.FAULT
    if evaluation.fault_count == 1 or evaluation.warn_count > 0:
        return HealthState.WARNING
    return HealthState.OK


def decide(results, calibration_evaluation=None) -> tuple[HealthState, float, str]:
    """Return (final_state, confidence, diagnostic_summary).

    The check-based decision is computed first; if a calibration evaluation is
    supplied, the final state is the more severe of the two (calibration never
    downgrades a check verdict).
    """
    # `executed` is stamped True by the SignalCheckManager once a check has run
    # (a NOT_EXECUTED result is left executed=False), so this filter == "ran".
    executed = [r for r in results if r.executed]
    if not executed:
        state, confidence, summary = HealthState.UNKNOWN, 0.0, "No signal health checks executed."
    else:
        fails = [r for r in executed if r.status is CheckStatus.FAIL]
        warns = [r for r in executed if r.status is CheckStatus.WARNING]
        critical_fails = [r for r in fails if r.category is CheckCategory.CRITICAL]
        major_fails = [
            r
            for r in fails
            if r.category in (CheckCategory.CRITICAL, CheckCategory.PRIMARY)
        ]

        if critical_fails or len(major_fails) >= 2:
            state = HealthState.FAULT
            culprits = major_fails  # superset of critical_fails; no contributors dropped
        elif fails or warns:
            state = HealthState.WARNING
            culprits = fails + warns
        else:
            state = HealthState.OK
            culprits = []

        if state is HealthState.OK:
            confidence = 1.0
            summary = f"OK: all {len(executed)} checks passed."
        else:
            confidence = (len(fails) + len(warns)) / len(executed)
            messages = []
            for r in culprits:
                if r.diagnostic_messages:
                    messages.extend(r.diagnostic_messages)
                else:
                    messages.append(f"{r.check_id} {r.status.value}")
            summary = f"{state.value}: " + "; ".join(messages)

    # Fold in calibration evaluation: the more severe verdict wins.
    if calibration_evaluation is not None:
        cal_state = _calibration_state(calibration_evaluation)
        if _SEVERITY_RANK[cal_state] > _SEVERITY_RANK[state]:
            state = cal_state
        if calibration_evaluation.summary:
            summary = (
                f"{summary} | {calibration_evaluation.summary}"
                if summary
                else calibration_evaluation.summary
            )

    return state, confidence, summary
```

- [ ] **Step 4: Pass the evaluation into `decide` from the pipeline**

In `app/health/pipeline.py`, change Stage 6:

```python
        final_state, confidence, summary = decide(results)
```

to:

```python
        final_state, confidence, summary = decide(results, calibration_evaluation)
```

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_fusion.py -q`
Expected: PASS (existing fusion tests + 5 new).

- [ ] **Step 6: Full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (~100 total: 88 prior + 5 calibration_eval + 2 pipeline + 5 fusion). Confirm the printed count is green with 0 failures.

- [ ] **Step 7: Headless end-to-end check**

```bash
.venv/bin/python -c "
import numpy as np
from app.health.calibration import generate_profile
from app.health.config import pipeline_for_profile
from app.health.models import AudioWindow, HealthState
SR, N = 44100, 110250
t = np.arange(int(6*SR))/SR
healthy = (0.3*np.sin(2*np.pi*1000*t)).astype(np.float32)
profile = generate_profile([healthy], SR, profile_id='p')
pipe = pipeline_for_profile('development', calibration_profile=profile)
# A window very unlike the calibrated 1kHz tone: loud 60Hz hum-ish + noise.
weird = (0.6*np.sin(2*np.pi*60*np.arange(N)/SR) + 0.2*np.random.default_rng(0).standard_normal(N)).astype(np.float32)
rep = pipe.analyze(AudioWindow(samples=weird, sample_rate=SR))
print('state:', rep.final_state.value, '| cal warn/fault:', rep.calibration_evaluation.warn_count, rep.calibration_evaluation.fault_count)
print('summary:', rep.diagnostic_summary)
"
```
Expected: prints a state (likely WARNING or FAULT) with non-zero calibration deviations and a summary naming the deviating calibrated measurements — demonstrating the calibration layer firing on an off-profile signal.

---

## Phase 3b Done

The pipeline now evaluates each window against a loaded `CalibrationProfile` and folds sensor-relative deviations into the final health state, additively (no profile → unchanged). Hand back to the owner for review, manual test, and commit. **Phase 3c** wires this into the UI (select/load a profile, show the active profile, a calibration column in the panel, "Generate profile" button).

---

## Self-Review

- **Spec coverage:** percentile deviation rule with warn band `[p5,p95]`+tolerance and fault beyond extended `[min,max]` (§3) — `_verdict`/`evaluate_calibration` (Task 1); `CalibrationEvaluation` with deviations + warn/fault counts + summary (§4) — Task 1; pipeline optional profile + Stage 4 (§5) — Task 2; config pass-through (§5) — Task 2 Step 4; fusion "more severe" with ≥2-fault→FAULT, 1-fault/any-warn→WARNING, no-profile→unchanged (§6) — Task 3; checks keep manual thresholds (§2) — untouched. Degenerate max==min handled via eps (§3) — `_verdict` + test. UI explicitly deferred to 3c (§7).
- **Placeholder scan:** no TBD/TODO; full code each step; concrete commands.
- **Type consistency:** `evaluate_calibration(results, profile, *, warn_margin_frac=0.1, fault_margin=0.5, eps=1e-9) -> CalibrationEvaluation`; `CalibrationEvaluation(deviations, warn_count, fault_count, summary)`; `MeasurementDeviation(check_id, measurement, value, p5, p95, verdict)`; `HealthAnalysisPipeline(manager=None, calibration_profile=None)`; `build_pipeline(config, calibration_profile=None)`; `pipeline_for_profile(name, calibration_profile=None)`; `decide(results, calibration_evaluation=None)`. `decide` duck-types the evaluation (reads `.warn_count`/`.fault_count`/`.summary`) so `fusion.py` needs no new import and stays decoupled. Stage 4 produces the evaluation that Stage 6 consumes.
- **Portability:** `calibration_eval.py` imports only stdlib + `app.health.models`. No import cycle (`fusion` doesn't import `calibration_eval`; `pipeline` imports both).
