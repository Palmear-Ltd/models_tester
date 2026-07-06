# Phase 5a — Startup Validator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A headless startup-validation engine: system-config checks + aggregation of a capture's `HealthReport`s into a single PASS / WARNING / FAIL `StartupResult`.

**Architecture:** A pure `app/health/startup.py` with `validate_system` (config errors/warnings), `aggregate_signal` (tally window states + per-check failure counts), `evaluate_startup` (apply the decision rule), and a `run_validation` convenience. Reports are duck-typed (`final_state`, `check_results[].status`/`.check_id`). No UI — that's Phase 5b.

**Tech Stack:** Python 3.13, stdlib (dataclasses, enum); pytest. `app/health/` stays NumPy/stdlib-only.

> **Commit policy:** Owner commits manually. **Do NOT run git commit/add/push.** End at "tests pass."

> **Reference spec:** `docs/superpowers/specs/2026-06-25-phase-5-startup-validation-design.md`. 5b (the "Validate Acquisition" button + 20 s capture) is the next plan.

> **Decision rule (§2):** FAIL if system errors OR no windows OR fault-fraction ≥ `fault_fraction` (0.25); WARNING if any WARNING/FAULT windows below that; PASS otherwise.

## Current State

- `app/health/models.py`: `HealthState` (OK/WARNING/FAULT/UNKNOWN); `CheckStatus` (PASS/WARNING/FAIL/NOT_EXECUTED); `HealthReport.final_state` and `.check_results` (list of `SignalCheckResult` with `.check_id`, `.status`).
- 117 tests pass.

## File Structure
- Create: `app/health/startup.py`, `tests/health/test_startup.py`.

---

## Task 1: System validation + signal aggregation

**Files:** Create `app/health/startup.py`; Test `tests/health/test_startup.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/health/test_startup.py`:

```python
from app.health.models import CheckStatus, HealthState
from app.health.startup import (
    SignalAggregate,
    SystemValidation,
    aggregate_signal,
    validate_system,
)


class _Check:
    def __init__(self, check_id, status):
        self.check_id = check_id
        self.status = status


class _Rep:
    def __init__(self, state, checks=()):
        self.final_state = state
        self.check_results = list(checks)


def test_system_ok_when_config_correct():
    v = validate_system(sample_rate=44100, input_ready=True, calibration_loaded=True)
    assert v.passed is True
    assert v.errors == []
    assert v.warnings == []


def test_system_errors_on_wrong_sample_rate_and_no_input():
    v = validate_system(sample_rate=22050, input_ready=False, calibration_loaded=True)
    assert v.passed is False
    assert len(v.errors) == 2


def test_missing_calibration_is_a_warning_not_an_error():
    v = validate_system(sample_rate=44100, input_ready=True, calibration_loaded=False)
    assert v.passed is True
    assert v.errors == []
    assert any("calibration" in w.lower() for w in v.warnings)


def test_aggregate_signal_counts_states_and_check_failures():
    reports = [
        _Rep(HealthState.OK, [_Check("T001", CheckStatus.PASS)]),
        _Rep(HealthState.WARNING, [_Check("T002", CheckStatus.WARNING)]),
        _Rep(HealthState.FAULT, [_Check("T001", CheckStatus.FAIL), _Check("T002", CheckStatus.WARNING)]),
        _Rep(HealthState.UNKNOWN, []),  # counted as fault for safety
    ]
    agg = aggregate_signal(reports)
    assert isinstance(agg, SignalAggregate)
    assert agg.total == 4
    assert agg.ok == 1
    assert agg.warning == 1
    assert agg.fault == 2  # FAULT + UNKNOWN
    assert agg.check_failures == {"T001": 1, "T002": 2}  # non-PASS occurrences


def test_aggregate_empty():
    agg = aggregate_signal([])
    assert agg.total == 0 and agg.ok == 0 and agg.check_failures == {}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_startup.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.health.startup'`.

- [ ] **Step 3: Implement the first half of `app/health/startup.py`**

```python
"""Startup Validation (spec Ch. 7): verify system config and aggregate a capture's
HealthReports into a single PASS / WARNING / FAIL decision.

Pure stdlib. Reports are duck-typed: each needs ``final_state`` (a HealthState) and
``check_results`` (items with ``check_id`` and ``status``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.health.models import CheckStatus, HealthState


@dataclass
class SystemValidation:
    passed: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def validate_system(
    *, sample_rate, input_ready, calibration_loaded, expected_sample_rate=44100
) -> SystemValidation:
    errors = []
    warnings = []
    if sample_rate != expected_sample_rate:
        errors.append(f"Sample rate {sample_rate} != expected {expected_sample_rate}")
    if not input_ready:
        errors.append("No input source ready")
    if not calibration_loaded:
        warnings.append("No calibration profile loaded")
    return SystemValidation(passed=not errors, errors=errors, warnings=warnings)


@dataclass
class SignalAggregate:
    total: int
    ok: int
    warning: int
    fault: int
    check_failures: dict = field(default_factory=dict)  # check_id -> non-PASS window count


def aggregate_signal(reports) -> SignalAggregate:
    ok = warning = fault = 0
    check_failures: dict = {}
    for report in reports:
        state = report.final_state
        if state is HealthState.OK:
            ok += 1
        elif state is HealthState.WARNING:
            warning += 1
        else:  # FAULT or UNKNOWN — count as fault for safety
            fault += 1
        for result in report.check_results:
            if result.status is not CheckStatus.PASS:
                check_failures[result.check_id] = check_failures.get(result.check_id, 0) + 1
    return SignalAggregate(
        total=len(reports), ok=ok, warning=warning, fault=fault, check_failures=check_failures
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_startup.py -q`
Expected: PASS (5 passed).

---

## Task 2: Decision (`evaluate_startup`, `run_validation`)

**Files:** Modify `app/health/startup.py`; Test `tests/health/test_startup.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/health/test_startup.py`:

```python
from app.health.startup import (  # noqa: E402
    StartupDecision,
    StartupResult,
    evaluate_startup,
    run_validation,
)


def _agg(total, ok, warning, fault):
    return SignalAggregate(total=total, ok=ok, warning=warning, fault=fault, check_failures={})


def _ok_system():
    return SystemValidation(passed=True, errors=[], warnings=[])


def test_pass_when_all_ok():
    r = evaluate_startup(_ok_system(), _agg(40, 40, 0, 0))
    assert r.decision is StartupDecision.PASS
    assert isinstance(r, StartupResult)


def test_warning_when_some_non_ok_below_fault_fraction():
    r = evaluate_startup(_ok_system(), _agg(40, 36, 3, 1))  # 1/40 fault < 0.25
    assert r.decision is StartupDecision.WARNING


def test_fail_when_fault_fraction_exceeded():
    r = evaluate_startup(_ok_system(), _agg(40, 20, 5, 15))  # 15/40 >= 0.25
    assert r.decision is StartupDecision.FAIL


def test_fail_when_system_errors():
    bad = SystemValidation(passed=False, errors=["No input source ready"], warnings=[])
    r = evaluate_startup(bad, _agg(40, 40, 0, 0))
    assert r.decision is StartupDecision.FAIL


def test_fail_when_no_windows():
    r = evaluate_startup(_ok_system(), _agg(0, 0, 0, 0))
    assert r.decision is StartupDecision.FAIL


def test_run_validation_end_to_end():
    reports = [_Rep(HealthState.OK, [_Check("T001", CheckStatus.PASS)]) for _ in range(40)]
    r = run_validation(reports, sample_rate=44100, input_ready=True, calibration_loaded=True)
    assert r.decision is StartupDecision.PASS
    assert r.signal.total == 40
    assert r.summary
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_startup.py -q`
Expected: FAIL — `ImportError: cannot import name 'StartupDecision'`.

- [ ] **Step 3: Append the decision logic to `app/health/startup.py`**

```python


class StartupDecision(Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class StartupResult:
    decision: StartupDecision
    system: SystemValidation
    signal: SignalAggregate
    summary: str


def evaluate_startup(system, signal, *, fault_fraction=0.25) -> StartupResult:
    if not system.passed:
        decision = StartupDecision.FAIL
    elif signal.total == 0:
        decision = StartupDecision.FAIL
    elif signal.fault / signal.total >= fault_fraction:
        decision = StartupDecision.FAIL
    elif (signal.warning + signal.fault) > 0:
        decision = StartupDecision.WARNING
    else:
        decision = StartupDecision.PASS

    parts = [
        f"{signal.ok} OK / {signal.warning} WARNING / {signal.fault} FAULT "
        f"of {signal.total} windows"
    ]
    if system.errors:
        parts.append("system: " + "; ".join(system.errors))
    if system.warnings:
        parts.append("; ".join(system.warnings))
    if signal.check_failures:
        top = sorted(signal.check_failures.items(), key=lambda kv: -kv[1])[:3]
        parts.append("top: " + ", ".join(f"{cid}×{n}" for cid, n in top))
    summary = f"{decision.value}: " + " | ".join(parts)

    return StartupResult(decision=decision, system=system, signal=signal, summary=summary)


def run_validation(
    reports, *, sample_rate, input_ready, calibration_loaded, fault_fraction=0.25
) -> StartupResult:
    system = validate_system(
        sample_rate=sample_rate, input_ready=input_ready, calibration_loaded=calibration_loaded
    )
    signal = aggregate_signal(reports)
    return evaluate_startup(system, signal, fault_fraction=fault_fraction)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_startup.py -q`
Expected: PASS (11 passed).

- [ ] **Step 5: Full suite + import purity**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (128 total: 117 + 11 new).
Run: `grep -nE "^import |^from " app/health/startup.py`
Expected: only stdlib (`dataclasses`, `enum`) + `app.health.models` (no NumPy/UI).

- [ ] **Step 6: Headless demo**

```bash
.venv/bin/python -c "
from app.health.models import CheckStatus, HealthState
from app.health.startup import run_validation
class C:
    def __init__(s,i,st): s.check_id=i; s.status=st
class R:
    def __init__(s,st,c=()): s.final_state=st; s.check_results=list(c)
good = [R(HealthState.OK, [C('T001',CheckStatus.PASS)]) for _ in range(40)]
bad = [R(HealthState.FAULT, [C('T001',CheckStatus.FAIL)]) for _ in range(40)]
print('good, no cal:', run_validation(good, sample_rate=44100, input_ready=True, calibration_loaded=False).summary)
print('faulty:', run_validation(bad, sample_rate=44100, input_ready=True, calibration_loaded=True).summary)
"
```
Expected: the good capture → `PASS: 40 OK / 0 WARNING / 0 FAULT of 40 windows | No calibration profile loaded` (the missing-calibration note is a non-blocking *system warning* that appears in the summary but does not change the PASS decision, since the signal is clean). The faulty capture → `FAIL: 0 OK / 0 WARNING / 40 FAULT of 40 windows | top: T001×40`.

---

## Phase 5a Done

The headless startup-validation engine is complete: `run_validation(reports, sample_rate=, input_ready=, calibration_loaded=)` → a `StartupResult` with a PASS/WARNING/FAIL decision, window tally, and summary. Hand back to the owner for review, manual test, and commit. **Phase 5b** adds the "Validate Acquisition" button: a bounded ~20 s capture that feeds the pipeline and shows this result.

---

## Self-Review

- **Spec coverage:** system validation (sample rate / input / calibration) → `validate_system` (§3) — Task 1; signal aggregation (window tally + per-check failures, UNKNOWN→fault) → `aggregate_signal` (§3) — Task 1; decision rule (system errors / no windows / fault-fraction → FAIL; some non-OK → WARNING; else PASS) → `evaluate_startup` (§2) — Task 2; convenience `run_validation` (§3) — Task 2. Missing-calibration is a non-blocking warning (§3). UI deferred to 5b (§4).
- **Placeholder scan:** no TBD/TODO; full code each step; concrete commands.
- **Type consistency:** `validate_system(*, sample_rate, input_ready, calibration_loaded, expected_sample_rate=44100) -> SystemValidation(passed, errors, warnings)`; `aggregate_signal(reports) -> SignalAggregate(total, ok, warning, fault, check_failures)`; `evaluate_startup(system, signal, *, fault_fraction=0.25) -> StartupResult(decision, system, signal, summary)`; `run_validation(reports, *, sample_rate, input_ready, calibration_loaded, fault_fraction=0.25)`; `StartupDecision` PASS/WARNING/FAIL. Reports duck-typed (`final_state`, `check_results[].status`/`.check_id`) — no `HealthReport` import needed; `HealthState`/`CheckStatus` imported for comparisons.
