# Phase 1 — Time-Domain Checks & Decision Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the seven time-domain Signal Health Checks (T001–T007) plus rule-based Decision Fusion so the tester's health indicator shows a real OK / WARNING / FAULT verdict (with diagnostics in the log) instead of the Phase 0 `UNKNOWN`.

**Architecture:** Each check is an independent pure-NumPy `SignalHealthCheck` in `app/health/checks/time_domain.py`, with manual default thresholds (calibration-derived thresholds arrive in Phase 3). Checks are tagged with a `CheckCategory` (CRITICAL/PRIMARY/SUPPORTING); the rewritten `decide()` in `fusion.py` combines their results into a final `HealthState` using the spec's Chapter 10 decision rules. A `defaults.py` factory wires all seven into the pipeline, and `main.py` switches to that populated pipeline and logs the diagnostic summary on each state change.

**Tech Stack:** Python 3.13, NumPy 2.x, pytest. Run via `.venv/bin/python -m pytest`. Tkinter (existing UI, unchanged behavior).

> **Commit policy for this repo:** The owner commits/pushes **manually** after reviewing and testing each phase. **Do NOT run `git commit`/`git add`/`git push`.** Each task ends at "tests pass."

> **References:** spec `docs/superpowers/specs/2026-06-23-audio-signal-health-monitoring-phased-implementation-design.md` (§5 Phase 1); `arch_update.md` §4.8 (T001–T007), §10 (Decision Strategy). Phase 0 plan: `docs/superpowers/plans/2026-06-23-phase-0-foundation-and-seam.md`.

> **Note on thresholds:** All threshold values below are **provisional manual defaults** for float32 audio in the range [-1, 1]. They are intentionally conservative so a healthy signal reads OK and only genuinely bad signals (silence, clipping) trip the critical checks. Phase 3 replaces them with calibration-derived values.

---

## Current State (end of Phase 0)

- `app/health/models.py`: `HealthState`(OK/WARNING/FAULT/UNKNOWN), `CheckStatus`(PASS/WARNING/FAIL/NOT_EXECUTED), `Measurement`(name,value,unit), immutable `AudioWindow`, `SignalCheckResult`(check_id,check_name,status,executed,execution_time,measurements,diagnostic_messages), `HealthReport`.
- `app/health/checks/base.py`: abstract `SignalHealthCheck` with class attrs `check_id`/`check_name` and abstractmethod `run(window, features) -> SignalCheckResult`.
- `app/health/manager.py`: `SignalCheckManager` — `register` (rejects empty `check_id`), `checks` property, `run_checks` (isolates failures, stamps `executed`/`execution_time`).
- `app/health/fusion.py`: minimal `decide(results) -> (HealthState, float, str)` returning UNKNOWN.
- `app/health/pipeline.py`: `HealthAnalysisPipeline(manager=None)`, `analyze(window) -> HealthReport`, 7 stage methods.
- `main.py`: builds `AudioWindow` from `session_buffer` each chunk, runs `self.health_pipeline.analyze(...)` in try/except, updates `self.health_label` via `_update_health_indicator`.
- 15 tests passing in `tests/health/`.

## File Structure

**Modify:**
- `app/health/models.py` — add `CheckCategory` enum; add `category` field to `SignalCheckResult`.
- `app/health/__init__.py` — export `CheckCategory`.
- `app/health/checks/base.py` — add `category` class attr.
- `app/health/manager.py` — stamp `result.category` from the check (both paths).
- `app/health/fusion.py` — rewrite `decide()` with Chapter 10 rules.
- `main.py` — use populated pipeline; log diagnostic summary on health-state change.

**Create:**
- `app/health/checks/time_domain.py` — `FlatlineCheck`, `SignalEnergyCheck`, `PeakAmplitudeCheck`, `ClippingCheck`, `CrestFactorCheck`, `DCOffsetCheck`, `ZeroCrossingRateCheck`.
- `app/health/defaults.py` — `default_time_domain_checks()`, `default_manager()`, `default_pipeline()`.
- `tests/health/test_time_domain.py`, `tests/health/test_fusion.py`, `tests/health/test_defaults.py`.

---

## Task 1: CheckCategory + result category stamping

**Files:**
- Modify: `app/health/models.py`, `app/health/__init__.py`, `app/health/checks/base.py`, `app/health/manager.py`
- Test: `tests/health/test_manager.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/health/test_manager.py`:

```python
from app.health.models import CheckCategory


class _CategorizedCheck(SignalHealthCheck):
    check_id = "T100"
    check_name = "Categorized"
    category = CheckCategory.CRITICAL

    def run(self, window, features):
        return SignalCheckResult(check_id=self.check_id, check_name=self.check_name)


def test_manager_stamps_category_on_success():
    manager = SignalCheckManager()
    manager.register(_CategorizedCheck())
    results = manager.run_checks(_window(), {})
    assert results[0].category is CheckCategory.CRITICAL


def test_default_check_category_is_primary():
    # A check that does not override `category` defaults to PRIMARY.
    assert _PassingCheck().category is CheckCategory.PRIMARY
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_manager.py -q`
Expected: FAIL with `ImportError: cannot import name 'CheckCategory'`.

- [ ] **Step 3: Add `CheckCategory` enum and `category` field**

In `app/health/models.py`, add this enum immediately after the `CheckStatus` enum (after its `NOT_EXECUTED = "NOT_EXECUTED"` line):

```python


class CheckCategory(Enum):
    """Importance of a check for Decision Fusion (spec Ch. 10.3)."""

    CRITICAL = "CRITICAL"
    PRIMARY = "PRIMARY"
    SUPPORTING = "SUPPORTING"
```

In `app/health/models.py`, in the `SignalCheckResult` dataclass, add a `category` field. Change:

```python
    status: CheckStatus = CheckStatus.NOT_EXECUTED
    executed: bool = False
```

to:

```python
    status: CheckStatus = CheckStatus.NOT_EXECUTED
    category: CheckCategory = CheckCategory.PRIMARY
    executed: bool = False
```

- [ ] **Step 4: Export `CheckCategory`**

In `app/health/__init__.py`, add `CheckCategory` to both the import block and `__all__` (keep alphabetical order):

```python
"""Audio Signal Health Monitoring subsystem (NumPy-only, UI-agnostic)."""
from app.health.models import (
    AudioWindow,
    CheckCategory,
    CheckStatus,
    HealthReport,
    HealthState,
    Measurement,
    SignalCheckResult,
)

__all__ = [
    "AudioWindow",
    "CheckCategory",
    "CheckStatus",
    "HealthReport",
    "HealthState",
    "Measurement",
    "SignalCheckResult",
]
```

- [ ] **Step 5: Add `category` class attr to the check base**

In `app/health/checks/base.py`, update the imports and class attrs. Change:

```python
from app.health.models import AudioWindow, SignalCheckResult


class SignalHealthCheck(ABC):
    """Evaluates exactly one property of an Audio Window.

    Subclasses set ``check_id`` / ``check_name`` and implement ``run``. Checks must
    not depend on one another; the manager runs each in isolation.
    """

    check_id: str = ""
    check_name: str = ""
```

to:

```python
from app.health.models import AudioWindow, CheckCategory, SignalCheckResult


class SignalHealthCheck(ABC):
    """Evaluates exactly one property of an Audio Window.

    Subclasses set ``check_id`` / ``check_name`` / ``category`` and implement
    ``run``. Checks must not depend on one another; the manager runs each in
    isolation.
    """

    check_id: str = ""
    check_name: str = ""
    category: CheckCategory = CheckCategory.PRIMARY
```

- [ ] **Step 6: Stamp the category in the manager**

In `app/health/manager.py`, update both result paths in `run_checks`. Change the success line:

```python
                result = check.run(window, features)
                result.executed = True
```

to:

```python
                result = check.run(window, features)
                result.executed = True
                result.category = check.category
```

And in the `except` branch, add `category=check.category,` to the `SignalCheckResult(...)` constructor (after the `check_name=...` argument):

```python
                result = SignalCheckResult(
                    check_id=getattr(check, "check_id", "") or check.__class__.__name__,
                    check_name=getattr(check, "check_name", "")
                    or check.__class__.__name__,
                    category=check.category,
                    status=CheckStatus.NOT_EXECUTED,
                    executed=False,
                    diagnostic_messages=[f"Check raised: {exc}"],
                )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/health/ -q`
Expected: PASS (17 passed — 15 prior + 2 new).

---

## Task 2: Critical time-domain checks (Flatline, Signal Energy, Clipping)

**Files:**
- Create: `app/health/checks/time_domain.py`
- Test: `tests/health/test_time_domain.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/health/test_time_domain.py`:

```python
import numpy as np

from app.health.checks.time_domain import (
    ClippingCheck,
    FlatlineCheck,
    SignalEnergyCheck,
)
from app.health.models import AudioWindow, CheckCategory, CheckStatus

SR = 44100
N = 110250  # 2.5 s


def _win(x):
    return AudioWindow(samples=np.asarray(x, dtype=np.float32), sample_rate=SR)


def _sine(freq=1000.0, n=N, amp=0.3):
    t = np.arange(n) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def _measure(result, name):
    return next(m.value for m in result.measurements if m.name == name)


def test_flatline_fails_on_silence():
    check = FlatlineCheck()
    assert check.category is CheckCategory.CRITICAL
    result = check.run(_win(np.zeros(N)), {})
    assert result.status is CheckStatus.FAIL
    assert result.diagnostic_messages


def test_flatline_passes_on_sine():
    result = FlatlineCheck().run(_win(_sine()), {})
    assert result.status is CheckStatus.PASS


def test_signal_energy_fails_on_silence():
    result = SignalEnergyCheck().run(_win(np.zeros(N)), {})
    assert result.status is CheckStatus.FAIL


def test_signal_energy_passes_on_sine():
    result = SignalEnergyCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS
    assert _measure(result, "rms") > 0.1


def test_clipping_fails_when_saturated():
    result = ClippingCheck().run(_win(np.ones(N)), {})
    assert result.status is CheckStatus.FAIL
    assert _measure(result, "clipping_ratio") == 1.0


def test_clipping_passes_on_clean_sine():
    result = ClippingCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_time_domain.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.health.checks.time_domain'`.

- [ ] **Step 3: Implement the critical checks**

Create `app/health/checks/time_domain.py`:

```python
"""Time-domain Signal Health Checks (spec §4.8, T001–T007).

Each check operates directly on the waveform (NumPy only) and reports a status
plus measurements. Thresholds are provisional manual defaults; Phase 3 replaces
them with calibration-derived values.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.health.checks.base import SignalHealthCheck
from app.health.models import (
    AudioWindow,
    CheckCategory,
    CheckStatus,
    Measurement,
    SignalCheckResult,
)


class FlatlineCheck(SignalHealthCheck):
    """T001 — detect complete loss of signal (disconnected sensor, dead cable)."""

    check_id = "T001"
    check_name = "Flatline Detection"
    category = CheckCategory.CRITICAL

    def __init__(self, min_std: float = 1e-5, min_peak_to_peak: float = 1e-4):
        self.min_std = min_std
        self.min_peak_to_peak = min_peak_to_peak

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        x = window.samples
        std = float(np.std(x))
        p2p = float(np.ptp(x)) if x.size else 0.0
        diagnostics = []
        if std < self.min_std or p2p < self.min_peak_to_peak:
            status = CheckStatus.FAIL
            diagnostics.append(
                f"Flatline: std={std:.2e}, peak-to-peak={p2p:.2e} below minimum"
            )
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[
                Measurement("std", std),
                Measurement("peak_to_peak", p2p),
            ],
            diagnostic_messages=diagnostics,
        )


class SignalEnergyCheck(SignalHealthCheck):
    """T002 — verify the signal carries enough (but not excessive) energy."""

    check_id = "T002"
    check_name = "Signal Energy"
    category = CheckCategory.CRITICAL

    def __init__(
        self,
        min_rms_fault: float = 1e-4,
        min_rms_warn: float = 1e-3,
        max_rms_warn: float = 0.7,
        max_rms_fault: float = 0.9,
    ):
        self.min_rms_fault = min_rms_fault
        self.min_rms_warn = min_rms_warn
        self.max_rms_warn = max_rms_warn
        self.max_rms_fault = max_rms_fault

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        x = window.samples
        rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
        diagnostics = []
        if rms < self.min_rms_fault or rms > self.max_rms_fault:
            status = CheckStatus.FAIL
            diagnostics.append(f"RMS energy {rms:.2e} outside acceptable range")
        elif rms < self.min_rms_warn or rms > self.max_rms_warn:
            status = CheckStatus.WARNING
            diagnostics.append(f"RMS energy {rms:.2e} outside expected range")
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[Measurement("rms", rms)],
            diagnostic_messages=diagnostics,
        )


class ClippingCheck(SignalHealthCheck):
    """T004 — detect ADC/amplifier saturation via the fraction of clipped samples."""

    check_id = "T004"
    check_name = "Clipping Detection"
    category = CheckCategory.CRITICAL

    def __init__(
        self,
        clipping_threshold: float = 0.99,
        warning_ratio: float = 0.001,
        fault_ratio: float = 0.01,
    ):
        self.clipping_threshold = clipping_threshold
        self.warning_ratio = warning_ratio
        self.fault_ratio = fault_ratio

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        x = window.samples
        n = x.size
        clipped = int(np.count_nonzero(np.abs(x) >= self.clipping_threshold))
        ratio = clipped / n if n else 0.0
        diagnostics = []
        if ratio >= self.fault_ratio:
            status = CheckStatus.FAIL
            diagnostics.append(f"Clipping ratio {ratio:.3%} indicates saturation")
        elif ratio >= self.warning_ratio:
            status = CheckStatus.WARNING
            diagnostics.append(f"Clipping ratio {ratio:.3%} elevated")
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[
                Measurement("clipping_ratio", ratio),
                Measurement("clipped_samples", float(clipped)),
            ],
            diagnostic_messages=diagnostics,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/health/test_time_domain.py -q`
Expected: PASS (6 passed).

---

## Task 3: Remaining time-domain checks (Peak, Crest, DC Offset, ZCR)

**Files:**
- Modify: `app/health/checks/time_domain.py` (append four classes)
- Test: `tests/health/test_time_domain.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/health/test_time_domain.py`:

```python
from app.health.checks.time_domain import (  # noqa: E402
    CrestFactorCheck,
    DCOffsetCheck,
    PeakAmplitudeCheck,
    ZeroCrossingRateCheck,
)


def test_peak_amplitude_passes_on_sine():
    result = PeakAmplitudeCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS
    assert _measure(result, "peak_amplitude") > 0.25


def test_peak_amplitude_warns_when_too_small():
    result = PeakAmplitudeCheck().run(_win(_sine(amp=1e-4)), {})
    assert result.status is CheckStatus.WARNING


def test_crest_factor_passes_on_sine():
    # A sine has crest factor ~1.41, inside the default [1.2, 50] band.
    result = CrestFactorCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS


def test_crest_factor_category_is_supporting():
    assert CrestFactorCheck().category is CheckCategory.SUPPORTING


def test_dc_offset_fails_on_large_bias():
    result = DCOffsetCheck().run(_win(_sine(amp=0.3) + 0.3), {})
    assert result.status is CheckStatus.FAIL
    assert abs(_measure(result, "dc_offset") - 0.3) < 0.01


def test_dc_offset_passes_on_centered_signal():
    result = DCOffsetCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS


def test_zcr_warns_on_alternating_signal():
    # Sign flips every sample -> ZCR ~1.0, above the default 0.8 warning bound.
    alt = np.tile([0.3, -0.3], N // 2).astype(np.float32)
    result = ZeroCrossingRateCheck().run(_win(alt), {})
    assert result.status is CheckStatus.WARNING


def test_zcr_passes_on_sine():
    result = ZeroCrossingRateCheck().run(_win(_sine(freq=1000.0, amp=0.3)), {})
    assert result.status is CheckStatus.PASS
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_time_domain.py -q`
Expected: FAIL with `ImportError: cannot import name 'CrestFactorCheck'`.

- [ ] **Step 3: Append the four remaining checks**

Append to `app/health/checks/time_domain.py`:

```python


class PeakAmplitudeCheck(SignalHealthCheck):
    """T003 — verify the peak amplitude is within the expected operating range."""

    check_id = "T003"
    check_name = "Peak Amplitude"
    category = CheckCategory.PRIMARY

    def __init__(self, min_peak: float = 1e-3, max_peak: float = 0.99):
        self.min_peak = min_peak
        self.max_peak = max_peak

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        x = window.samples
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        diagnostics = []
        if peak < self.min_peak or peak > self.max_peak:
            status = CheckStatus.WARNING
            diagnostics.append(f"Peak amplitude {peak:.3e} outside expected range")
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[Measurement("peak_amplitude", peak)],
            diagnostic_messages=diagnostics,
        )


class CrestFactorCheck(SignalHealthCheck):
    """T005 — evaluate waveform dynamics via peak-to-RMS ratio."""

    check_id = "T005"
    check_name = "Crest Factor"
    category = CheckCategory.SUPPORTING

    def __init__(self, min_crest: float = 1.2, max_crest: float = 50.0):
        self.min_crest = min_crest
        self.max_crest = max_crest

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        x = window.samples
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
        crest = peak / rms if rms > 0 else 0.0
        diagnostics = []
        # A zero RMS (dead signal) is handled by FlatlineCheck; skip here.
        if rms > 0 and (crest < self.min_crest or crest > self.max_crest):
            status = CheckStatus.WARNING
            diagnostics.append(f"Crest factor {crest:.2f} outside expected range")
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[Measurement("crest_factor", crest)],
            diagnostic_messages=diagnostics,
        )


class DCOffsetCheck(SignalHealthCheck):
    """T006 — detect a constant acquisition bias (DC offset)."""

    check_id = "T006"
    check_name = "DC Offset"
    category = CheckCategory.PRIMARY

    def __init__(
        self, max_dc_offset_warn: float = 0.02, max_dc_offset_fault: float = 0.1
    ):
        self.max_dc_offset_warn = max_dc_offset_warn
        self.max_dc_offset_fault = max_dc_offset_fault

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        x = window.samples
        dc = float(np.mean(x)) if x.size else 0.0
        magnitude = abs(dc)
        diagnostics = []
        if magnitude > self.max_dc_offset_fault:
            status = CheckStatus.FAIL
            diagnostics.append(f"DC offset {dc:.3f} exceeds fault limit")
        elif magnitude > self.max_dc_offset_warn:
            status = CheckStatus.WARNING
            diagnostics.append(f"DC offset {dc:.3f} elevated")
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[Measurement("dc_offset", dc)],
            diagnostic_messages=diagnostics,
        )


class ZeroCrossingRateCheck(SignalHealthCheck):
    """T007 — flag excessive high-frequency content via zero-crossing rate."""

    check_id = "T007"
    check_name = "Zero Crossing Rate"
    category = CheckCategory.SUPPORTING

    def __init__(self, max_zcr: float = 0.8):
        self.max_zcr = max_zcr

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        x = window.samples
        if x.size < 2:
            zcr = 0.0
        else:
            signs = np.signbit(x)
            crossings = int(np.count_nonzero(np.diff(signs)))
            zcr = crossings / (x.size - 1)
        diagnostics = []
        if zcr > self.max_zcr:
            status = CheckStatus.WARNING
            diagnostics.append(f"Zero-crossing rate {zcr:.2f} abnormally high")
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[Measurement("zero_crossing_rate", zcr)],
            diagnostic_messages=diagnostics,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/health/test_time_domain.py -q`
Expected: PASS (14 passed — 6 from Task 2 + 8 new).

---

## Task 4: Rule-based Decision Fusion

**Files:**
- Modify: `app/health/fusion.py`
- Test: `tests/health/test_fusion.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/health/test_fusion.py`:

```python
from app.health.fusion import decide
from app.health.models import CheckCategory, CheckStatus, HealthState, SignalCheckResult


def _res(cid, status, category, executed=True, diag=None):
    return SignalCheckResult(
        check_id=cid,
        check_name=cid,
        status=status,
        category=category,
        executed=executed,
        diagnostic_messages=list(diag) if diag else [],
    )


def test_empty_results_is_unknown():
    state, confidence, summary = decide([])
    assert state is HealthState.UNKNOWN
    assert confidence == 0.0
    assert summary


def test_only_unexecuted_is_unknown():
    r = _res("T001", CheckStatus.NOT_EXECUTED, CheckCategory.CRITICAL, executed=False)
    assert decide([r])[0] is HealthState.UNKNOWN


def test_all_pass_is_ok_full_confidence():
    rs = [
        _res("T001", CheckStatus.PASS, CheckCategory.CRITICAL),
        _res("T003", CheckStatus.PASS, CheckCategory.PRIMARY),
    ]
    state, confidence, summary = decide(rs)
    assert state is HealthState.OK
    assert confidence == 1.0


def test_critical_fail_is_fault():
    rs = [
        _res("T001", CheckStatus.FAIL, CheckCategory.CRITICAL, diag=["flatline"]),
        _res("T003", CheckStatus.PASS, CheckCategory.PRIMARY),
    ]
    assert decide(rs)[0] is HealthState.FAULT


def test_two_major_fails_is_fault():
    rs = [
        _res("T003", CheckStatus.FAIL, CheckCategory.PRIMARY),
        _res("T006", CheckStatus.FAIL, CheckCategory.PRIMARY),
    ]
    assert decide(rs)[0] is HealthState.FAULT


def test_single_primary_fail_is_warning():
    rs = [
        _res("T003", CheckStatus.FAIL, CheckCategory.PRIMARY),
        _res("T001", CheckStatus.PASS, CheckCategory.CRITICAL),
    ]
    assert decide(rs)[0] is HealthState.WARNING


def test_supporting_warning_is_warning():
    rs = [
        _res("T005", CheckStatus.WARNING, CheckCategory.SUPPORTING),
        _res("T001", CheckStatus.PASS, CheckCategory.CRITICAL),
    ]
    assert decide(rs)[0] is HealthState.WARNING


def test_summary_includes_diagnostics():
    rs = [_res("T001", CheckStatus.FAIL, CheckCategory.CRITICAL, diag=["Flatline: dead"])]
    state, confidence, summary = decide(rs)
    assert state is HealthState.FAULT
    assert "Flatline" in summary
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_fusion.py -q`
Expected: FAIL (e.g. `test_all_pass_is_ok_full_confidence` fails because the current `decide()` always returns UNKNOWN).

- [ ] **Step 3: Rewrite `decide()`**

Replace the entire contents of `app/health/fusion.py` with:

```python
"""Decision Fusion — combines Signal Check Results into a final health state.

Implements the spec's Chapter 10 decision rules:
- any CRITICAL-category check failing, or two or more CRITICAL/PRIMARY checks
  failing, -> FAULT
- any other failure or warning -> WARNING
- all executed checks passing -> OK
- no executed checks -> UNKNOWN

Confidence is a Phase 1 heuristic: the fraction of executed checks that agree
with the chosen state (passes for OK, non-passes for WARNING/FAULT).
"""
from __future__ import annotations

from app.health.models import CheckCategory, CheckStatus, HealthState, SignalCheckResult


def decide(results: list[SignalCheckResult]) -> tuple[HealthState, float, str]:
    """Return (final_state, confidence, diagnostic_summary)."""
    executed = [r for r in results if r.executed]
    if not executed:
        return HealthState.UNKNOWN, 0.0, "No signal health checks executed."

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
        culprits = critical_fails or major_fails
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

    return state, confidence, summary
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/health/test_fusion.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Confirm the pipeline tests still hold**

Run: `.venv/bin/python -m pytest tests/health/test_pipeline.py -q`
Expected: PASS (3 passed — the no-check pipeline still returns UNKNOWN since no checks are registered there).

---

## Task 5: Default pipeline factory

**Files:**
- Create: `app/health/defaults.py`
- Test: `tests/health/test_defaults.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/health/test_defaults.py`:

```python
import numpy as np

from app.health.defaults import default_manager, default_pipeline
from app.health.models import AudioWindow, HealthState

SR = 44100
N = 110250


def _win(x):
    return AudioWindow(samples=np.asarray(x, dtype=np.float32), sample_rate=SR)


def test_default_manager_registers_seven_checks():
    assert len(default_manager().checks) == 7


def test_default_pipeline_ok_on_clean_sine():
    t = np.arange(N) / SR
    sig = 0.3 * np.sin(2 * np.pi * 1000.0 * t)
    report = default_pipeline().analyze(_win(sig))
    assert report.final_state is HealthState.OK
    assert len(report.check_results) == 7


def test_default_pipeline_fault_on_silence():
    report = default_pipeline().analyze(_win(np.zeros(N)))
    assert report.final_state is HealthState.FAULT
    assert report.diagnostic_summary
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_defaults.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.health.defaults'`.

- [ ] **Step 3: Implement the factory**

Create `app/health/defaults.py`:

```python
"""Factories that assemble the standard Phase 1 health-monitoring pipeline.

Phase 2 replaces this with the configuration-profile system; for now it simply
registers the seven time-domain checks with their default manual thresholds.
"""
from __future__ import annotations

from app.health.checks.base import SignalHealthCheck
from app.health.checks.time_domain import (
    ClippingCheck,
    CrestFactorCheck,
    DCOffsetCheck,
    FlatlineCheck,
    PeakAmplitudeCheck,
    SignalEnergyCheck,
    ZeroCrossingRateCheck,
)
from app.health.manager import SignalCheckManager
from app.health.pipeline import HealthAnalysisPipeline


def default_time_domain_checks() -> list[SignalHealthCheck]:
    """The seven time-domain checks (T001–T007) with default thresholds."""
    return [
        FlatlineCheck(),
        SignalEnergyCheck(),
        PeakAmplitudeCheck(),
        ClippingCheck(),
        CrestFactorCheck(),
        DCOffsetCheck(),
        ZeroCrossingRateCheck(),
    ]


def default_manager() -> SignalCheckManager:
    manager = SignalCheckManager()
    for check in default_time_domain_checks():
        manager.register(check)
    return manager


def default_pipeline() -> HealthAnalysisPipeline:
    return HealthAnalysisPipeline(manager=default_manager())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/health/test_defaults.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the whole health suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (42 total: 8 models + 6 manager + 3 pipeline + 14 time_domain + 8 fusion + 3 defaults). Confirm the printed count is green with 0 failures.

---

## Task 6: Wire the populated pipeline into the tester + state-change logging

**Files:**
- Modify: `main.py`

> Locate anchors by their quoted text, not line numbers (earlier edits shift later lines). Verify each anchor before editing.

- [ ] **Step 1: Switch the import to the default pipeline factory**

In `main.py`, find:

```python
from app.health.pipeline import HealthAnalysisPipeline
from app.health.models import AudioWindow, HealthState
```

Replace those two lines with:

```python
from app.health.defaults import default_pipeline
from app.health.models import AudioWindow, HealthState
```

- [ ] **Step 2: Use the populated pipeline and track last state**

In `main.py` `__init__`, find:

```python
        self.health_pipeline = HealthAnalysisPipeline()
        self.latest_health_report = None
```

Replace with:

```python
        self.health_pipeline = default_pipeline()
        self.latest_health_report = None
        self._last_health_state = None
```

- [ ] **Step 3: Log the diagnostic summary on state change**

In `main.py`, find the `_update_health_indicator` method:

```python
        state = report.final_state
        self.health_label.configure(
            text=f"Signal Health: {state.value}",
            foreground=colors.get(state, "gray"),
        )
```

Replace that block with:

```python
        state = report.final_state
        self.health_label.configure(
            text=f"Signal Health: {state.value}",
            foreground=colors.get(state, "gray"),
        )
        # Log only on transitions to avoid flooding the log every 0.5s.
        if state != self._last_health_state:
            self._last_health_state = state
            self.log(f"Signal health {state.value}: {report.diagnostic_summary}")
```

- [ ] **Step 4: Verify main.py parses and the change is additive**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('main.py parses OK')"`
Expected: `main.py parses OK`

Run: `git diff --stat main.py`
Expected: a small diff — 2 import lines changed, ~4 lines added in `__init__` and `_update_health_indicator`. Confirm no inference/feature/model logic was touched.

- [ ] **Step 5: Headless end-to-end check**

Run:
```bash
.venv/bin/python -c "
import numpy as np
from app.health.defaults import default_pipeline
from app.health.models import AudioWindow, HealthState
SR, N = 44100, 110250
pipe = default_pipeline()
t = np.arange(N)/SR
sine = (0.3*np.sin(2*np.pi*1000*t)).astype(np.float32)
ok = pipe.analyze(AudioWindow(samples=sine, sample_rate=SR))
bad = pipe.analyze(AudioWindow(samples=np.zeros(N, dtype=np.float32), sample_rate=SR))
print('sine ->', ok.final_state.value, '| silence ->', bad.final_state.value, '|', bad.diagnostic_summary)
assert ok.final_state is HealthState.OK
assert bad.final_state is HealthState.FAULT
print('end-to-end OK')
"
```
Expected: `sine -> OK | silence -> FAULT | FAULT: ...` then `end-to-end OK`.

- [ ] **Step 6: Confirm the full suite still passes**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass, 0 failures.

- [ ] **Step 7: Manual GUI verification (owner)**

(Requires full deps: `uv pip install --python .venv/bin/python -r requirements.txt`.) Launch `./.venv/bin/python launcher.py`. Start a mic test: with a live signal the indicator should read **OK** (green); cover/disconnect the sensor or feed silence and it should turn **FAULT** (red) with a `Signal health FAULT: ...` line in the log. Classification behavior must be unchanged.

---

## Phase 1 Done

The health indicator now shows real OK/WARNING/FAULT verdicts from seven time-domain checks, with diagnostics logged on each transition. Hand back to the owner for review, manual test, and commit. Phase 2 adds frequency-domain checks, a shared Feature Preparation stage, the configuration-profile system, and a dedicated health panel.

---

## Self-Review

- **Spec coverage (§4.8 / §10):** T001 Flatline, T002 Signal Energy, T003 Peak Amplitude, T004 Clipping, T005 Crest Factor, T006 DC Offset, T007 ZCR — all in Task 2/3. Decision rules (Ch. 10.5), check categories (Ch. 10.3 critical/primary/supporting), confidence (Ch. 10.8), explainable diagnostics (Ch. 10.10) — Task 4. Manual thresholds (Ch. 4.7.5) — per-check `__init__` defaults. Indicator + diagnostic logging (Phase 1 visible feature) — Task 6. Mandatory-check enforcement and the full config hierarchy are intentionally deferred to Phase 2.
- **Placeholder scan:** no TBD/TODO; every step has full code and exact commands. Thresholds are concrete numeric defaults.
- **Type consistency:** `decide(results) -> (HealthState, float, str)` unchanged; `SignalCheckResult` gains `category: CheckCategory` (Task 1) used by fusion (Task 4) and stamped by the manager (Task 1); every check returns `SignalCheckResult` with `check_id`/`check_name`/`status`/`measurements`/`diagnostic_messages`; `default_pipeline()` used in `main.py` (Task 6) is defined in Task 5; check class names match between `time_domain.py`, `defaults.py`, and the tests.
- **End-to-end sanity:** clean 0.3-amp 1 kHz sine → all seven PASS → OK; silence → Flatline + Signal Energy (both CRITICAL) FAIL → FAULT. Verified by the Task 5 and Task 6 checks.
