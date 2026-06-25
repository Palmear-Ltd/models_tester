# Phase 4a — History Mechanism & Stability Checks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the pipeline a bounded cross-window measurement history and add the three stability checks (S001–S003) that read it to flag intermittent/gradual changes.

**Architecture:** `HealthAnalysisPipeline` keeps a `deque` of per-window measurement snapshots; each `analyze` injects the prior snapshots as `features["history"]` and appends the current window after checks run. Stateless `stability.py` checks read that history. The three checks are registered in `config.REGISTRY` under a new `stability` category. Pure NumPy; headless.

**Tech Stack:** Python 3.13, NumPy, stdlib `collections.deque`; pytest. `app/health/` stays NumPy-only.

> **Commit policy:** Owner commits manually. **Do NOT run git commit/add/push.** End at "tests pass."

> **Reference spec:** `docs/superpowers/specs/2026-06-25-phase-4a-stability-checks-design.md`. Runtime Monitoring is 4b; timeline plot is 4c.

## Current State (anchors)

- `app/health/pipeline.py`: `HealthAnalysisPipeline.__init__(self, manager=None, calibration_profile=None)`; `analyze` does `features = self._prepare_features(window)` then `results = self.manager.run_checks(window, features)`. Imports `time`, `uuid`, `from typing import Any, Optional`.
- `app/health/checks/base.py`: `SignalHealthCheck` (class attrs `check_id`/`check_name`/`category`; abstractmethod `run(window, features) -> SignalCheckResult`).
- `app/health/config.py`: `REGISTRY` is a `list[CheckSpec]` of 11 entries (T001–T007 "time_domain", F001–F004 "frequency_domain"); `_profile_configs()` defines profiles; the `minimal` profile sets `categories={"time_domain": False, "frequency_domain": False}`.
- `tests/health/test_config.py`: `ALL_IDS` (11 ids), `assert len(REGISTRY) == 11`, dev/diagnostic `== 11`, `test_production_profile_excludes_electrical_hum` asserts `len == 10`, `test_mandatory_checks_run_even_when_categories_disabled` sets `categories={"time_domain": False, "frequency_domain": False}` and asserts `{"T001","T002","T004"}`.
- `app/health/defaults.py` + `tests/health/test_defaults.py` (11) are the LEGACY path — leave untouched.
- 102 tests pass.

## File Structure
- Modify: `app/health/pipeline.py` (history), `app/health/config.py` (register S001–S003 + minimal profile), `tests/health/test_config.py` (counts).
- Create: `app/health/checks/stability.py`, `tests/health/test_stability.py`.
- Modify test: `tests/health/test_pipeline.py` (history test).

---

## Task 1: Pipeline history

**Files:** Modify `app/health/pipeline.py`; Test `tests/health/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/health/test_pipeline.py`:

```python
from app.health.models import Measurement  # noqa: E402


def test_pipeline_history_accumulates_and_passes_prior_windows():
    seen = []

    class _HistorySpy(SignalHealthCheck):
        check_id = "SPY"
        check_name = "spy"

        def run(self, window, features):
            seen.append(len(features.get("history", [])))
            return SignalCheckResult(
                check_id="SPY", check_name="spy", measurements=[Measurement("v", 1.0)]
            )

    manager = SignalCheckManager()
    manager.register(_HistorySpy())
    pipeline = HealthAnalysisPipeline(manager=manager, history_length=3)
    for _ in range(5):
        pipeline.analyze(_window())
    # Each run sees prior windows only, bounded by history_length=3.
    assert seen == [0, 1, 2, 3, 3]
```

(`SignalHealthCheck`, `SignalCheckManager`, `HealthAnalysisPipeline`, `SignalCheckResult`, `_window` are already imported/defined at the top of `test_pipeline.py`.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_pipeline.py::test_pipeline_history_accumulates_and_passes_prior_windows -q`
Expected: FAIL — `HealthAnalysisPipeline()` got an unexpected keyword argument `history_length`.

- [ ] **Step 3: Add the history deque + injection in `pipeline.py`**

Add the import after `import uuid`:

```python
from collections import deque
```

Change `__init__`:

```python
    def __init__(self, manager: Optional[SignalCheckManager] = None, calibration_profile=None):
        self.manager = manager if manager is not None else SignalCheckManager()
        self.calibration_profile = calibration_profile
```

to:

```python
    def __init__(self, manager: Optional[SignalCheckManager] = None, calibration_profile=None, history_length: int = 20):
        self.manager = manager if manager is not None else SignalCheckManager()
        self.calibration_profile = calibration_profile
        self._history: deque = deque(maxlen=history_length)
```

Change this block in `analyze`:

```python
        # Stage 2 — Feature Preparation (Phase 0: no-op)
        features = self._prepare_features(window)
        # Stage 3 — Signal Health Checks
        results = self.manager.run_checks(window, features)
```

to:

```python
        # Stage 2 — Feature Preparation
        features = self._prepare_features(window)
        features["history"] = list(self._history)  # prior windows (oldest -> newest)
        # Stage 3 — Signal Health Checks
        results = self.manager.run_checks(window, features)
        # Record this window's measurements as the next entry of the stability history.
        self._history.append(
            {r.check_id: {m.name: m.value for m in r.measurements} for r in results}
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_pipeline.py -q`
Expected: all pass (existing + 1 new).

---

## Task 2: Stability checks (S001–S003)

**Files:** Create `app/health/checks/stability.py`; Test `tests/health/test_stability.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/health/test_stability.py`:

```python
import numpy as np

from app.health.checks.stability import (
    EnergyStabilityCheck,
    LongTermNoiseFloorCheck,
    SpectralStabilityCheck,
)
from app.health.models import AudioWindow, CheckCategory, CheckStatus


def _win():
    return AudioWindow(samples=np.zeros(100, dtype=np.float32), sample_rate=44100)


def _hist(check_id, measurement, values):
    return [{check_id: {measurement: v}} for v in values]


def _m(result, name):
    return next(x.value for x in result.measurements if x.name == name)


def test_energy_stability_passes_on_stable_series():
    r = EnergyStabilityCheck().run(_win(), {"history": _hist("T002", "rms", [0.2] * 8)})
    assert r.status is CheckStatus.PASS
    assert _m(r, "energy_cv") < 0.01
    assert r.category is CheckCategory.SUPPORTING


def test_energy_stability_warns_on_jitter():
    vals = [0.1, 0.4, 0.1, 0.5, 0.1, 0.6, 0.1, 0.5]
    r = EnergyStabilityCheck().run(_win(), {"history": _hist("T002", "rms", vals)})
    assert r.status is CheckStatus.WARNING


def test_stability_passes_with_insufficient_history():
    r = EnergyStabilityCheck().run(_win(), {"history": _hist("T002", "rms", [0.2, 0.2])})
    assert r.status is CheckStatus.PASS


def test_stability_passes_when_source_measurement_absent():
    hist = [{"F001": {"spectral_centroid": 1000.0}}] * 8  # no T002.rms
    r = EnergyStabilityCheck().run(_win(), {"history": hist})
    assert r.status is CheckStatus.PASS


def test_spectral_stability_warns_on_centroid_jitter():
    vals = [800, 2000, 800, 2200, 700, 2400, 750, 2100]
    r = SpectralStabilityCheck().run(
        _win(), {"history": _hist("F001", "spectral_centroid", vals)}
    )
    assert r.status is CheckStatus.WARNING


def test_noise_floor_warns_when_high():
    r = LongTermNoiseFloorCheck().run(_win(), {"history": _hist("T002", "rms", [0.1] * 8)})
    assert r.status is CheckStatus.WARNING
    assert _m(r, "noise_floor") > 0.05


def test_noise_floor_passes_when_low():
    r = LongTermNoiseFloorCheck().run(_win(), {"history": _hist("T002", "rms", [0.01] * 8)})
    assert r.status is CheckStatus.PASS
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_stability.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.health.checks.stability'`.

- [ ] **Step 3: Implement `app/health/checks/stability.py`**

```python
"""Stability Signal Health Checks (spec §4.10, S001–S003).

These read the pipeline's per-window measurement history from
``features["history"]`` — a list of ``{check_id: {measurement_name: value}}``
snapshots (oldest -> newest, prior windows only) — and flag intermittent or
gradual changes. All SUPPORTING and stateless; they PASS when there is too little
history or the source measurement is absent. Thresholds are provisional manual
defaults. Pure NumPy.
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


def _series(history, check_id: str, measurement: str) -> list[float]:
    """Pull a measurement's values across the history snapshots (skipping gaps)."""
    values = []
    for snap in history:
        v = snap.get(check_id, {}).get(measurement)
        if v is not None:
            values.append(float(v))
    return values


def _coefficient_of_variation(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    return float(arr.std() / mean) if mean > 0 else 0.0


class EnergyStabilityCheck(SignalHealthCheck):
    """S001 — flag unstable signal energy over recent windows (loose contacts, etc.)."""

    check_id = "S001"
    check_name = "Energy Stability"
    category = CheckCategory.SUPPORTING

    def __init__(self, min_samples: int = 5, max_variation: float = 0.5):
        self.min_samples = min_samples
        self.max_variation = max_variation

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        series = _series(features.get("history", []), "T002", "rms")
        if len(series) < self.min_samples:
            return self._result(0.0, CheckStatus.PASS, [])
        cv = _coefficient_of_variation(series)
        if cv > self.max_variation:
            return self._result(
                cv, CheckStatus.WARNING,
                [f"Energy variation {cv:.2f} over recent windows"],
            )
        return self._result(cv, CheckStatus.PASS, [])

    def _result(self, cv, status, diagnostics) -> SignalCheckResult:
        return SignalCheckResult(
            check_id=self.check_id, check_name=self.check_name, status=status,
            measurements=[Measurement("energy_cv", cv)], diagnostic_messages=diagnostics,
        )


class SpectralStabilityCheck(SignalHealthCheck):
    """S002 — flag drifting spectral centroid over recent windows (sensor ageing)."""

    check_id = "S002"
    check_name = "Spectral Stability"
    category = CheckCategory.SUPPORTING

    def __init__(self, min_samples: int = 5, max_variation: float = 0.3):
        self.min_samples = min_samples
        self.max_variation = max_variation

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        series = _series(features.get("history", []), "F001", "spectral_centroid")
        if len(series) < self.min_samples:
            return self._result(0.0, CheckStatus.PASS, [])
        cv = _coefficient_of_variation(series)
        if cv > self.max_variation:
            return self._result(
                cv, CheckStatus.WARNING,
                [f"Spectral centroid variation {cv:.2f} over recent windows"],
            )
        return self._result(cv, CheckStatus.PASS, [])

    def _result(self, cv, status, diagnostics) -> SignalCheckResult:
        return SignalCheckResult(
            check_id=self.check_id, check_name=self.check_name, status=status,
            measurements=[Measurement("centroid_cv", cv)], diagnostic_messages=diagnostics,
        )


class LongTermNoiseFloorCheck(SignalHealthCheck):
    """S003 — flag an elevated background noise floor (estimated from recent RMS)."""

    check_id = "S003"
    check_name = "Long-Term Noise Floor"
    category = CheckCategory.SUPPORTING

    def __init__(self, min_samples: int = 5, max_noise_floor: float = 0.05):
        self.min_samples = min_samples
        self.max_noise_floor = max_noise_floor

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        series = _series(features.get("history", []), "T002", "rms")
        if len(series) < self.min_samples:
            return self._result(0.0, CheckStatus.PASS, [])
        noise_floor = float(np.percentile(series, 25))
        if noise_floor > self.max_noise_floor:
            return self._result(
                noise_floor, CheckStatus.WARNING,
                [f"Background noise floor {noise_floor:.3f} elevated"],
            )
        return self._result(noise_floor, CheckStatus.PASS, [])

    def _result(self, noise_floor, status, diagnostics) -> SignalCheckResult:
        return SignalCheckResult(
            check_id=self.check_id, check_name=self.check_name, status=status,
            measurements=[Measurement("noise_floor", noise_floor)],
            diagnostic_messages=diagnostics,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_stability.py -q`
Expected: PASS (7 passed).

---

## Task 3: Register stability checks in config & profiles

**Files:** Modify `app/health/config.py`; Test `tests/health/test_config.py`

- [ ] **Step 1: Update the failing tests in `tests/health/test_config.py`**

(a) Change `ALL_IDS` to include the three new ids:

```python
ALL_IDS = {
    "T001", "T002", "T003", "T004", "T005", "T006", "T007",
    "F001", "F002", "F003", "F004",
    "S001", "S002", "S003",
}
```

(b) Change `assert len(REGISTRY) == 11` to `assert len(REGISTRY) == 14`.

(c) In `test_mandatory_checks_run_even_when_categories_disabled`, change the config to also disable stability:

```python
    cfg = HealthConfig(categories={"time_domain": False, "frequency_domain": False, "stability": False})
```

(d) Change the dev/diagnostic count test body to 14:

```python
def test_development_and_diagnostic_have_all_eleven():
    assert len(_profile_ids("development")) == 14
    assert len(_profile_ids("diagnostic")) == 14
```

(e) In `test_production_profile_excludes_electrical_hum`, change `assert len(ids) == 10` to `assert len(ids) == 13`.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_config.py -q`
Expected: FAIL (registry has 11 entries, dev profile has 11, etc.).

- [ ] **Step 3: Register the stability checks and fix the minimal profile in `app/health/config.py`**

Add the import after the `from app.health.checks.time_domain import (...)` block:

```python
from app.health.checks.stability import (
    EnergyStabilityCheck,
    LongTermNoiseFloorCheck,
    SpectralStabilityCheck,
)
```

Append three entries to `REGISTRY` (after the `CheckSpec("F004", ...)` line, before the closing `]`):

```python
    CheckSpec("S001", EnergyStabilityCheck, "stability"),
    CheckSpec("S002", SpectralStabilityCheck, "stability"),
    CheckSpec("S003", LongTermNoiseFloorCheck, "stability"),
```

In `_profile_configs()`, change the `minimal` profile so it also disables the stability category:

```python
        "minimal": HealthConfig(
            profile="minimal",
            categories={"time_domain": False, "frequency_domain": False, "stability": False},
        ),
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (~112 total: 102 prior + 1 pipeline + 7 stability + the test_config edits are modifications not additions). The legacy `tests/health/test_defaults.py` still asserts 11 (unchanged — `defaults.py` is not the live path).

- [ ] **Step 6: Headless end-to-end check**

```bash
.venv/bin/python -c "
import numpy as np
from app.health.config import pipeline_for_profile
from app.health.models import AudioWindow
SR, N = 44100, 110250
pipe = pipeline_for_profile('development')  # now 14 checks incl. S001-S003
ids = []
rng = np.random.default_rng(0)
for k in range(8):
    # alternate loud and quiet windows to make energy unstable
    amp = 0.4 if k % 2 == 0 else 0.02
    t = np.arange(N)/SR
    sig = (amp*np.sin(2*np.pi*1000*t) + 0.01*rng.standard_normal(N)).astype(np.float32)
    rep = pipe.analyze(AudioWindow(samples=sig, sample_rate=SR))
    ids = [r.check_id for r in rep.check_results]
print('checks per window:', len(ids), '| stability present:', {'S001','S002','S003'} <= set(ids))
s1 = next(r for r in rep.check_results if r.check_id=='S001')
print('S001 energy_cv:', s1.measurements[0].value, '| status:', s1.status.value)
"
```
Expected: `checks per window: 14 | stability present: True`, and after several unstable windows S001's `energy_cv` is sizeable (likely WARNING) — showing the cross-window history feeding the stability check.

---

## Phase 4a Done

The pipeline now carries a bounded measurement history and runs three stability checks (S001–S003) that detect intermittent/gradual change across windows; they appear in the panel and are characterized by calibration. Hand back to the owner for review, manual test, and commit. **Phase 4b** adds the `RuntimeMonitor` (fault persistence, recovery, events, smoothed state); **4c** the timeline plot.

---

## Self-Review

- **Spec coverage:** pipeline-maintained history injected via `features["history"]`, prior windows only, bounded by `history_length` (§2) — Task 1; S001 Energy Stability (energy_cv), S002 Spectral Stability (centroid_cv), S003 Long-Term Noise Floor (noise_floor), all SUPPORTING, PASS on insufficient history / missing source (§3) — Task 2; new `stability` registry category + profile counts (dev/diag 14, production 13, minimal 3) + minimal disables stability (§4) — Task 3. Headless tests with synthetic history (§6) — Task 2. Out-of-scope (4b/4c, defaults.py) respected.
- **Placeholder scan:** no TBD/TODO; full code each step; concrete commands and expected counts.
- **Type consistency:** `_series(history, check_id, measurement) -> list[float]`; checks read `features.get("history", [])`; snapshot shape `{check_id: {measurement: value}}` written by the pipeline (Task 1) and read by `_series` (Task 2); `CheckSpec(check_id, factory, category)` rows added with category `"stability"` (Task 3); `minimal` categories now include `"stability": False` so `is_active` excludes them. `EnergyStabilityCheck`/`SpectralStabilityCheck`/`LongTermNoiseFloorCheck` names match between `stability.py`, the `config.py` import/REGISTRY, and the tests.
