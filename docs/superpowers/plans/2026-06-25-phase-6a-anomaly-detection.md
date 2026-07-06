# Phase 6a — Anomaly Detection & Confidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute a holistic anomaly distance of each window from the calibration profile (RMS z-distance) and turn it into the report's confidence, filling pipeline Stage 5.

**Architecture:** A pure `app/health/anomaly.py` (`AnomalyResult`, `detect_anomaly`) scores a window's measurements against the profile's per-measurement `mean`/`std`. The pipeline's Stage 5 calls it when a profile is loaded; `fusion.decide` gains an optional `anomaly_result` that sets `confidence` (state untouched — calibration owns state). Duck-typed; NumPy-only.

**Tech Stack:** Python 3.13, NumPy, stdlib (dataclasses); pytest. `app/health/` stays NumPy/stdlib-only.

> **Commit policy:** Owner commits manually. **Do NOT run git commit/add/push.** End at "tests pass."

> **Reference spec:** `docs/superpowers/specs/2026-06-25-phase-6-anomaly-detection-design.md`. 6b (surface confidence/anomaly in the UI) is the next plan.

> **Decision (§2):** diagonal RMS z-distance (profiles store only per-measurement stats); anomaly feeds CONFIDENCE, never `final_state`.

## Current State (anchors)

- `app/health/calibration.py`: `CalibrationProfile.statistics: dict[str, dict[str, MeasurementStats]]` (keyed `check_id` → `measurement_name`); `MeasurementStats` has `.mean`, `.std` (floats).
- `app/health/models.py`: `SignalCheckResult` has `.check_id`, `.measurements` (each `Measurement` has `.name`, `.value`), `.executed`, `.status`, `.category`, `.diagnostic_messages`; `CheckStatus`, `CheckCategory`, `HealthState`.
- `app/health/fusion.py`: `def decide(results, calibration_evaluation=None) -> tuple[HealthState, float, str]:` ends with a calibration fold then `return state, confidence, summary`.
- `app/health/pipeline.py`: `analyze` has `anomaly_result = self._detect_anomalies(features, results)` and later `final_state, confidence, summary = decide(results, calibration_evaluation)`; `_detect_anomalies(self, features, results)` is a stub returning `None`. The pipeline holds `self.calibration_profile`.
- 128 tests pass.

## File Structure
- Create: `app/health/anomaly.py`, `tests/health/test_anomaly.py`.
- Modify: `app/health/fusion.py` (+ `tests/health/test_fusion.py`), `app/health/pipeline.py` (+ `tests/health/test_pipeline.py`).

---

## Task 1: `detect_anomaly` (RMS z-distance)

**Files:** Create `app/health/anomaly.py`; Test `tests/health/test_anomaly.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/health/test_anomaly.py`:

```python
from app.health.anomaly import AnomalyResult, detect_anomaly


class _Stat:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


class _Profile:
    def __init__(self, statistics):
        self.statistics = statistics


class _M:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class _R:
    def __init__(self, check_id, measurements):
        self.check_id = check_id
        self.measurements = measurements


def _profile():
    return _Profile({"T002": {"rms": _Stat(mean=0.2, std=0.05)},
                     "F001": {"spectral_centroid": _Stat(mean=1000.0, std=100.0)}})


def test_at_mean_is_not_anomalous():
    results = [_R("T002", [_M("rms", 0.2)]), _R("F001", [_M("spectral_centroid", 1000.0)])]
    r = detect_anomaly(results, _profile())
    assert isinstance(r, AnomalyResult)
    assert r.distance == 0.0
    assert r.is_anomalous is False
    assert r.confidence == 1.0


def test_far_out_is_anomalous_with_low_confidence():
    # rms 8 sigma out, centroid 1 sigma out -> RMS z = sqrt((64+1)/2) ~ 5.7 > 3
    results = [_R("T002", [_M("rms", 0.6)]), _R("F001", [_M("spectral_centroid", 1100.0)])]
    r = detect_anomaly(results, _profile(), threshold=3.0)
    assert r.is_anomalous is True
    assert r.distance > 3.0
    assert r.confidence == 0.0
    assert r.contributors[0][0] == "T002.rms"  # dominant deviation


def test_returns_none_when_no_measurements_match_profile():
    results = [_R("Z999", [_M("foo", 1.0)])]
    assert detect_anomaly(results, _profile()) is None


def test_skips_zero_std_measurements():
    profile = _Profile({"T002": {"rms": _Stat(mean=0.2, std=0.0)}})
    results = [_R("T002", [_M("rms", 5.0)])]
    assert detect_anomaly(results, profile) is None  # only stat skipped -> no usable z


def test_confidence_is_half_at_threshold():
    # one measurement exactly `threshold` sigma out -> distance == threshold
    profile = _Profile({"T002": {"rms": _Stat(mean=0.0, std=1.0)}})
    results = [_R("T002", [_M("rms", 3.0)])]
    r = detect_anomaly(results, profile, threshold=3.0)
    assert r.distance == 3.0
    assert r.confidence == 0.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_anomaly.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.health.anomaly'`.

- [ ] **Step 3: Implement `app/health/anomaly.py`**

```python
"""Anomaly detection (spec Ch. 6 / Phase 6): a holistic distance of a window's
measurements from the calibration profile, turned into a confidence.

Diagonal Mahalanobis (RMS z-distance) — calibration profiles store only
per-measurement stats, so this treats measurements as independent. Pure NumPy.
The profile is duck-typed: ``profile.statistics[check_id][name]`` with ``.mean`` /
``.std``; results are duck-typed (``check_id`` + ``measurements`` of ``name``/``value``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AnomalyResult:
    distance: float
    threshold: float
    is_anomalous: bool
    contributors: list  # list[tuple[str, float]] — top (label, z) by |z|
    confidence: float


def detect_anomaly(results, profile, *, threshold: float = 3.0):
    statistics = profile.statistics
    deviations = []  # (label, z)
    for result in results:
        check_stats = statistics.get(result.check_id)
        if not check_stats:
            continue
        for measurement in result.measurements:
            stat = check_stats.get(measurement.name)
            if stat is None or stat.std <= 0:
                continue
            z = (float(measurement.value) - stat.mean) / stat.std
            deviations.append((f"{result.check_id}.{measurement.name}", z))

    if not deviations:
        return None

    z_values = np.array([z for _, z in deviations], dtype=np.float64)
    distance = float(np.sqrt(np.mean(z_values ** 2)))
    is_anomalous = distance > threshold
    confidence = max(0.0, 1.0 - distance / (2 * threshold))
    contributors = sorted(deviations, key=lambda item: -abs(item[1]))[:3]
    return AnomalyResult(
        distance=distance,
        threshold=threshold,
        is_anomalous=is_anomalous,
        contributors=contributors,
        confidence=confidence,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_anomaly.py -q`
Expected: PASS (5 passed).

---

## Task 2: Wire anomaly into fusion + pipeline

**Files:** Modify `app/health/fusion.py`, `app/health/pipeline.py`; Test `tests/health/test_fusion.py`, `tests/health/test_pipeline.py`

- [ ] **Step 1: Write the failing fusion test**

Append to `tests/health/test_fusion.py` (it already imports `decide`; add what it lacks):

```python
from app.health.models import (  # noqa: E402,F811
    CheckCategory,
    CheckStatus,
    HealthState,
    SignalCheckResult,
)


class _Anom:
    def __init__(self, confidence, is_anomalous=True, distance=4.0):
        self.confidence = confidence
        self.is_anomalous = is_anomalous
        self.distance = distance
        self.contributors = [("F001.spectral_centroid", 4.8)]


def _passing_results():
    return [
        SignalCheckResult(
            check_id="T002", check_name="Signal Energy",
            status=CheckStatus.PASS, category=CheckCategory.CRITICAL, executed=True,
        )
    ]


def test_decide_without_anomaly_keeps_check_confidence():
    state, confidence, _ = decide(_passing_results())
    assert state is HealthState.OK
    assert confidence == 1.0


def test_decide_with_anomaly_sets_confidence_and_keeps_state():
    state, confidence, summary = decide(_passing_results(), None, _Anom(confidence=0.25))
    assert state is HealthState.OK  # anomaly never changes state
    assert confidence == 0.25
    assert "anomaly distance" in summary
    assert "F001.spectral_centroid" in summary
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_fusion.py::test_decide_with_anomaly_sets_confidence_and_keeps_state -q`
Expected: FAIL — `decide()` takes no third positional argument.

- [ ] **Step 3: Add the `anomaly_result` parameter to `decide`**

In `app/health/fusion.py`, change the signature:

```python
def decide(results, calibration_evaluation=None) -> tuple[HealthState, float, str]:
```

to:

```python
def decide(results, calibration_evaluation=None, anomaly_result=None) -> tuple[HealthState, float, str]:
```

Then find the end of the function:

```python
    return state, confidence, summary
```

and replace it with:

```python
    # Fold in the anomaly result: it sets confidence (calibration owns state).
    if anomaly_result is not None:
        confidence = anomaly_result.confidence
        label = "ANOMALOUS" if anomaly_result.is_anomalous else "normal"
        note = f"anomaly distance {anomaly_result.distance:.1f} ({label})"
        if anomaly_result.contributors:
            top_label, top_z = anomaly_result.contributors[0]
            note += f", {top_label} z={top_z:.1f}"
        summary = f"{summary} | {note}" if summary else note

    return state, confidence, summary
```

- [ ] **Step 4: Run to verify the fusion test passes**

Run: `.venv/bin/python -m pytest tests/health/test_fusion.py -q`
Expected: all pass.

- [ ] **Step 5: Write the failing pipeline test**

Append to `tests/health/test_pipeline.py`:

```python
def test_pipeline_populates_anomaly_and_confidence_with_profile():
    import numpy as np
    from app.health.calibration import generate_profile
    from app.health.config import pipeline_for_profile

    sr, n = 44100, int(44100 * 2.5)
    rng = np.random.default_rng(0)
    healthy = [(0.2 * np.sin(2 * np.pi * 1000 * np.arange(n) / sr)
                + 0.01 * rng.standard_normal(n)).astype(np.float32) for _ in range(3)]
    profile = generate_profile(healthy, sr, profile_id="t")

    pipe = pipeline_for_profile("development", calibration_profile=profile)
    report = pipe.analyze(AudioWindow(samples=healthy[0], sample_rate=sr))
    assert report.anomaly_result is not None
    assert report.confidence == report.anomaly_result.confidence


def test_pipeline_without_profile_has_no_anomaly():
    from app.health.config import pipeline_for_profile

    pipe = pipeline_for_profile("development")  # no calibration profile
    report = pipe.analyze(_window())
    assert report.anomaly_result is None
```

(`AudioWindow`, `_window` are already imported/defined at the top of `test_pipeline.py`.)

- [ ] **Step 6: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_pipeline.py::test_pipeline_populates_anomaly_and_confidence_with_profile -q`
Expected: FAIL — `report.anomaly_result is None` (Stage 5 is still a stub).

- [ ] **Step 7: Implement Stage 5 + pass the anomaly into `decide` in `pipeline.py`**

In `app/health/pipeline.py`, replace the stub:

```python
    def _detect_anomalies(self, features: dict, results: list) -> Optional[Any]:
        # Phase 0 no-op; Phase 6 scores the feature vector against healthy behaviour.
        return None
```

with:

```python
    def _detect_anomalies(self, features: dict, results: list) -> Optional[Any]:
        # Phase 6: holistic RMS z-distance of this window's measurements from the
        # calibration profile (only when a profile is loaded).
        if self.calibration_profile is None:
            return None
        from app.health.anomaly import detect_anomaly

        return detect_anomaly(results, self.calibration_profile)
```

Then find:

```python
        final_state, confidence, summary = decide(results, calibration_evaluation)
```

and replace with:

```python
        final_state, confidence, summary = decide(results, calibration_evaluation, anomaly_result)
```

- [ ] **Step 8: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_pipeline.py -q`
Expected: all pass.

- [ ] **Step 9: Full suite + import purity**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (≈138: 128 + 5 anomaly + 2 fusion + 2 pipeline, plus the existing ones).
Run: `grep -nE "^import |^from " app/health/anomaly.py`
Expected: only `dataclasses` + `numpy` (no UI; profile/results are duck-typed).

- [ ] **Step 10: Headless demo**

```bash
.venv/bin/python -c "
import numpy as np
from app.health.calibration import generate_profile
from app.health.config import pipeline_for_profile
from app.health.models import AudioWindow
sr, n = 44100, int(44100*2.5)
rng = np.random.default_rng(0)
healthy = [(0.2*np.sin(2*np.pi*1000*np.arange(n)/sr)+0.01*rng.standard_normal(n)).astype(np.float32) for _ in range(4)]
profile = generate_profile(healthy, sr, profile_id='t')
pipe = pipeline_for_profile('development', calibration_profile=profile)
healthy_rep = pipe.analyze(AudioWindow(samples=healthy[0], sample_rate=sr))
off = (2.0*np.sin(2*np.pi*5000*np.arange(n)/sr)).astype(np.float32)  # loud, off-band
off_rep = pipe.analyze(AudioWindow(samples=off, sample_rate=sr))
print('healthy: dist=%.2f conf=%.2f anomalous=%s' % (healthy_rep.anomaly_result.distance, healthy_rep.confidence, healthy_rep.anomaly_result.is_anomalous))
print('off:     dist=%.2f conf=%.2f anomalous=%s' % (off_rep.anomaly_result.distance, off_rep.confidence, off_rep.anomaly_result.is_anomalous))
"
```
Expected: the healthy window has a small distance / high confidence / `anomalous=False`; the off-profile window has a larger distance, lower confidence, and `anomalous=True`.

---

## Phase 6a Done

Stage 5 now scores each window against the calibration profile and the report carries an `anomaly_result` + a profile-driven `confidence`; with no profile, behavior is unchanged. Hand back to the owner for review, manual test, and commit. **Phase 6b** surfaces the confidence and anomaly in the indicator/log.

---

## Self-Review

- **Spec coverage:** RMS z-distance over per-measurement `mean`/`std`, skip `std≤0`, `None` when nothing matches (§2/§3) — Task 1; `confidence = max(0, 1 − distance/(2·threshold))`, contributors top-3 by |z| (§3) — Task 1; Stage 5 calls `detect_anomaly` only with a profile (§3) — Task 2 Step 7; `decide(..., anomaly_result=None)` sets confidence + summary note, never state (§2/§3) — Task 2 Steps 3/7; report carries `anomaly_result` + matching `confidence` — Task 2 tests. UI deferred to 6b (§4).
- **Placeholder scan:** no TBD/TODO; full code each step; concrete commands and expected outputs.
- **Type consistency:** `detect_anomaly(results, profile, *, threshold=3.0) -> AnomalyResult | None`; `AnomalyResult(distance, threshold, is_anomalous, contributors, confidence)`; `decide(results, calibration_evaluation=None, anomaly_result=None)` reads `anomaly_result.confidence`/`.is_anomalous`/`.distance`/`.contributors`; pipeline passes `anomaly_result` (already computed at Stage 5) into `decide`. Profile (`statistics[check_id][name].mean/.std`) and results (`check_id`/`measurements[].name/.value`) are duck-typed — anomaly.py imports only `dataclasses` + `numpy`; the `detect_anomaly` import inside the pipeline is function-local to keep import order clean.
