# Phase 2a — Feature Preparation & Frequency-Domain Checks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared Feature Preparation stage (one rFFT per window) and the four frequency-domain Signal Health Checks (F001–F004), wired into the default pipeline so the tester's existing health indicator/log also reflects spectral problems (abnormal spectral shape, broadband noise, electrical hum).

**Architecture:** A pure-NumPy `feature_prep.prepare_features(window)` computes a DC-removed, Hann-windowed power spectrum + frequency bins once; the pipeline's Stage 2 calls it and passes the result to every check. Four new checks in `checks/frequency_domain.py` consume that shared spectrum (they never recompute an FFT). Each uses provisional manual thresholds (Phase 3 calibration replaces them) and is WARNING-only or measurement-only, since manual spectral thresholds are unreliable pre-calibration. Windows with no spectral energy (silence/non-finite) pass the frequency checks — the time-domain critical checks own that failure.

**Tech Stack:** Python 3.13, NumPy 2.x, pytest. Run via `.venv/bin/python -m pytest`.

> **Commit policy:** Owner commits/pushes **manually** after review. **Do NOT run `git commit`/`git add`/`git push`.** Each task ends at "tests pass."

> **References:** spec `arch_update.md` §3.6 (Feature Preparation), §4.9 (F001–F004), §10.3 (categories). Design spec §5 Phase 2. Phase 1 plan: `docs/superpowers/plans/2026-06-23-phase-1-time-domain-checks-and-fusion.md`.

> **Thresholds:** all values below are **provisional manual defaults** for float32 audio in [-1,1] at 44.1 kHz; lenient by design so a healthy signal reads OK. Phase 3 replaces them with calibration-derived values.

---

## Current State (end of Phase 1)

- `app/health/models.py`: `HealthState`, `CheckStatus`, `CheckCategory`(CRITICAL/PRIMARY/SUPPORTING), `Measurement(name,value,unit="")`, immutable `AudioWindow` (`.samples` read-only float32 1-D, `.sample_rate`), `SignalCheckResult`(status, category, executed, execution_time, measurements, diagnostic_messages), `HealthReport`.
- `app/health/checks/base.py`: `SignalHealthCheck` (class attrs `check_id`/`check_name`/`category`; abstractmethod `run(window, features) -> SignalCheckResult`).
- `app/health/checks/time_domain.py`: T001–T007.
- `app/health/manager.py`: `SignalCheckManager` (stamps `category`, isolates failures, times each check).
- `app/health/pipeline.py`: `HealthAnalysisPipeline.analyze` Stage 2 calls `self._prepare_features(window)` which currently `return {}`.
- `app/health/fusion.py`: rule-based `decide(results)`.
- `app/health/defaults.py`: `default_time_domain_checks()`, `default_manager()`, `default_pipeline()`.
- 52 tests pass (`tests/health/` + `tests/test_scaler.py`).

## File Structure

**Create:**
- `app/health/feature_prep.py` — `prepare_features(window) -> dict`.
- `app/health/checks/frequency_domain.py` — `SpectralShapeCheck` (F001), `SpectralFlatnessCheck` (F002), `BandEnergyDistributionCheck` (F003), `ElectricalHumCheck` (F004).
- `tests/health/test_feature_prep.py`, `tests/health/test_frequency_domain.py`.

**Modify:**
- `app/health/pipeline.py` — Stage 2 calls `prepare_features`.
- `app/health/defaults.py` — register the four frequency checks too.
- `tests/health/test_defaults.py` — expect 11 checks.
- `main.py` — introduce a `SAMPLE_RATE` constant (the literal is now load-bearing for frequency bins).

---

## Task 1: Feature Preparation stage

**Files:**
- Create: `app/health/feature_prep.py`
- Modify: `app/health/pipeline.py`
- Test: `tests/health/test_feature_prep.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/health/test_feature_prep.py`:

```python
import numpy as np

from app.health.feature_prep import prepare_features
from app.health.models import AudioWindow

SR = 44100
N = 110250


def _win(x):
    return AudioWindow(samples=np.asarray(x, dtype=np.float32), sample_rate=SR)


def _sine(freq, n=N, amp=0.3):
    t = np.arange(n) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def test_prepare_features_keys_and_shapes():
    feats = prepare_features(_win(_sine(1000.0)))
    assert set(feats) >= {"freqs", "power_spectrum", "sample_rate", "n"}
    assert feats["freqs"].shape == feats["power_spectrum"].shape
    assert feats["freqs"].shape[0] == N // 2 + 1  # rfft length
    assert feats["sample_rate"] == SR


def test_power_spectrum_peaks_at_tone_frequency():
    feats = prepare_features(_win(_sine(1000.0)))
    peak_freq = feats["freqs"][int(np.argmax(feats["power_spectrum"]))]
    assert abs(peak_freq - 1000.0) < 25.0  # within one rFFT bin-ish


def test_dc_is_removed():
    # A signal with a large DC offset must not put all energy in the 0 Hz bin.
    feats = prepare_features(_win(_sine(1000.0) + 0.5))
    assert feats["power_spectrum"][0] < feats["power_spectrum"].max()


def test_silence_has_zero_power():
    feats = prepare_features(_win(np.zeros(N)))
    assert float(feats["power_spectrum"].sum()) == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_feature_prep.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.health.feature_prep'`.

- [ ] **Step 3: Implement `feature_prep.py`**

Create `app/health/feature_prep.py`:

```python
"""Shared spectral representation computed once per analysis window (spec §3.6).

Frequency-domain checks consume this instead of each recomputing an FFT. Pure
NumPy. DC is removed and a Hann window applied before the rFFT to reduce spectral
leakage and stop the 0 Hz bin from dominating spectral statistics (DC is checked
separately by the time-domain DC Offset check).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.health.models import AudioWindow


def prepare_features(window: AudioWindow) -> dict[str, Any]:
    x = window.samples
    n = int(x.size)
    if n == 0:
        return {
            "sample_rate": int(window.sample_rate),
            "n": 0,
            "freqs": np.zeros(0, dtype=np.float64),
            "power_spectrum": np.zeros(0, dtype=np.float64),
        }
    xd = x.astype(np.float64)
    xd = xd - xd.mean()  # remove DC
    spectrum = np.fft.rfft(xd * np.hanning(n))
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / window.sample_rate)
    return {
        "sample_rate": int(window.sample_rate),
        "n": n,
        "freqs": freqs,
        "power_spectrum": power,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_feature_prep.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Wire it into the pipeline**

In `app/health/pipeline.py`, add the import after the existing `from app.health.fusion import decide` line:

```python
from app.health.feature_prep import prepare_features
```

Then change the `_prepare_features` method body from:

```python
    def _prepare_features(self, window: AudioWindow) -> dict[str, Any]:
        return {}
```

to:

```python
    def _prepare_features(self, window: AudioWindow) -> dict[str, Any]:
        return prepare_features(window)
```

- [ ] **Step 6: Confirm the full suite still passes**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (56 total: 52 prior + 4 new). Time-domain checks ignore `features`, so no behavior change.

---

## Task 2: Spectral Shape (F001) & Spectral Flatness (F002)

**Files:**
- Create: `app/health/checks/frequency_domain.py`
- Test: `tests/health/test_frequency_domain.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/health/test_frequency_domain.py`:

```python
import numpy as np

from app.health.checks.frequency_domain import (
    SpectralFlatnessCheck,
    SpectralShapeCheck,
)
from app.health.feature_prep import prepare_features
from app.health.models import AudioWindow, CheckCategory, CheckStatus

SR = 44100
N = 110250
RNG = np.random.default_rng(0)


def _feats(x):
    return prepare_features(
        AudioWindow(samples=np.asarray(x, dtype=np.float32), sample_rate=SR)
    )


def _win(x):
    return AudioWindow(samples=np.asarray(x, dtype=np.float32), sample_rate=SR)


def _sine(freq, n=N, amp=0.3):
    t = np.arange(n) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def _measure(result, name):
    return next(m.value for m in result.measurements if m.name == name)


def test_spectral_shape_passes_on_sine_and_reports_centroid():
    check = SpectralShapeCheck()
    assert check.category is CheckCategory.PRIMARY
    x = _sine(1000.0)
    result = check.run(_win(x), _feats(x))
    assert result.status is CheckStatus.PASS
    assert abs(_measure(result, "spectral_centroid") - 1000.0) < 50.0


def test_spectral_shape_passes_on_silence():
    # No spectral energy -> the frequency check defers to the time-domain checks.
    result = SpectralShapeCheck().run(_win(np.zeros(N)), _feats(np.zeros(N)))
    assert result.status is CheckStatus.PASS


def test_spectral_flatness_low_for_tone_high_for_noise():
    tone = _sine(1000.0)
    noise = RNG.uniform(-0.3, 0.3, N)
    flat_tone = SpectralFlatnessCheck().run(_win(tone), _feats(tone))
    flat_noise = SpectralFlatnessCheck().run(_win(noise), _feats(noise))
    assert flat_tone.status is CheckStatus.PASS
    assert _measure(flat_tone, "spectral_flatness") < _measure(
        flat_noise, "spectral_flatness"
    )


def test_spectral_flatness_warns_when_above_threshold():
    noise = RNG.uniform(-0.3, 0.3, N)
    # Force the boundary deterministically with an explicit low threshold.
    result = SpectralFlatnessCheck(maximum_flatness=0.05).run(_win(noise), _feats(noise))
    assert result.status is CheckStatus.WARNING
    assert result.category is CheckCategory.SUPPORTING
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_frequency_domain.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.health.checks.frequency_domain'`.

- [ ] **Step 3: Implement F001 and F002**

Create `app/health/checks/frequency_domain.py`:

```python
"""Frequency-domain Signal Health Checks (spec §4.9, F001–F004).

Each check consumes the shared spectrum from feature preparation
(`features["power_spectrum"]`, `features["freqs"]`) and never recomputes an FFT.
Pure NumPy. Thresholds are provisional manual defaults; Phase 3 replaces them with
calibration-derived values. A window with no spectral energy (silence/non-finite)
PASSES here — the time-domain critical checks (Flatline, Signal Energy) own that
failure, so the frequency layer must not double-report it.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from app.health.checks.base import SignalHealthCheck
from app.health.models import (
    AudioWindow,
    CheckCategory,
    CheckStatus,
    Measurement,
    SignalCheckResult,
)


def _usable_spectrum(features: dict[str, Any]) -> Optional[tuple]:
    """Return (freqs, power, total_power) or None when there is no usable energy."""
    freqs = features.get("freqs")
    power = features.get("power_spectrum")
    if freqs is None or power is None or power.size == 0:
        return None
    total = float(power.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    return freqs, power, total


class SpectralShapeCheck(SignalHealthCheck):
    """F001 — verify overall frequency response (centroid, bandwidth, roll-off)."""

    check_id = "F001"
    check_name = "Spectral Shape"
    category = CheckCategory.PRIMARY

    def __init__(
        self,
        centroid_range: tuple = (50.0, 12000.0),
        bandwidth_range: tuple = (0.0, 12000.0),
        rolloff_fraction: float = 0.85,
        rolloff_range: tuple = (0.0, 20000.0),
    ):
        self.centroid_range = centroid_range
        self.bandwidth_range = bandwidth_range
        self.rolloff_fraction = rolloff_fraction
        self.rolloff_range = rolloff_range

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        spec = _usable_spectrum(features)
        if spec is None:
            return SignalCheckResult(
                check_id=self.check_id,
                check_name=self.check_name,
                status=CheckStatus.PASS,
                measurements=[
                    Measurement("spectral_centroid", 0.0, "Hz"),
                    Measurement("spectral_bandwidth", 0.0, "Hz"),
                    Measurement("spectral_rolloff", 0.0, "Hz"),
                ],
            )
        freqs, power, total = spec
        centroid = float((freqs * power).sum() / total)
        bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * power).sum() / total))
        cumulative = np.cumsum(power)
        idx = int(np.searchsorted(cumulative, self.rolloff_fraction * total))
        rolloff = float(freqs[min(idx, freqs.size - 1)])

        diagnostics = []
        if not (self.centroid_range[0] <= centroid <= self.centroid_range[1]):
            diagnostics.append(f"Spectral centroid {centroid:.0f} Hz outside expected range")
        if not (self.bandwidth_range[0] <= bandwidth <= self.bandwidth_range[1]):
            diagnostics.append(f"Spectral bandwidth {bandwidth:.0f} Hz outside expected range")
        if not (self.rolloff_range[0] <= rolloff <= self.rolloff_range[1]):
            diagnostics.append(f"Spectral roll-off {rolloff:.0f} Hz outside expected range")
        status = CheckStatus.WARNING if diagnostics else CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[
                Measurement("spectral_centroid", centroid, "Hz"),
                Measurement("spectral_bandwidth", bandwidth, "Hz"),
                Measurement("spectral_rolloff", rolloff, "Hz"),
            ],
            diagnostic_messages=diagnostics,
        )


class SpectralFlatnessCheck(SignalHealthCheck):
    """F002 — measure how noise-like the spectrum is (geometric/arithmetic mean)."""

    check_id = "F002"
    check_name = "Spectral Flatness"
    category = CheckCategory.SUPPORTING

    def __init__(self, maximum_flatness: float = 0.6):
        self.maximum_flatness = maximum_flatness

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        spec = _usable_spectrum(features)
        if spec is None:
            return SignalCheckResult(
                check_id=self.check_id,
                check_name=self.check_name,
                status=CheckStatus.PASS,
                measurements=[Measurement("spectral_flatness", 0.0)],
            )
        _, power, _ = spec
        p = power + 1e-20
        geometric = float(np.exp(np.mean(np.log(p))))
        arithmetic = float(np.mean(p))
        flatness = geometric / arithmetic if arithmetic > 0 else 0.0
        diagnostics = []
        if flatness > self.maximum_flatness:
            status = CheckStatus.WARNING
            diagnostics.append(f"Spectral flatness {flatness:.3f} indicates broadband noise")
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[Measurement("spectral_flatness", flatness)],
            diagnostic_messages=diagnostics,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_frequency_domain.py -q`
Expected: PASS (4 passed).

---

## Task 3: Band Energy Distribution (F003) & Electrical Hum (F004)

**Files:**
- Modify: `app/health/checks/frequency_domain.py` (append two classes)
- Test: `tests/health/test_frequency_domain.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/health/test_frequency_domain.py`:

```python
from app.health.checks.frequency_domain import (  # noqa: E402
    BandEnergyDistributionCheck,
    ElectricalHumCheck,
)


def test_band_energy_reports_ratios_and_passes():
    x = _sine(1000.0)
    result = BandEnergyDistributionCheck().run(_win(x), _feats(x))
    assert result.status is CheckStatus.PASS
    assert result.category is CheckCategory.PRIMARY
    # A 1 kHz tone puts most energy in the 500–2000 Hz band.
    assert _measure(result, "band_500_2000_ratio") > 0.5


def test_band_energy_passes_on_silence():
    result = BandEnergyDistributionCheck().run(_win(np.zeros(N)), _feats(np.zeros(N)))
    assert result.status is CheckStatus.PASS


def test_electrical_hum_warns_on_mains_tone():
    hum = _sine(60.0, amp=0.3)
    result = ElectricalHumCheck(fundamental_frequency=60.0).run(_win(hum), _feats(hum))
    assert result.status is CheckStatus.WARNING
    assert result.category is CheckCategory.SUPPORTING
    assert _measure(result, "hum_ratio") > 0.5


def test_electrical_hum_passes_on_clean_tone():
    x = _sine(1000.0)
    result = ElectricalHumCheck(fundamental_frequency=60.0).run(_win(x), _feats(x))
    assert result.status is CheckStatus.PASS
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_frequency_domain.py -q`
Expected: FAIL with `ImportError: cannot import name 'BandEnergyDistributionCheck'`.

- [ ] **Step 3: Append F003 and F004**

Append to `app/health/checks/frequency_domain.py`:

```python


class BandEnergyDistributionCheck(SignalHealthCheck):
    """F003 — report the energy distribution across frequency bands.

    Phase 2a is measurement-only (always PASS): it characterises the band
    distribution for the panel and for calibration. Phase 3 adds calibrated
    thresholds on the expected distribution.
    """

    check_id = "F003"
    check_name = "Band Energy Distribution"
    category = CheckCategory.PRIMARY

    def __init__(self, bands: tuple = (
        (0.0, 500.0),
        (500.0, 2000.0),
        (2000.0, 8000.0),
        (8000.0, 22050.0),
    )):
        self.bands = bands

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        spec = _usable_spectrum(features)
        measurements = []
        if spec is None:
            for lo, hi in self.bands:
                measurements.append(
                    Measurement(f"band_{int(lo)}_{int(hi)}_ratio", 0.0)
                )
            return SignalCheckResult(
                check_id=self.check_id,
                check_name=self.check_name,
                status=CheckStatus.PASS,
                measurements=measurements,
            )
        freqs, power, total = spec
        for lo, hi in self.bands:
            mask = (freqs >= lo) & (freqs < hi)
            ratio = float(power[mask].sum()) / total
            measurements.append(Measurement(f"band_{int(lo)}_{int(hi)}_ratio", ratio))
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=CheckStatus.PASS,
            measurements=measurements,
        )


class ElectricalHumCheck(SignalHealthCheck):
    """F004 — detect mains-frequency interference (fundamental + harmonics)."""

    check_id = "F004"
    check_name = "Electrical Hum Detection"
    category = CheckCategory.SUPPORTING

    def __init__(
        self,
        fundamental_frequency: float = 60.0,
        harmonic_count: int = 3,
        analysis_bandwidth: float = 2.0,
        max_hum_ratio: float = 0.1,
    ):
        self.fundamental_frequency = fundamental_frequency
        self.harmonic_count = harmonic_count
        self.analysis_bandwidth = analysis_bandwidth
        self.max_hum_ratio = max_hum_ratio

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        spec = _usable_spectrum(features)
        if spec is None:
            return SignalCheckResult(
                check_id=self.check_id,
                check_name=self.check_name,
                status=CheckStatus.PASS,
                measurements=[
                    Measurement("hum_ratio", 0.0),
                    Measurement("hum_energy", 0.0),
                ],
            )
        freqs, power, total = spec
        hum_energy = 0.0
        for k in range(1, self.harmonic_count + 1):
            target = self.fundamental_frequency * k
            mask = np.abs(freqs - target) <= self.analysis_bandwidth
            hum_energy += float(power[mask].sum())
        hum_ratio = hum_energy / total
        diagnostics = []
        if hum_ratio > self.max_hum_ratio:
            status = CheckStatus.WARNING
            diagnostics.append(
                f"Mains hum ratio {hum_ratio:.3f} around {self.fundamental_frequency:.0f} Hz"
            )
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[
                Measurement("hum_ratio", hum_ratio),
                Measurement("hum_energy", hum_energy),
            ],
            diagnostic_messages=diagnostics,
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_frequency_domain.py -q`
Expected: PASS (8 passed).

---

## Task 4: Register frequency checks in the default pipeline

**Files:**
- Modify: `app/health/defaults.py`
- Test: `tests/health/test_defaults.py`

- [ ] **Step 1: Update the failing tests**

In `tests/health/test_defaults.py`, change `test_default_manager_registers_seven_checks` to expect 11 and update the sine test's count. Replace:

```python
def test_default_manager_registers_seven_checks():
    assert len(default_manager().checks) == 7
```

with:

```python
def test_default_manager_registers_all_checks():
    assert len(default_manager().checks) == 11  # 7 time-domain + 4 frequency-domain
```

And in `test_default_pipeline_ok_on_clean_sine`, change `assert len(report.check_results) == 7` to `assert len(report.check_results) == 11`.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_defaults.py -q`
Expected: FAIL (`assert 7 == 11`, and the import of `test_default_manager_registers_seven_checks` no longer exists — only the count asserts fail).

- [ ] **Step 3: Register the frequency checks**

In `app/health/defaults.py`, add the import after the time-domain import block:

```python
from app.health.checks.frequency_domain import (
    BandEnergyDistributionCheck,
    ElectricalHumCheck,
    SpectralFlatnessCheck,
    SpectralShapeCheck,
)
```

Add a frequency-checks factory after `default_time_domain_checks`:

```python
def default_frequency_domain_checks() -> list[SignalHealthCheck]:
    """The four frequency-domain checks (F001–F004) with default thresholds."""
    return [
        SpectralShapeCheck(),
        SpectralFlatnessCheck(),
        BandEnergyDistributionCheck(),
        ElectricalHumCheck(),
    ]
```

Change `default_manager` to register both sets:

```python
def default_manager() -> SignalCheckManager:
    manager = SignalCheckManager()
    for check in default_time_domain_checks() + default_frequency_domain_checks():
        manager.register(check)
    return manager
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_defaults.py -q`
Expected: PASS (3 passed). The clean-sine case stays OK (all 11 pass) and silence stays FAULT (time-domain critical checks).

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (64 total: 52 prior + 4 feature_prep + 8 frequency_domain; the defaults count assertions changed, not added).

---

## Task 5: Introduce a shared SAMPLE_RATE constant in main.py

The frequency checks derive frequency bins from `window.sample_rate`, so the `44100` literal used when constructing the `AudioWindow` is now load-bearing. Replace the magic numbers with one named constant (additive, behavior-preserving).

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Define the constant**

In `main.py`, find the import line `from app.health.models import AudioWindow, HealthState` and add immediately after it:

```python

SAMPLE_RATE = 44100  # acquisition sample rate (Hz); the whole pipeline runs at this rate
```

- [ ] **Step 2: Use it where the AudioWindow is built**

In `main.py`, find (inside `handle_audio_chunk`):

```python
            window = AudioWindow(samples=self.session_buffer, sample_rate=44100)
```

Replace with:

```python
            window = AudioWindow(samples=self.session_buffer, sample_rate=SAMPLE_RATE)
```

- [ ] **Step 3: Verify parse + additive diff**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('main.py parses OK')"`
Expected: `main.py parses OK`

Run: `git diff main.py | grep '^-' | grep -v '^---'`
Expected: only the single `sample_rate=44100` line shows as replaced (plus the import line is unchanged — the constant is added after it). No inference/feature/model logic removed.

- [ ] **Step 4: Confirm the suite still passes**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (64 total).

---

## Phase 2a Done

Feature Preparation runs once per window and the four frequency-domain checks (F001–F004) flow through the existing indicator/log via `default_pipeline`: electrical hum and broadband noise now surface as WARNING, with spectral measurements (centroid, bandwidth, roll-off, flatness, band ratios, hum ratio) recorded on every report — ready for the Phase 2b panel. Hand back to the owner for review, manual test, and commit. Phase 2b adds the dedicated health panel; Phase 2c adds the configuration-profile system.

---

## Self-Review

- **Spec coverage:** Feature Preparation shared FFT/power/freqs (§3.6) — Task 1. F001 Spectral Shape (centroid/bandwidth/roll-off), F002 Spectral Flatness — Task 2. F003 Band Energy Distribution, F004 Electrical Hum — Task 3. Categories per §10.3 (F001/F003 PRIMARY, F002/F004 SUPPORTING). Shared-representation reuse (§3.12 Efficiency) — checks read `features`, never recompute the FFT. Silence/non-finite handled (no double-reporting; time-domain critical checks own catastrophic failure).
- **Placeholder scan:** no TBD/TODO; every step has full code and exact commands. F003 measurement-only PASS is an explicit, documented Phase 2a decision (calibration adds thresholds in Phase 3), not a stub.
- **Type consistency:** `prepare_features(window) -> dict` with keys `freqs`/`power_spectrum`/`sample_rate`/`n`, consumed by `_usable_spectrum` and every F-check; `decide()`/`SignalCheckResult` unchanged; check class names match between `frequency_domain.py`, `defaults.py`, and the tests; `default_manager()` now registers `default_time_domain_checks() + default_frequency_domain_checks()` (11 total).
- **No regression:** time-domain checks ignore `features`; the pipeline computing it always adds one cheap rFFT (<< 0.5 s). `main.py` change is the constant only — additive.
