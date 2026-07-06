# Phase 3a — Calibration Profile Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a `CalibrationProfile` — per-check, per-measurement statistical reference values — by running the health pipeline over healthy WAV recordings, saved as JSON via a `calibrate.py` CLI.

**Architecture:** A pure NumPy/stdlib `app/health/calibration.py` windows a healthy signal (2.5 s / 0.5 s hop), runs the existing pipeline (development profile = all 11 checks) over each window, accumulates each measurement, and computes statistics (mean/median/std/min/max/p5/p95). A top-level `calibrate.py` CLI handles WAV loading (soundfile/librosa) so the core stays portable. The profile stores **statistics only** — deriving decision thresholds and wiring them into checks is Phase 3b.

**Tech Stack:** Python 3.13, NumPy, stdlib `json`/`dataclasses`; CLI uses soundfile + librosa. pytest.

> **Commit policy:** Owner commits manually. **Do NOT run git commit/add/push.** End at "tests pass."

> **References:** spec `arch_update.md` §6 (Calibration), §5.4 (Calibration Profile object), §6.5 (statistics). Design spec `docs/superpowers/specs/2026-06-23-audio-signal-health-monitoring-phased-implementation-design.md` §5 Phase 3.

> **Phase 3 split (this is 3a of 2):**
> - **3a (this plan):** calibration profile *generation* — the data object, the stats core, and the CLI. Headless; **no change to runtime app behavior.**
> - **3b (next, needs its own short design pass):** *use* the profile — derive thresholds from the stats (percentile / k·std / hybrid — a real design choice), have checks consume calibration-derived thresholds (config `threshold_source: calibration`), the Calibration Evaluation pipeline stage (§3.8), the panel's calibration column, and a "Generate calibration profile" button.

> **Design decisions (3a):** profile stores statistical reference values only (§6.6 — thresholds derived later); statistics set is mean/median/std/min/max/p5/p95 + count (§6.5); calibration runs the **development** profile so all 11 checks' measurements are characterized; core is NumPy-only (WAV loading lives in the CLI); profiles are JSON.

---

## Current State

- `app/health/config.py`: `pipeline_for_profile("development")` → `HealthAnalysisPipeline` with all 11 checks.
- `app/health/models.py`: `AudioWindow(samples, sample_rate)`; `SignalCheckResult.measurements` is a list of `Measurement(name, value, unit)`.
- `HealthReport.check_results` is the per-window list of `SignalCheckResult`.
- WAVs are read elsewhere via `soundfile.read(path, always_2d=True)`. `test_data/audio_signal_health/` holds health recordings (e.g. `broken_mic/`, `mic_issue/`); a healthy set can be the calibration source.
- 81 tests pass.

## File Structure

**Create:**
- `app/health/calibration.py` — `MeasurementStats`, `CalibrationProfile`, `compute_stats`, `iter_windows`, `generate_profile`, `save_profile`, `load_profile`. Pure NumPy/stdlib.
- `calibrate.py` (repo root) — CLI: load WAV(s) → `generate_profile` → `save_profile`. Uses soundfile/librosa.
- `tests/health/test_calibration.py`, `tests/test_calibrate_cli.py`.

---

## Task 1: Calibration data objects + statistics

**Files:**
- Create: `app/health/calibration.py`
- Test: `tests/health/test_calibration.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/health/test_calibration.py`:

```python
from app.health.calibration import CalibrationProfile, MeasurementStats, compute_stats


def test_compute_stats_basic():
    stats = compute_stats([0.0, 1.0, 2.0, 3.0, 4.0])
    assert stats.count == 5
    assert stats.mean == 2.0
    assert stats.median == 2.0
    assert stats.minimum == 0.0
    assert stats.maximum == 4.0
    assert stats.p5 < stats.p95


def test_calibration_profile_defaults():
    p = CalibrationProfile(profile_id="x")
    assert p.profile_id == "x"
    assert p.version == 1
    assert p.sample_rate == 44100
    assert p.statistics == {}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_calibration.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.health.calibration'`.

- [ ] **Step 3: Create `app/health/calibration.py` (objects + stats)**

```python
"""Calibration: build a statistical reference profile from healthy recordings.

Runs the health pipeline over windows of known-good audio and records, per check
measurement, the statistical reference values (spec §6). Stores statistics only —
decision thresholds are derived from these in Phase 3b. Pure stdlib + NumPy; WAV
loading lives in the `calibrate.py` CLI to keep this module portable.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Iterator, Optional

import numpy as np

from app.health.models import AudioWindow


@dataclass
class MeasurementStats:
    count: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    p5: float
    p95: float


@dataclass
class CalibrationProfile:
    profile_id: str
    version: int = 1
    sensor_info: str = ""
    sample_rate: int = 44100
    window_seconds: float = 2.5
    interval_seconds: float = 0.5
    created: str = ""
    window_count: int = 0
    # statistics[check_id][measurement_name] -> MeasurementStats
    statistics: dict = field(default_factory=dict)


def compute_stats(values) -> MeasurementStats:
    arr = np.asarray(values, dtype=np.float64)
    return MeasurementStats(
        count=int(arr.size),
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        minimum=float(np.min(arr)),
        maximum=float(np.max(arr)),
        p5=float(np.percentile(arr, 5)),
        p95=float(np.percentile(arr, 95)),
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_calibration.py -q`
Expected: PASS (2 passed).

---

## Task 2: Windowing + profile generation

**Files:**
- Modify: `app/health/calibration.py` (append)
- Test: `tests/health/test_calibration.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/health/test_calibration.py`:

```python
import numpy as np  # noqa: E402

from app.health.calibration import generate_profile, iter_windows  # noqa: E402

SR = 44100


def _sine(seconds, freq=1000.0, amp=0.3):
    n = int(seconds * SR)
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_iter_windows_count_and_length():
    # 6 s signal, 2.5 s window, 0.5 s hop -> windows at 0,0.5,...,3.5 s = 8 windows.
    windows = list(iter_windows(_sine(6.0), SR, window_seconds=2.5, hop_seconds=0.5))
    assert len(windows) == 8
    assert all(w.shape[0] == int(2.5 * SR) for w in windows)


def test_iter_windows_too_short_yields_nothing():
    assert list(iter_windows(_sine(1.0), SR)) == []


def test_generate_profile_characterizes_all_checks():
    profile = generate_profile([_sine(6.0)], SR, profile_id="test")
    assert profile.window_count == 8
    assert profile.profile_id == "test"
    assert profile.created  # ISO date set
    # All 11 checks contributed measurements (development profile).
    assert set(profile.statistics) >= {
        "T001", "T002", "T003", "T004", "T005", "T006", "T007",
        "F001", "F002", "F003", "F004",
    }
    # A known measurement is present with the right sample count.
    rms = profile.statistics["T002"]["rms"]
    assert rms.count == 8
    assert rms.mean > 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_calibration.py -q`
Expected: FAIL — `ImportError: cannot import name 'iter_windows'`.

- [ ] **Step 3: Append windowing + generation to `app/health/calibration.py`**

```python


def iter_windows(samples, sample_rate, window_seconds=2.5, hop_seconds=0.5) -> Iterator[np.ndarray]:
    """Yield consecutive fixed-length windows (last partial window is dropped)."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    win = int(round(window_seconds * sample_rate))
    hop = int(round(hop_seconds * sample_rate))
    if win <= 0 or hop <= 0 or x.size < win:
        return
    for start in range(0, x.size - win + 1, hop):
        yield x[start:start + win]


def generate_profile(
    signals,
    sample_rate,
    *,
    profile_id,
    sensor_info="",
    pipeline=None,
    window_seconds=2.5,
    hop_seconds=0.5,
) -> CalibrationProfile:
    """Run the pipeline over every window of every signal and accumulate stats.

    `signals` is an iterable of 1-D arrays (each a full recording). `pipeline`
    defaults to the development profile (all checks).
    """
    if pipeline is None:
        from app.health.config import pipeline_for_profile

        pipeline = pipeline_for_profile("development")

    collected: dict[str, dict[str, list[float]]] = {}
    window_count = 0
    for signal in signals:
        for window in iter_windows(signal, sample_rate, window_seconds, hop_seconds):
            report = pipeline.analyze(
                AudioWindow(samples=window, sample_rate=sample_rate)
            )
            window_count += 1
            for result in report.check_results:
                per_check = collected.setdefault(result.check_id, {})
                for m in result.measurements:
                    per_check.setdefault(m.name, []).append(float(m.value))

    statistics = {
        cid: {name: compute_stats(vals) for name, vals in meas.items()}
        for cid, meas in collected.items()
    }
    return CalibrationProfile(
        profile_id=profile_id,
        sensor_info=sensor_info,
        sample_rate=int(sample_rate),
        window_seconds=window_seconds,
        interval_seconds=hop_seconds,
        created=date.today().isoformat(),
        window_count=window_count,
        statistics=statistics,
    )
```

(The `pipeline_for_profile` import is local to avoid any import cycle and keep module-load cheap.)

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_calibration.py -q`
Expected: PASS (5 passed).

---

## Task 3: Save / load the profile as JSON

**Files:**
- Modify: `app/health/calibration.py` (append)
- Test: `tests/health/test_calibration.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/health/test_calibration.py`:

```python
from app.health.calibration import load_profile, save_profile  # noqa: E402


def test_save_load_round_trip(tmp_path):
    profile = generate_profile([_sine(6.0)], SR, profile_id="rt", sensor_info="piezo")
    path = tmp_path / "profile.json"
    save_profile(profile, str(path))
    loaded = load_profile(str(path))
    assert loaded.profile_id == "rt"
    assert loaded.sensor_info == "piezo"
    assert loaded.window_count == profile.window_count
    rms = loaded.statistics["T002"]["rms"]
    assert isinstance(rms, MeasurementStats)
    assert rms.count == profile.statistics["T002"]["rms"].count
    assert rms.mean == profile.statistics["T002"]["rms"].mean
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_calibration.py -q`
Expected: FAIL — `ImportError: cannot import name 'save_profile'`.

- [ ] **Step 3: Append save/load to `app/health/calibration.py`**

```python


def save_profile(profile: CalibrationProfile, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(profile), f, indent=2)


def load_profile(path: str) -> CalibrationProfile:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["statistics"] = {
        cid: {name: MeasurementStats(**s) for name, s in meas.items()}
        for cid, meas in data.get("statistics", {}).items()
    }
    return CalibrationProfile(**data)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_calibration.py -q`
Expected: PASS (6 passed).

---

## Task 4: `calibrate.py` CLI

**Files:**
- Create: `calibrate.py` (repo root)
- Test: `tests/test_calibrate_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_calibrate_cli.py`:

```python
import numpy as np
import soundfile as sf

import calibrate
from app.health.calibration import load_profile

SR = 44100


def test_cli_run_generates_profile(tmp_path):
    # A 6 s healthy-ish sine written to a WAV.
    n = int(6.0 * SR)
    sig = (0.3 * np.sin(2 * np.pi * 1000.0 * np.arange(n) / SR)).astype(np.float32)
    wav = tmp_path / "healthy.wav"
    sf.write(str(wav), sig, SR)
    out = tmp_path / "profile.json"

    calibrate.run(str(wav), str(out), profile_id="cli_test", sensor_info="piezo")

    profile = load_profile(str(out))
    assert profile.profile_id == "cli_test"
    assert profile.window_count == 8
    assert "T002" in profile.statistics
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibrate_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'calibrate'`.

- [ ] **Step 3: Create `calibrate.py`**

```python
"""CLI: generate a CalibrationProfile JSON from healthy WAV recordings.

Usage:
  ./.venv/bin/python calibrate.py --input <folder-or-wav> --output profile.json \
      --profile-id piezo_v1 [--sensor-info "piezo, housing A"]

WAV loading (soundfile/librosa) lives here so app/health stays NumPy-only.
"""
import argparse
import glob
import os

import librosa
import numpy as np
import soundfile as sf

from app.health.calibration import generate_profile, save_profile

TARGET_SR = 44100


def _load_wav(path):
    data, fs = sf.read(path, always_2d=True)
    mono = np.mean(data, axis=1).astype(np.float32)
    if fs != TARGET_SR:
        mono = librosa.resample(mono, orig_sr=fs, target_sr=TARGET_SR).astype(np.float32)
    return mono


def _gather_wavs(input_path):
    if os.path.isdir(input_path):
        return sorted(glob.glob(os.path.join(input_path, "**", "*.wav"), recursive=True))
    return [input_path]


def run(input_path, output_path, profile_id, sensor_info=""):
    paths = _gather_wavs(input_path)
    if not paths:
        raise SystemExit(f"No WAV files found at {input_path}")
    signals = [_load_wav(p) for p in paths]
    profile = generate_profile(
        signals, TARGET_SR, profile_id=profile_id, sensor_info=sensor_info
    )
    save_profile(profile, output_path)
    print(
        f"Wrote {output_path}: {profile.window_count} windows from {len(paths)} "
        f"file(s), {len(profile.statistics)} checks characterized."
    )
    return profile


def main():
    parser = argparse.ArgumentParser(
        description="Generate a calibration profile from healthy WAV recordings."
    )
    parser.add_argument("--input", required=True, help="WAV file or folder (recursive)")
    parser.add_argument("--output", required=True, help="Output profile JSON path")
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--sensor-info", default="")
    args = parser.parse_args()
    run(args.input, args.output, args.profile_id, args.sensor_info)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_calibrate_cli.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (expect 89 total: 81 prior + 6 calibration + 1 CLI + ... confirm the printed count is green with 0 failures).

- [ ] **Step 6: Real-data smoke run (owner, optional)**

Point the CLI at a folder of **healthy** recordings and inspect the JSON:
```bash
./.venv/bin/python calibrate.py --input test_data/audio_signal_health/<healthy_set> \
    --output models/9_1_2/calibration.json --profile-id piezo_9_1_2 --sensor-info "piezo"
```
Expected: a JSON with `statistics` for all 11 checks; sanity-check that e.g. `T002.rms` mean/p5/p95 look reasonable for a healthy signal.

---

## Phase 3a Done

A calibration profile (per-check, per-measurement statistics) can now be generated from healthy recordings via `calibrate.py` and saved/loaded as JSON. No runtime app behavior changed yet. Hand back to the owner for review, manual test, and commit. **Phase 3b** (a short design pass first) derives decision thresholds from these statistics, wires checks to use them, adds the Calibration Evaluation stage, and surfaces calibration in the UI.

---

## Self-Review

- **Spec coverage (§6):** runs the same pipeline over healthy recordings (§6.3) — `generate_profile` uses `pipeline_for_profile("development")`; per-measurement statistics mean/median/std/min/max/p5/p95 (§6.5) — `compute_stats`/`MeasurementStats`; stores statistical reference values, not thresholds (§6.6) — profile has `statistics`, no thresholds (deferred to 3b); CalibrationProfile metadata id/version/sensor/acquisition config/date (§5.4) — `CalibrationProfile` fields; reusable JSON profile (§6.7) — `save_profile`/`load_profile`. Threshold generation (§6.6) and recalibration UX are Phase 3b.
- **Placeholder scan:** no TBD/TODO; full code in every step; concrete commands.
- **Type consistency:** `compute_stats(values) -> MeasurementStats`; `iter_windows(samples, sample_rate, window_seconds, hop_seconds)`; `generate_profile(signals, sample_rate, *, profile_id, sensor_info, pipeline, window_seconds, hop_seconds) -> CalibrationProfile`; `save_profile(profile, path)` / `load_profile(path) -> CalibrationProfile` (reconstructs `MeasurementStats`); CLI `run(input_path, output_path, profile_id, sensor_info="")` matches the test call. `statistics[check_id][measurement_name] = MeasurementStats` is consistent across generate/save/load and the tests.
- **Portability:** `app/health/calibration.py` imports only stdlib + NumPy + `app.health.*`; WAV/soundfile/librosa live in the root `calibrate.py`. The `pipeline_for_profile` import is function-local.
