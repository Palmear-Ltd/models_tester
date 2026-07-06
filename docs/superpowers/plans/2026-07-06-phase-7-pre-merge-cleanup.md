# Phase 7 — Pre-merge Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the three deferred follow-ups (fmin/fmax UI, report persistence, full-covariance Mahalanobis anomaly detection) so the Audio Signal Health Monitoring subsystem can merge to `main`.

**Architecture:** Three independent sub-phases. 7a threads two existing classifier params through the Settings UI. 7b adds a pure serialization module in the portable core plus file-writing triggers in `main.py`. 7c replaces the diagonal anomaly detector with a true Mahalanobis distance backed by an extended calibration profile (mean vector + covariance matrix) and a chi-square threshold, then regenerates the one existing profile.

**Tech Stack:** Python 3.13, NumPy, stdlib `math`/`json`, Tkinter (UI), pytest. Tests run under `.venv` (numpy + pytest only).

## Global Constraints

- **`app/health/` imports ONLY stdlib + NumPy.** Never import tester/UI/tflite/librosa/matplotlib. Portable to Flutter later. Reports/profiles are duck-typed across module boundaries. (The chi-square helper uses stdlib `math` only — OK.)
- **Health is additive.** Anomaly/serialization changes must never block, slow, or alter classification.
- **Checks declare `category` as a class attribute ONLY** — not relevant to these tasks, but do not touch check `run` signatures.
- **Anomaly sets confidence only, never state.** `fusion.decide` keeps that contract. No profile / no covariance → unchanged behavior (anomaly returns `None`).
- **Do NOT run `git add`/`commit`/`push`.** The owner commits manually after review. Leave all changes in the working tree. (No commit steps appear in this plan by design.)
- **Run tests with** `.venv/bin/python -m pytest tests/ -q`. Baseline before Phase 7: **137 passing**.
- **Verify NumPy claims** against the installed version before acting — subagents hallucinate deprecations (`np.ptp`, `np.hanning`, `np.cov` are all valid).

---

### Task 1: Wire fmin/fmax into Settings UI (7a)

**Files:**
- Modify: `main.py` (add tk vars near line 70-75; pass into `extract_features` at ~887; add to saved-results dict at ~857)
- Modify: `app/ui/settings_dialog.py` (`_build_preprocessing_tab`, after the Up Cut field at line 101)
- Modify: `app/audio/features.py:24` (remove the `#TODO` comment)
- Test: `tests/test_settings_dialog.py`

**Interfaces:**
- Produces: `app.fmin_var` (tk.DoubleVar, default 50.0), `app.fmax_var` (tk.DoubleVar, default 10000.0); `extract_features(..., fmin=<val>, fmax=<val>)`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_settings_dialog.py`. This asserts the dialog references the new vars without needing a display, by recording attribute access on a fake app.

```python
def test_preprocessing_tab_binds_fmin_fmax():
    # SettingsDialog builds its widgets against app.<var> attributes. Confirm the
    # preprocessing tab references fmin/fmax vars by exercising _build_preprocessing_tab
    # with a recording fake that captures which app attributes are read.
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        import pytest
        pytest.skip("no display available")
    root.withdraw()
    from tkinter import ttk

    class _App:
        def __init__(self):
            self.low_cut_var = tk.DoubleVar(value=500.0)
            self.up_cut_var = tk.DoubleVar(value=8000.0)
            self.sub_win_size_var = tk.DoubleVar(value=0.05)
            self.sub_hop_size_var = tk.DoubleVar(value=0.025)
            self.n_mels_var = tk.IntVar(value=32)
            self.seq_len_var = tk.IntVar(value=98)
            self.use_filter_var = tk.BooleanVar(value=True)
            self.fmin_var = tk.DoubleVar(value=50.0)
            self.fmax_var = tk.DoubleVar(value=10000.0)
            self.is_one_shot_model = False
            self.root = root

    app = _App()
    dlg = SettingsDialog(app)
    nb = ttk.Notebook(root)
    dlg._build_preprocessing_tab(nb)  # must not raise; references app.fmin_var/app.fmax_var
    assert app.fmin_var.get() == 50.0
    assert app.fmax_var.get() == 10000.0
    root.destroy()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_settings_dialog.py::test_preprocessing_tab_binds_fmin_fmax -v`
Expected: FAIL (either `AttributeError` inside `_build_preprocessing_tab` once it references `app.fmin_var`, or the test skips if no display — in CI-without-display it skips, which is acceptable; the real gate is Steps 3-6 + the import check).

- [ ] **Step 3: Add the two entry fields** in `app/ui/settings_dialog.py._build_preprocessing_tab`, immediately after the Up Cut rows (after line 101). Shift the subsequent rows (FFT Win Size etc.) down by 2:

```python
        ttk.Label(tab, text="Mel fmin (Hz):").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.fmin_var, width=15).grid(row=3, column=1, sticky="w", padx=5)

        ttk.Label(tab, text="Mel fmax (Hz):").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.fmax_var, width=15).grid(row=4, column=1, sticky="w", padx=5)
```

Then renumber the remaining grid rows in that method so they stay in order (FFT Win Size → row 5, Hop Size → row 6, Mel Bands → row 7, Seq Len → row 8).

- [ ] **Step 4: Add the tk vars** in `main.py` right after line 72 (`self.sub_win_size_var = ...`):

```python
        self.fmin_var = tk.DoubleVar(value=50.0)
        self.fmax_var = tk.DoubleVar(value=10000.0)
```

- [ ] **Step 5: Thread into inference** — in `main.py.run_inference`, extend the `extract_features` call (currently ending at line 896 `use_filter=...`) to pass the two params:

```python
            specs = self.processor.extract_features(
                audio_data,
                sr=44100,
                n_mels=self.model_n_mels,
                seq_len=self.model_seq_len,
                low_cut=self.low_cut_var.get(),
                up_cut=self.up_cut_var.get(),
                fmin=self.fmin_var.get(),
                fmax=self.fmax_var.get(),
                sub_win_size_sec=self.sub_win_size_var.get(),
                sub_hop_size_sec=self.sub_hop_size_var.get(),
                use_filter=self.use_filter_var.get()
            )
```

Also add to the saved-results dict (after line 858 `"up_cut": ...`):

```python
                "fmin": self.fmin_var.get(),
                "fmax": self.fmax_var.get(),
```

- [ ] **Step 6: Remove the TODO** at `app/audio/features.py:24` — delete the line `    #TODO: maybe add and wire fmin and fmax in the settings UI window`. Leave `extract_features`'s signature (`fmin=50, fmax=10000`) unchanged.

- [ ] **Step 7: Verify headless**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); ast.parse(open('app/ui/settings_dialog.py').read())"`
Then: `.venv/bin/python -m pytest tests/test_settings_dialog.py -q`
Expected: parse OK; settings-dialog tests pass or skip (no display). Full suite still 137 (Task 1 adds no headless-guaranteed test since it needs Tk).

---

### Task 2: Report serialization module (7b, pure core)

**Files:**
- Create: `app/health/serialization.py`
- Test: `tests/health/test_serialization.py`

**Interfaces:**
- Produces: `startup_result_to_dict(result) -> dict`, `anomaly_event_to_dict(anomaly, *, source, timestamp) -> dict`. Both return JSON-serializable dicts. Duck-typed inputs (StartupResult from `startup.py`; AnomalyResult from `anomaly.py`).

- [ ] **Step 1: Write the failing tests** in `tests/health/test_serialization.py`:

```python
import json
from app.health.serialization import startup_result_to_dict, anomaly_event_to_dict


class _Sys:
    passed = True
    errors = []
    warnings = ["No calibration profile loaded"]


class _Sig:
    total = 40
    ok = 38
    warning = 1
    fault = 1
    check_failures = {"T002": 2}


class _Decision:
    value = "WARNING"


class _Startup:
    decision = _Decision()
    system = _Sys()
    signal = _Sig()
    summary = "WARNING: 38 OK / 1 WARNING / 1 FAULT of 40 windows"


class _Anomaly:
    distance = 5.2
    threshold = 4.0
    is_anomalous = True
    contributors = [("T002.rms", 3.1), ("F001.spectral_centroid", 1.2)]
    confidence = 0.35


def test_startup_result_to_dict_is_json_serializable():
    d = startup_result_to_dict(_Startup())
    assert d["decision"] == "WARNING"
    assert d["signal"]["total"] == 40
    assert d["signal"]["check_failures"] == {"T002": 2}
    assert d["system"]["warnings"] == ["No calibration profile loaded"]
    assert d["summary"].startswith("WARNING")
    json.dumps(d)  # must not raise


def test_anomaly_event_to_dict_is_json_serializable():
    d = anomaly_event_to_dict(_Anomaly(), source="mic", timestamp="20260706_120000")
    assert d["distance"] == 5.2
    assert d["threshold"] == 4.0
    assert d["is_anomalous"] is True
    assert d["confidence"] == 0.35
    assert d["source"] == "mic"
    assert d["timestamp"] == "20260706_120000"
    assert d["contributors"] == [["T002.rms", 3.1], ["F001.spectral_centroid", 1.2]]
    json.dumps(d)  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/health/test_serialization.py -v`
Expected: FAIL with `ModuleNotFoundError: app.health.serialization`.

- [ ] **Step 3: Implement `app/health/serialization.py`:**

```python
"""Serialize duck-typed health reports to JSON-ready dicts (stdlib only).

File writing lives in the app layer (main.py); this module stays pure and
portable so it is testable under the numpy-only venv.
"""
from __future__ import annotations


def startup_result_to_dict(result) -> dict:
    system = result.system
    signal = result.signal
    return {
        "decision": result.decision.value,
        "summary": result.summary,
        "system": {
            "passed": bool(system.passed),
            "errors": list(system.errors),
            "warnings": list(system.warnings),
        },
        "signal": {
            "total": signal.total,
            "ok": signal.ok,
            "warning": signal.warning,
            "fault": signal.fault,
            "check_failures": dict(signal.check_failures),
        },
    }


def anomaly_event_to_dict(anomaly, *, source, timestamp) -> dict:
    return {
        "timestamp": timestamp,
        "source": source,
        "distance": float(anomaly.distance),
        "threshold": float(anomaly.threshold),
        "is_anomalous": bool(anomaly.is_anomalous),
        "confidence": float(anomaly.confidence),
        "contributors": [[label, float(v)] for label, v in anomaly.contributors],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/health/test_serialization.py -v`
Expected: PASS (2 tests).

---

### Task 3: Persist reports from main.py (7b, app wiring)

**Files:**
- Modify: `main.py` (import serialization + StartupResult already imported; add a `_write_report` helper; call it in `_show_validation_result` and in the anomaly rising-edge branch of `_update_health_indicator`)
- Modify: `.gitignore` (add `reports/`)

**Interfaces:**
- Consumes: `startup_result_to_dict`, `anomaly_event_to_dict` from Task 2.

- [ ] **Step 1: Add `reports/` to `.gitignore`** — append a line `reports/` to `.gitignore`.

- [ ] **Step 2: Import the serializers** in `main.py` near the other `app.health` imports (the `from app.health.startup import ...` line, ~24):

```python
from app.health.serialization import startup_result_to_dict, anomaly_event_to_dict
```

- [ ] **Step 3: Add a `_write_report` helper** on `ModelsTesterApp` (place it just above `_show_validation_result`, ~line 711):

```python
    def _write_report(self, kind, payload):
        """Write a health report dict to reports/<kind>_<timestamp>.json (best-effort)."""
        try:
            os.makedirs("reports", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join("reports", f"{kind}_{ts}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self.log(f"[report] wrote {path}")
        except Exception as e:
            self.log(f"[report] write error: {e}")
```

(`os`, `json`, and `datetime` are already imported in main.py — confirm at implementation time via `grep -n "^import\|^from" main.py`.)

- [ ] **Step 4: Persist the startup report** — in `_show_validation_result`, after the `self.log(f"[validation] {result.summary}")` line (~718):

```python
        self._write_report("startup", startup_result_to_dict(result))
```

- [ ] **Step 5: Persist the anomaly report** — in `_update_health_indicator`, inside the rising-edge branch (after the existing `self.log(f"[anomaly] ...")` at ~749):

```python
            source = "wav" if self.input_type_var.get() == "file" else "mic"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._write_report("anomaly", anomaly_event_to_dict(anomaly, source=source, timestamp=ts))
```

(Confirm the file-vs-mic discriminator: check `grep -n "input_type_var" main.py` for the exact value used for WAV input; adjust the literal if it is not `"file"`.)

- [ ] **Step 6: Verify headless**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read())"`
Then: `.venv/bin/python -m pytest tests/ -q`
Expected: parse OK; suite still green (139 now: +2 from Task 2).

---

### Task 4: Chi-square helper (7c foundation)

**Files:**
- Create: `app/health/chi2.py`
- Test: `tests/health/test_chi2.py`

**Interfaces:**
- Produces: `chi2_ppf(p, df) -> float` (inverse CDF / critical value), `chi2_cdf(x, df) -> float`. Pure `math` — no NumPy, no SciPy.

- [ ] **Step 1: Write the failing tests** in `tests/health/test_chi2.py` (reference values from standard chi-square tables):

```python
import math
from app.health.chi2 import chi2_ppf, chi2_cdf


def test_ppf_matches_known_values():
    assert math.isclose(chi2_ppf(0.95, 1), 3.8415, abs_tol=1e-2)
    assert math.isclose(chi2_ppf(0.95, 2), 5.9915, abs_tol=1e-2)
    assert math.isclose(chi2_ppf(0.999, 1), 10.828, abs_tol=1e-2)
    assert math.isclose(chi2_ppf(0.95, 10), 18.307, abs_tol=1e-2)


def test_cdf_ppf_roundtrip():
    for df in (1, 3, 8, 14):
        x = chi2_ppf(0.9, df)
        assert math.isclose(chi2_cdf(x, df), 0.9, abs_tol=1e-4)


def test_cdf_monotonic_and_bounded():
    assert chi2_cdf(0.0, 5) == 0.0
    assert 0.0 <= chi2_cdf(1.0, 5) <= chi2_cdf(20.0, 5) <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/health/test_chi2.py -v`
Expected: FAIL with `ModuleNotFoundError: app.health.chi2`.

- [ ] **Step 3: Implement `app/health/chi2.py`** (regularized lower incomplete gamma via series / continued fraction; bisection for the inverse):

```python
"""Chi-square CDF and inverse CDF in pure stdlib math (no SciPy/NumPy).

chi2_cdf(x, df) = P(df/2, x/2), the regularized lower incomplete gamma.
chi2_ppf inverts it by bisection. Adequate precision for anomaly thresholds.
"""
from __future__ import annotations

import math

_MAXIT = 300
_EPS = 1e-14
_FPMIN = 1e-300


def _gammap(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x)."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        # Series representation.
        ap = a
        s = 1.0 / a
        d = s
        for _ in range(_MAXIT):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * _EPS:
                break
        return s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # Continued fraction for Q(a, x) = 1 - P(a, x).
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _MAXIT):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def chi2_cdf(x: float, df: int) -> float:
    if x <= 0.0:
        return 0.0
    return _gammap(df / 2.0, x / 2.0)


def chi2_ppf(p: float, df: int) -> float:
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return math.inf
    lo, hi = 0.0, max(1.0, float(df))
    while chi2_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/health/test_chi2.py -v`
Expected: PASS (3 tests).

---

### Task 5: Extend calibration profile with mean vector + covariance (7c)

**Files:**
- Modify: `app/health/calibration.py` (`CalibrationProfile`, `generate_profile`, `load_profile`)
- Test: `tests/health/test_calibration.py`

**Interfaces:**
- Produces: `CalibrationProfile.version == 2`, `.feature_index: list[list[str]]` (ordered `[check_id, name]` pairs), `.mean_vector: list[float]`, `.covariance: list[list[float]]`. `generate_profile(...)` populates all three. `save_profile`/`load_profile` round-trip them (plain JSON lists; no NumPy in the serialized form).

- [ ] **Step 1: Write the failing tests** — append to `tests/health/test_calibration.py`:

```python
import numpy as np
from app.health.calibration import (
    CalibrationProfile, generate_profile, save_profile, load_profile,
)
from app.health.models import (
    AudioWindow, Measurement, SignalCheckResult, CheckStatus, HealthReport, HealthState,
)


class _FakePipeline:
    """Emits two correlated measurements per window so covariance is non-diagonal."""
    def __init__(self):
        self._i = 0

    def analyze(self, window):
        self._i += 1
        a = float(self._i % 7)
        b = a * 2.0 + 1.0  # perfectly correlated with a
        res = SignalCheckResult(
            check_id="T002", check_name="rms", status=CheckStatus.PASS, executed=True,
            measurements=[Measurement("rms", a), Measurement("peak", b)],
        )
        return HealthReport(timestamp=0.0, window_id="x", check_results=[res],
                            final_state=HealthState.OK)


def test_generate_profile_stores_mean_and_covariance():
    sig = np.zeros(int(44100 * 6.0), dtype=np.float32)  # 6 s -> several windows
    profile = generate_profile([sig], 44100, profile_id="test_v2", pipeline=_FakePipeline())
    assert profile.version == 2
    assert [tuple(k) for k in profile.feature_index] == [("T002", "peak"), ("T002", "rms")] \
        or [tuple(k) for k in profile.feature_index] == [("T002", "rms"), ("T002", "peak")]
    D = len(profile.feature_index)
    assert len(profile.mean_vector) == D
    assert len(profile.covariance) == D and all(len(row) == D for row in profile.covariance)


def test_profile_v2_round_trips(tmp_path):
    sig = np.zeros(int(44100 * 6.0), dtype=np.float32)
    profile = generate_profile([sig], 44100, profile_id="test_v2", pipeline=_FakePipeline())
    path = tmp_path / "p.json"
    save_profile(profile, str(path))
    loaded = load_profile(str(path))
    assert loaded.version == 2
    assert loaded.feature_index == profile.feature_index
    assert loaded.mean_vector == profile.mean_vector
    assert loaded.covariance == profile.covariance
    # statistics (used by calibration_eval) still present
    assert "T002" in loaded.statistics
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/health/test_calibration.py -k "covariance or round_trips" -v`
Expected: FAIL (`CalibrationProfile` has no `feature_index`/`mean_vector`/`covariance`; `version` is 1).

- [ ] **Step 3: Extend the dataclass** in `app/health/calibration.py` — add fields to `CalibrationProfile` (keep existing fields; bump default version):

```python
@dataclass
class CalibrationProfile:
    profile_id: str
    version: int = 2
    sensor_info: str = ""
    sample_rate: int = 44100
    window_seconds: float = 2.5
    interval_seconds: float = 0.5
    created: str = ""
    window_count: int = 0
    # statistics[check_id][measurement_name] -> MeasurementStats (used by calibration_eval)
    statistics: dict[str, dict[str, MeasurementStats]] = field(default_factory=dict)
    # Full-covariance anomaly model (Phase 7c):
    feature_index: list = field(default_factory=list)      # ordered [check_id, name] pairs
    mean_vector: list = field(default_factory=list)        # length D
    covariance: list = field(default_factory=list)         # D x D nested lists
```

- [ ] **Step 4: Compute mean/covariance in `generate_profile`.** Collect per-window full vectors alongside the existing per-measurement lists, then build the model from windows that contain every feature (the key intersection keeps the matrix rectangular). Replace the body of `generate_profile` from the `collected` loop through the `return` with:

```python
    collected: dict[str, dict[str, list[float]]] = {}
    per_window: list[dict[tuple, float]] = []  # {(check_id, name): value} per window
    window_count = 0
    for signal in signals:
        for window in iter_windows(signal, sample_rate, window_seconds, hop_seconds):
            report = pipeline.analyze(
                AudioWindow(samples=window, sample_rate=sample_rate)
            )
            window_count += 1
            wd: dict[tuple, float] = {}
            for result in report.check_results:
                per_check = collected.setdefault(result.check_id, {})
                for m in result.measurements:
                    per_check.setdefault(m.name, []).append(float(m.value))
                    wd[(result.check_id, m.name)] = float(m.value)
            per_window.append(wd)

    statistics = {
        cid: {name: compute_stats(vals) for name, vals in meas.items()}
        for cid, meas in collected.items()
    }

    # Feature model: keys present in EVERY window (intersection => complete matrix), sorted.
    feature_index_tuples: list[tuple] = []
    mean_vector: list = []
    covariance: list = []
    if per_window:
        common = set(per_window[0])
        for wd in per_window[1:]:
            common &= set(wd)
        feature_index_tuples = sorted(common)
        if feature_index_tuples:
            matrix = np.array(
                [[wd[k] for k in feature_index_tuples] for wd in per_window],
                dtype=np.float64,
            )
            mean_vector = matrix.mean(axis=0).tolist()
            if matrix.shape[0] > 1:
                cov = np.cov(matrix, rowvar=False)
            else:
                cov = np.zeros((matrix.shape[1], matrix.shape[1]))
            covariance = np.atleast_2d(cov).tolist()

    return CalibrationProfile(
        profile_id=profile_id,
        sensor_info=sensor_info,
        sample_rate=int(sample_rate),
        window_seconds=window_seconds,
        interval_seconds=hop_seconds,
        created=date.today().isoformat(),
        window_count=window_count,
        statistics=statistics,
        feature_index=[list(k) for k in feature_index_tuples],
        mean_vector=mean_vector,
        covariance=covariance,
    )
```

- [ ] **Step 5: Update `load_profile`** so the new list fields survive the round-trip. The current `CalibrationProfile(**data)` already passes them through since they are plain lists — but `load_profile` must tolerate a v1 profile lacking the keys. Replace `load_profile` with:

```python
def load_profile(path: str) -> CalibrationProfile:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["statistics"] = {
        cid: {name: MeasurementStats(**s) for name, s in meas.items()}
        for cid, meas in data.get("statistics", {}).items()
    }
    # v1 profiles predate the covariance model; defaults leave anomaly detection a no-op.
    data.setdefault("feature_index", [])
    data.setdefault("mean_vector", [])
    data.setdefault("covariance", [])
    return CalibrationProfile(**data)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/health/test_calibration.py -q`
Expected: PASS (existing calibration tests + 2 new).

---

### Task 6: Full-covariance Mahalanobis anomaly + fusion/UI wording (7c)

**Files:**
- Rewrite: `app/health/anomaly.py`
- Modify: `app/health/fusion.py` (anomaly note wording, ~line 96-103)
- Modify: `main.py._update_health_indicator` (anomaly log wording, ~line 748-749)
- Rewrite: `tests/health/test_anomaly.py`

**Interfaces:**
- Consumes: `chi2_ppf` from Task 4; `profile.feature_index`, `profile.mean_vector`, `profile.covariance` from Task 5.
- Produces: `detect_anomaly(results, profile, *, p=0.001) -> AnomalyResult | None`. `AnomalyResult` fields unchanged (`distance`, `threshold`, `is_anomalous`, `contributors`, `confidence`), but `distance` is now the Mahalanobis distance `d`, `threshold` is `sqrt(chi2_ppf(1-p, df))`, and each `contributors` entry is `(label, contribution_to_d2)`.

- [ ] **Step 1: Rewrite `tests/health/test_anomaly.py`** for the covariance model:

```python
import numpy as np
from app.health.anomaly import AnomalyResult, detect_anomaly


class _M:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class _R:
    def __init__(self, check_id, measurements):
        self.check_id = check_id
        self.measurements = measurements


class _Profile:
    def __init__(self, feature_index, mean_vector, covariance):
        self.feature_index = feature_index
        self.mean_vector = mean_vector
        self.covariance = covariance


def _profile():
    # Two independent features, unit-ish variance.
    return _Profile(
        feature_index=[["T002", "rms"], ["F001", "spectral_centroid"]],
        mean_vector=[0.2, 1000.0],
        covariance=[[0.0025, 0.0], [0.0, 10000.0]],  # std 0.05 and 100
    )


def test_at_mean_is_not_anomalous():
    results = [_R("T002", [_M("rms", 0.2)]), _R("F001", [_M("spectral_centroid", 1000.0)])]
    r = detect_anomaly(results, _profile())
    assert isinstance(r, AnomalyResult)
    assert r.distance < 1e-6
    assert r.is_anomalous is False
    assert abs(r.confidence - 1.0) < 1e-6


def test_far_out_is_anomalous():
    # rms 12 std out, centroid 1 std out -> d^2 = 144 + 1 = 145, d ~ 12.04
    results = [_R("T002", [_M("rms", 0.8)]), _R("F001", [_M("spectral_centroid", 1100.0)])]
    r = detect_anomaly(results, _profile(), p=0.001)
    assert r.is_anomalous is True
    assert r.distance > r.threshold
    assert r.confidence == 0.0
    assert r.contributors[0][0] == "T002.rms"  # dominant contributor


def test_correlation_caught_that_diagonal_would_miss():
    # Two strongly correlated features; a point that moves against the correlation
    # is far in Mahalanobis distance even though each marginal z is modest.
    profile = _Profile(
        feature_index=[["C", "a"], ["C", "b"]],
        mean_vector=[0.0, 0.0],
        covariance=[[1.0, 0.9], [0.9, 1.0]],
    )
    results = [_R("C", [_M("a", 2.0), _M("b", -2.0)])]  # opposes the +0.9 correlation
    r = detect_anomaly(profile=profile, results=results, p=0.05)
    # Diagonal z-distance would be sqrt((4+4)/2)=2.0; full Mahalanobis is much larger.
    assert r.distance > 4.0
    assert r.is_anomalous is True


def test_returns_none_without_covariance():
    profile = _Profile(feature_index=[], mean_vector=[], covariance=[])
    results = [_R("T002", [_M("rms", 0.2)])]
    assert detect_anomaly(results, profile) is None


def test_singular_covariance_is_regularized_not_crash():
    profile = _Profile(
        feature_index=[["C", "a"], ["C", "b"]],
        mean_vector=[0.0, 0.0],
        covariance=[[1.0, 1.0], [1.0, 1.0]],  # singular
    )
    results = [_R("C", [_M("a", 1.0), _M("b", 1.0)])]
    r = detect_anomaly(results, profile)
    assert isinstance(r, AnomalyResult)
    assert np.isfinite(r.distance)


def test_missing_feature_uses_submatrix():
    # Only one of the two profile features is present this window.
    results = [_R("T002", [_M("rms", 0.2)])]  # no spectral_centroid
    r = detect_anomaly(results, _profile())
    assert isinstance(r, AnomalyResult)
    assert r.distance < 1e-6  # rms at its mean -> zero distance over the 1-D submatrix
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/health/test_anomaly.py -v`
Expected: FAIL (current `detect_anomaly` uses `profile.statistics` and a `threshold=` kwarg, not the covariance model).

- [ ] **Step 3: Rewrite `app/health/anomaly.py`:**

```python
"""Anomaly detection (spec Ch. 6 / Phase 7c): full-covariance Mahalanobis distance
of a window's measurements from the calibration profile, thresholded by a
chi-square critical value and turned into a confidence.

The profile is duck-typed: ``profile.feature_index`` (ordered [check_id, name]
pairs), ``profile.mean_vector`` (length D), ``profile.covariance`` (D x D). Results
are duck-typed (``check_id`` + ``measurements`` of ``name``/``value``). Pure NumPy
plus the stdlib chi-square helper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.health.chi2 import chi2_ppf


@dataclass
class AnomalyResult:
    distance: float          # Mahalanobis distance d
    threshold: float         # sqrt(chi2 critical value) at (1 - p, df)
    is_anomalous: bool
    contributors: list       # top (label, contribution-to-d^2) by |contribution|
    # Distance-from-profile normality, not a probability of good health: 1.0 at the
    # profile centroid, 0.5 at the threshold, 0.0 at twice the threshold.
    confidence: float


def detect_anomaly(results, profile, *, p: float = 0.001):
    feature_index = getattr(profile, "feature_index", None)
    mean_vector = getattr(profile, "mean_vector", None)
    covariance = getattr(profile, "covariance", None)
    if not feature_index or not mean_vector or not covariance:
        return None  # no covariance model (e.g. v1 profile) -> unchanged behavior

    # Map this window's measurements by (check_id, name).
    value_by_key = {}
    for result in results:
        for m in result.measurements:
            value_by_key[(result.check_id, m.name)] = float(m.value)

    keys = [tuple(k) for k in feature_index]
    mu_full = np.asarray(mean_vector, dtype=np.float64)
    cov_full = np.asarray(covariance, dtype=np.float64)

    # Subselect the dimensions actually present (and finite) this window.
    idx = [
        i for i, key in enumerate(keys)
        if key in value_by_key and np.isfinite(value_by_key[key])
    ]
    if not idx:
        return None

    x = np.array([value_by_key[keys[i]] for i in idx], dtype=np.float64)
    mu = mu_full[idx]
    cov = cov_full[np.ix_(idx, idx)]

    # Regularize for singular / ill-conditioned covariance.
    d = cov.shape[0]
    ridge = 1e-9 * (np.trace(cov) / d if d else 1.0)
    cov = cov + (ridge if ridge > 0 else 1e-12) * np.eye(d)

    diff = x - mu
    try:
        sol = np.linalg.solve(cov, diff)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(cov) @ diff

    per_dim = diff * sol  # each dim's contribution to d^2
    d2 = float(max(0.0, np.sum(per_dim)))
    distance = float(np.sqrt(d2))

    crit2 = chi2_ppf(1.0 - p, d)
    threshold = float(np.sqrt(crit2))
    is_anomalous = d2 > crit2
    confidence = max(0.0, 1.0 - distance / (2.0 * threshold)) if threshold > 0 else 0.0

    order = np.argsort(-np.abs(per_dim))[:3]
    contributors = [
        (f"{keys[idx[i]][0]}.{keys[idx[i]][1]}", float(per_dim[i])) for i in order
    ]

    return AnomalyResult(
        distance=distance,
        threshold=threshold,
        is_anomalous=is_anomalous,
        contributors=contributors,
        confidence=confidence,
    )
```

- [ ] **Step 4: Run the anomaly tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/health/test_anomaly.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Update the fusion note wording** in `app/health/fusion.py` (the anomaly block at ~96-103) so the contributor label reads as a contribution, not a z-score:

```python
    # Fold in the anomaly result: it sets confidence (calibration owns state).
    if anomaly_result is not None:
        confidence = anomaly_result.confidence
        label = "ANOMALOUS" if anomaly_result.is_anomalous else "normal"
        note = f"anomaly distance {anomaly_result.distance:.1f} ({label})"
        if anomaly_result.contributors:
            top_label, top_c = anomaly_result.contributors[0]
            note += f", {top_label} contrib={top_c:.1f}"
        summary = f"{summary} | {note}" if summary else note
```

- [ ] **Step 6: Update the anomaly log wording** in `main.py._update_health_indicator` (~748-749):

```python
            top_label, top_c = anomaly.contributors[0] if anomaly.contributors else ("", 0.0)
            self.log(f"[anomaly] distance {anomaly.distance:.1f} — {top_label} contrib={top_c:.1f}")
```

- [ ] **Step 7: Run fusion + pipeline tests, then the full suite**

Run: `.venv/bin/python -m pytest tests/health/test_fusion.py tests/health/test_pipeline.py -v`
Then: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass. If `test_fusion.py` or `test_pipeline.py` hard-code an anomaly stub with the old `statistics`/`threshold` shape or the `z=` wording, update those fixtures to the covariance-model / `contrib=` wording (the fusion note change is cosmetic; adjust the asserted substring). Report the final passing count.

- [ ] **Step 8: Verify main.py still imports/parses**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read())"`
Expected: no output (parse OK).

---

### Task 7: Regenerate the 9_1_2 calibration profile + final verification

**Files:**
- Regenerate: `models/9_1_2/calibration.json`

> This task runs full tooling; the operator (Claude in this session) performs it — it is **not** a TDD subagent task. Requires only the numpy-only `.venv` (the health feature path needs no tflite/librosa; but `calibrate.py` imports `soundfile`/`librosa` for WAV loading — if those are absent from `.venv`, load the WAVs with a small stdlib `wave`-based shim or run under the full environment).

- [ ] **Step 1: Regenerate the profile** from the sanded-needle recordings, preserving the profile id and sensor info:

```bash
.venv/bin/python calibrate.py \
  --input test_data/audio_signal_health/sanded_needle_1 \
  --output models/9_1_2/calibration.json \
  --profile-id sanded_piezo_9_1_2 \
  --sensor-info piezo
```

Expected: prints `Wrote models/9_1_2/calibration.json: <N> windows ... checks characterized.`

- [ ] **Step 2: Confirm the new profile carries the covariance model**

```bash
.venv/bin/python -c "import json; d=json.load(open('models/9_1_2/calibration.json')); print('version', d['version'], 'D', len(d['feature_index']), 'cov', len(d['covariance']))"
```

Expected: `version 2`, `D` = number of features, `cov` = D (square matrix).

- [ ] **Step 3: Smoke-test load + anomaly end-to-end** on the regenerated profile:

```bash
.venv/bin/python -c "
from app.health.calibration import load_profile
from app.health.config import pipeline_for_profile
from app.health.models import AudioWindow
import numpy as np
prof = load_profile('models/9_1_2/calibration.json')
pipe = pipeline_for_profile('development', calibration_profile=prof)
rep = pipe.analyze(AudioWindow(samples=np.zeros(int(44100*2.5), dtype=np.float32), sample_rate=44100))
print('state', rep.final_state, 'conf', round(rep.confidence, 3), 'anom', rep.anomaly_result is not None)
"
```

Expected: runs without error; prints a state, a confidence, and `anom True` (a profile is loaded so anomaly runs).

- [ ] **Step 4: Full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green. Record the final count (baseline 137 + Task 2's 2 + Task 4's 3 + Task 5's 2 + Task 6 net for the rewritten anomaly file). Report it.

---

## Self-Review notes

- **Spec coverage:** 7a → Task 1; 7b → Tasks 2-3; 7c profile format → Task 5; 7c Mahalanobis + chi-square → Tasks 4, 6; profile regeneration → Task 7. All spec sections covered.
- **Type consistency:** `AnomalyResult` field names unchanged across fusion/main consumers; `detect_anomaly` signature changes from `threshold=` to `p=` (all call sites are `pipeline._detect_anomalies` which passes neither, and tests — both updated). `feature_index`/`mean_vector`/`covariance` names identical in calibration.py (producer) and anomaly.py (consumer).
- **No commit steps** — per repo workflow, the owner commits manually.
- **Watch item for the implementer of Task 6 Step 7:** grep `tests/health/test_fusion.py` and `test_pipeline.py` for any anomaly stub using the old `.statistics`/`threshold` shape or a `z=` assertion and update it; do not weaken a real assertion, only realign wording/shape.
