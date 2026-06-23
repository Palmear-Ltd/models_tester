# Phase 0 — Foundation & Integration Seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `app/health/` package (data objects + pipeline skeleton + check manager) and wire it into the tester so every 2.5s window produces a Health Report, surfaced as an `UNKNOWN` health indicator — with zero change to the existing CNN behavior.

**Architecture:** A self-contained, NumPy-only `app/health/` package implements the conceptual data model and a 7-stage `HealthAnalysisPipeline` (stages wired, most are no-ops in Phase 0). `main.py`'s `handle_audio_chunk` builds an immutable `AudioWindow` from `session_buffer` and runs the pipeline alongside `run_inference`, storing the latest report and updating a new indicator label. The health package never imports tester/UI code.

**Tech Stack:** Python 3.11, NumPy 2.x, Tkinter (existing UI), pytest (added as dev dependency). Run via the project `.venv`.

> **Commit policy for this repo:** The repo owner commits and pushes **manually** after reviewing/testing each phase. **Do NOT run `git commit` or `git push` in any task.** Each task ends at "run the tests and confirm they pass." A final review/commit is the owner's.

> **Reference spec:** `docs/superpowers/specs/2026-06-23-audio-signal-health-monitoring-phased-implementation-design.md` (§5 Phase 0, §6 module layout).

---

## File Structure

**Create:**
- `app/health/__init__.py` — package exports
- `app/health/models.py` — `HealthState`, `CheckStatus`, `Measurement`, `AudioWindow`, `SignalCheckResult`, `HealthReport`
- `app/health/checks/__init__.py` — checks subpackage
- `app/health/checks/base.py` — `SignalHealthCheck` abstract interface
- `app/health/manager.py` — `SignalCheckManager` (registry + isolated execution)
- `app/health/fusion.py` — `decide()` (minimal Phase 0 decision)
- `app/health/pipeline.py` — `HealthAnalysisPipeline` (7 stages)
- `conftest.py` (repo root, empty) — ensures repo root is on `sys.path` for tests
- `requirements-dev.txt` — pytest
- `tests/__init__.py`, `tests/health/__init__.py`
- `tests/health/test_models.py`, `tests/health/test_manager.py`, `tests/health/test_pipeline.py`

**Modify:**
- `main.py` — imports (after line 19), `__init__` (after line 37), UI indicator (after line 262), `handle_audio_chunk` (after line 697), plus a new `_update_health_indicator` helper.

---

## Task 1: Dev environment & test scaffolding

**Files:**
- Create: `requirements-dev.txt`, `conftest.py`, `tests/__init__.py`, `tests/health/__init__.py`

- [ ] **Step 1: Create the dev requirements file**

Create `requirements-dev.txt`:

```text
# Development-only dependencies (not needed to run the app)
pytest==8.3.4
```

- [ ] **Step 2: Create the venv if missing and install deps**

The app expects a `.venv` (see `run.command`). Create it and install runtime + dev deps:

```bash
test -d .venv || (command -v uv >/dev/null 2>&1 && uv venv .venv || python3 -m venv .venv)
command -v uv >/dev/null 2>&1 && uv pip install --python .venv/bin/python -r requirements.txt -r requirements-dev.txt || .venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
```

Expected: completes without error; `.venv/bin/python` exists.

- [ ] **Step 3: Create empty root conftest and test package markers**

Create `conftest.py` (repo root):

```python
# Present so pytest adds the repo root to sys.path, making `app` importable in tests.
```

Create `tests/__init__.py` (empty) and `tests/health/__init__.py` (empty).

- [ ] **Step 4: Verify pytest runs (no tests yet)**

Run: `.venv/bin/python -m pytest -q`
Expected: exits cleanly reporting "no tests ran" (exit code 5) — confirms pytest is installed and discovers the `tests/` dir.

---

## Task 2: Conceptual data model

**Files:**
- Create: `app/health/models.py`, `app/health/__init__.py`
- Test: `tests/health/test_models.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/health/test_models.py`:

```python
import numpy as np
import pytest

from app.health.models import (
    AudioWindow,
    CheckStatus,
    HealthReport,
    HealthState,
    Measurement,
    SignalCheckResult,
)


def test_audio_window_derived_properties():
    samples = np.zeros(110250, dtype=np.float32)
    window = AudioWindow(samples=samples, sample_rate=44100)
    assert window.sample_count == 110250
    assert window.channel_count == 1
    assert window.window_duration == pytest.approx(2.5)


def test_audio_window_is_immutable():
    window = AudioWindow(samples=np.ones(10, dtype=np.float32), sample_rate=44100)
    with pytest.raises(ValueError):
        window.samples[0] = 5.0


def test_audio_window_copies_source_samples():
    src = np.ones(10, dtype=np.float32)
    window = AudioWindow(samples=src, sample_rate=44100)
    src[0] = 99.0  # mutating the source must not affect the window
    assert window.samples[0] == 1.0


def test_measurement_is_frozen():
    m = Measurement(name="rms", value=0.1, unit="")
    with pytest.raises(Exception):
        m.value = 0.2


def test_health_report_defaults():
    report = HealthReport(timestamp=0.0, window_id="abc")
    assert report.final_state is HealthState.UNKNOWN
    assert report.check_results == []
    assert report.confidence == 0.0


def test_signal_check_result_defaults():
    result = SignalCheckResult(check_id="T000", check_name="dummy")
    assert result.status is CheckStatus.NOT_EXECUTED
    assert result.executed is False
    assert result.measurements == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/health/test_models.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'app.health'`.

- [ ] **Step 3: Implement the data model**

Create `app/health/models.py`:

```python
"""Conceptual data objects for the Audio Signal Health Monitoring subsystem.

Pure-data, NumPy-only. This module imports nothing from the tester or UI so the
logic stays portable. See the Phase 0 design spec for the conceptual model (Ch. 5).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class HealthState(Enum):
    OK = "OK"
    WARNING = "WARNING"
    FAULT = "FAULT"
    UNKNOWN = "UNKNOWN"


class CheckStatus(Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    NOT_EXECUTED = "NOT_EXECUTED"


@dataclass(frozen=True)
class Measurement:
    """A single numerical value produced by a Signal Health Check."""

    name: str
    value: float
    unit: str = ""


@dataclass(frozen=True)
class AudioWindow:
    """Immutable 2.5s mono analysis window. The fundamental pipeline input."""

    samples: np.ndarray
    sample_rate: int
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        # Store a private, read-only 1-D float32 copy so the window is immutable
        # and decoupled from the caller's buffer.
        arr = np.array(self.samples, dtype=np.float32).reshape(-1)
        arr.setflags(write=False)
        object.__setattr__(self, "samples", arr)

    @property
    def sample_count(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channel_count(self) -> int:
        return 1

    @property
    def window_duration(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.sample_count / float(self.sample_rate)


@dataclass
class SignalCheckResult:
    """Outcome of one Signal Health Check for one Audio Window."""

    check_id: str
    check_name: str
    status: CheckStatus = CheckStatus.NOT_EXECUTED
    executed: bool = False
    execution_time: float = 0.0
    measurements: list[Measurement] = field(default_factory=list)
    diagnostic_messages: list[str] = field(default_factory=list)


@dataclass
class HealthReport:
    """Final output of the Health Analysis Pipeline for one Audio Window."""

    timestamp: float
    window_id: str
    check_results: list[SignalCheckResult] = field(default_factory=list)
    calibration_evaluation: Optional[Any] = None
    anomaly_result: Optional[Any] = None
    final_state: HealthState = HealthState.UNKNOWN
    confidence: float = 0.0
    diagnostic_summary: str = ""
```

Create `app/health/__init__.py`:

```python
"""Audio Signal Health Monitoring subsystem (NumPy-only, UI-agnostic)."""
from app.health.models import (
    AudioWindow,
    CheckStatus,
    HealthReport,
    HealthState,
    Measurement,
    SignalCheckResult,
)

__all__ = [
    "AudioWindow",
    "CheckStatus",
    "HealthReport",
    "HealthState",
    "Measurement",
    "SignalCheckResult",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/health/test_models.py -v`
Expected: PASS (6 passed).

---

## Task 3: Check interface & Signal Check Manager

**Files:**
- Create: `app/health/checks/__init__.py`, `app/health/checks/base.py`, `app/health/manager.py`
- Test: `tests/health/test_manager.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/health/test_manager.py`:

```python
import numpy as np

from app.health.checks.base import SignalHealthCheck
from app.health.manager import SignalCheckManager
from app.health.models import AudioWindow, CheckStatus, SignalCheckResult


def _window():
    return AudioWindow(samples=np.zeros(100, dtype=np.float32), sample_rate=44100)


class _PassingCheck(SignalHealthCheck):
    check_id = "T999"
    check_name = "Passing"

    def run(self, window, features):
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=CheckStatus.PASS,
        )


class _ExplodingCheck(SignalHealthCheck):
    check_id = "T998"
    check_name = "Exploding"

    def run(self, window, features):
        raise RuntimeError("boom")


def test_register_and_run_check():
    manager = SignalCheckManager()
    manager.register(_PassingCheck())
    results = manager.run_checks(_window(), {})
    assert len(results) == 1
    assert results[0].check_id == "T999"
    assert results[0].executed is True
    assert results[0].status is CheckStatus.PASS
    assert results[0].execution_time >= 0.0


def test_failing_check_is_isolated():
    manager = SignalCheckManager()
    manager.register(_ExplodingCheck())
    manager.register(_PassingCheck())
    results = manager.run_checks(_window(), {})
    assert len(results) == 2  # one failure does not stop the others
    failed = results[0]
    assert failed.check_id == "T998"
    assert failed.executed is False
    assert failed.status is CheckStatus.NOT_EXECUTED
    assert any("boom" in msg for msg in failed.diagnostic_messages)
    assert results[1].status is CheckStatus.PASS


def test_checks_property_returns_copy():
    manager = SignalCheckManager()
    manager.register(_PassingCheck())
    checks = manager.checks
    checks.clear()
    assert len(manager.checks) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/health/test_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.health.checks'`.

- [ ] **Step 3: Implement the check interface and manager**

Create `app/health/checks/__init__.py` (empty):

```python
"""Signal Health Checks (time-domain, frequency-domain, stability)."""
```

Create `app/health/checks/base.py`:

```python
"""Base interface every Signal Health Check implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.health.models import AudioWindow, SignalCheckResult


class SignalHealthCheck(ABC):
    """Evaluates exactly one property of an Audio Window.

    Subclasses set ``check_id`` / ``check_name`` and implement ``run``. Checks must
    not depend on one another; the manager runs each in isolation.
    """

    check_id: str = ""
    check_name: str = ""

    @abstractmethod
    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        """Measure the property and return a SignalCheckResult."""
        raise NotImplementedError
```

Create `app/health/manager.py`:

```python
"""Signal Check Manager: registry and isolated execution of Signal Health Checks."""
from __future__ import annotations

import time
from typing import Any

from app.health.checks.base import SignalHealthCheck
from app.health.models import AudioWindow, CheckStatus, SignalCheckResult


class SignalCheckManager:
    """Holds the registered checks and runs them, isolating individual failures."""

    def __init__(self):
        self._checks: list[SignalHealthCheck] = []

    def register(self, check: SignalHealthCheck) -> None:
        self._checks.append(check)

    @property
    def checks(self) -> list[SignalHealthCheck]:
        return list(self._checks)

    def run_checks(
        self, window: AudioWindow, features: dict[str, Any]
    ) -> list[SignalCheckResult]:
        results: list[SignalCheckResult] = []
        for check in self._checks:
            start = time.perf_counter()
            try:
                result = check.run(window, features)
                result.executed = True
            except Exception as exc:  # isolate one check's failure from the rest
                result = SignalCheckResult(
                    check_id=getattr(check, "check_id", "") or check.__class__.__name__,
                    check_name=getattr(check, "check_name", "")
                    or check.__class__.__name__,
                    status=CheckStatus.NOT_EXECUTED,
                    executed=False,
                    diagnostic_messages=[f"Check raised: {exc}"],
                )
            result.execution_time = time.perf_counter() - start
            results.append(result)
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/health/test_manager.py -v`
Expected: PASS (3 passed).

---

## Task 4: Decision fusion (minimal) & Health Analysis Pipeline

**Files:**
- Create: `app/health/fusion.py`, `app/health/pipeline.py`
- Test: `tests/health/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/health/test_pipeline.py`:

```python
import numpy as np

from app.health.checks.base import SignalHealthCheck
from app.health.manager import SignalCheckManager
from app.health.models import AudioWindow, CheckStatus, HealthState, SignalCheckResult
from app.health.pipeline import HealthAnalysisPipeline


def _window():
    return AudioWindow(samples=np.zeros(110250, dtype=np.float32), sample_rate=44100)


class _PassingCheck(SignalHealthCheck):
    check_id = "T999"
    check_name = "Passing"

    def run(self, window, features):
        return SignalCheckResult(
            check_id=self.check_id, check_name=self.check_name, status=CheckStatus.PASS
        )


def test_pipeline_with_no_checks_is_unknown():
    report = HealthAnalysisPipeline().analyze(_window())
    assert report.final_state is HealthState.UNKNOWN
    assert report.check_results == []
    assert report.confidence == 0.0
    assert report.window_id  # non-empty id
    assert report.diagnostic_summary


def test_pipeline_runs_registered_checks():
    manager = SignalCheckManager()
    manager.register(_PassingCheck())
    report = HealthAnalysisPipeline(manager=manager).analyze(_window())
    assert len(report.check_results) == 1
    assert report.check_results[0].check_id == "T999"


def test_pipeline_produces_unique_window_ids():
    pipeline = HealthAnalysisPipeline()
    id1 = pipeline.analyze(_window()).window_id
    id2 = pipeline.analyze(_window()).window_id
    assert id1 != id2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/health/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.health.pipeline'`.

- [ ] **Step 3: Implement fusion and the pipeline**

Create `app/health/fusion.py`:

```python
"""Decision Fusion — combines Signal Check Results into a final health state.

Phase 0 is minimal: with no registered checks (and therefore no results), the
state is UNKNOWN. Rule-based fusion over real results arrives in Phase 1.
"""
from __future__ import annotations

from app.health.models import HealthState, SignalCheckResult


def decide(results: list[SignalCheckResult]) -> tuple[HealthState, float, str]:
    """Return (final_state, confidence, diagnostic_summary)."""
    if not results:
        return HealthState.UNKNOWN, 0.0, "No signal health checks executed."
    # Phase 0: checks exist but no decision rules yet; report UNKNOWN until Phase 1.
    return HealthState.UNKNOWN, 0.0, "Signal health checks ran; fusion pending (Phase 1)."
```

Create `app/health/pipeline.py`:

```python
"""Health Analysis Pipeline — evaluates one Audio Window into one Health Report.

Seven stages (spec Ch. 3). In Phase 0 the stages are wired but preprocessing,
feature preparation, calibration, and anomaly detection are no-ops.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from app.health.fusion import decide
from app.health.manager import SignalCheckManager
from app.health.models import AudioWindow, HealthReport


class HealthAnalysisPipeline:
    def __init__(self, manager: Optional[SignalCheckManager] = None):
        self.manager = manager if manager is not None else SignalCheckManager()

    def analyze(self, window: AudioWindow) -> HealthReport:
        # Stage 1 — Preprocessing (Phase 0: no-op)
        window = self._preprocess(window)
        # Stage 2 — Feature Preparation (Phase 0: no-op)
        features = self._prepare_features(window)
        # Stage 3 — Signal Health Checks
        results = self.manager.run_checks(window, features)
        # Stage 4 — Calibration Evaluation (Phase 0: no-op)
        calibration_evaluation = None
        # Stage 5 — Anomaly Detection (Phase 0: no-op)
        anomaly_result = None
        # Stage 6 — Decision Fusion
        final_state, confidence, summary = decide(results)
        # Stage 7 — Health Report Generation
        return HealthReport(
            timestamp=time.time(),
            window_id=uuid.uuid4().hex,
            check_results=results,
            calibration_evaluation=calibration_evaluation,
            anomaly_result=anomaly_result,
            final_state=final_state,
            confidence=confidence,
            diagnostic_summary=summary,
        )

    def _preprocess(self, window: AudioWindow) -> AudioWindow:
        return window

    def _prepare_features(self, window: AudioWindow) -> dict[str, Any]:
        return {}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/health/test_pipeline.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the whole health test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: PASS (12 passed total).

---

## Task 5: Wire the pipeline into the tester + health indicator

**Files:**
- Modify: `main.py` (imports after line 19; `__init__` after line 37; UI label after line 262; `_update_health_indicator` helper; `handle_audio_chunk` after line 697)

- [ ] **Step 1: Add the imports**

In `main.py`, immediately after line 19 (`from inference_utils import AudioProcessor`), add:

```python
from app.health.pipeline import HealthAnalysisPipeline
from app.health.models import AudioWindow, HealthState
```

- [ ] **Step 2: Construct the pipeline in `__init__`**

In `main.py`, immediately after line 37 (`self.processor = AudioProcessor()`), add:

```python
        self.health_pipeline = HealthAnalysisPipeline()
        self.latest_health_report = None
```

- [ ] **Step 3: Add the health indicator label**

In `main.py`, immediately after line 262 (`self.diag_label.pack(pady=10)`), add:

```python

        # Signal Health Indicator (Audio Signal Health Monitoring subsystem)
        self.health_label = ttk.Label(
            dash_frame,
            text="Signal Health: UNKNOWN",
            font=("Helvetica", 12, "bold"),
            foreground="gray",
        )
        self.health_label.pack(pady=4)
```

- [ ] **Step 4: Add the `_update_health_indicator` helper**

In `main.py`, add this method to `ModelsTesterApp` directly above `def handle_audio_chunk(self, chunk):` (currently line 683):

```python
    def _update_health_indicator(self, report):
        colors = {
            HealthState.OK: "green",
            HealthState.WARNING: "orange",
            HealthState.FAULT: "red",
            HealthState.UNKNOWN: "gray",
        }
        state = report.final_state
        self.health_label.configure(
            text=f"Signal Health: {state.value}",
            foreground=colors.get(state, "gray"),
        )
```

- [ ] **Step 5: Run the pipeline in `handle_audio_chunk`**

In `main.py`, immediately after line 697 (`self.raw_audio_snapshot = self.session_buffer.copy()`), add:

```python

        # Audio signal health monitoring — additive; never blocks or alters inference.
        try:
            window = AudioWindow(samples=self.session_buffer, sample_rate=44100)
            self.latest_health_report = self.health_pipeline.analyze(window)
            self._update_health_indicator(self.latest_health_report)
        except Exception as e:
            self.log(f"Health monitoring error: {e}")
```

- [ ] **Step 6: Verify `main.py` imports cleanly (no Tk window)**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('main.py parses OK')"`
Expected: `main.py parses OK`.

Then verify the health wiring works headlessly (no Tkinter needed):

Run:
```bash
.venv/bin/python -c "
import numpy as np
from app.health.pipeline import HealthAnalysisPipeline
from app.health.models import AudioWindow, HealthState
buf = np.zeros(int(44100*2.5), dtype=np.float32)
r = HealthAnalysisPipeline().analyze(AudioWindow(samples=buf, sample_rate=44100))
assert r.final_state is HealthState.UNKNOWN
print('seam OK:', r.final_state.value, '| summary:', r.diagnostic_summary)
"
```
Expected: `seam OK: UNKNOWN | summary: No signal health checks executed.`

- [ ] **Step 7: Manual UI verification (owner)**

Launch the app and confirm the new indicator appears and updates:

```bash
.venv/bin/python launcher.py
```
Expected: The dashboard shows **"Signal Health: UNKNOWN"** (gray) below the diagnosis label. Start a test (mic or a WAV file); the app classifies exactly as before and the indicator remains "UNKNOWN" (no checks yet). No "Health monitoring error" lines appear in the log.

---

## Phase 0 Done

When all tasks pass, Phase 0 is complete: the tester runs unchanged plus a live (UNKNOWN) signal-health indicator backed by a fully wired pipeline. **Hand back to the owner for review, manual test, and commit.** Phase 1 (time-domain checks + real decision fusion) will replace the minimal `decide()` and register the first checks.

---

## Self-Review

- **Spec coverage (Phase 0 scope §5):** data objects `AudioWindow`/`Measurement`/`SignalCheckResult`/`HealthReport` + `HealthState` (Task 2 ✓); 7-stage pipeline skeleton + `SignalCheckManager` registry (Tasks 3–4 ✓); integration seam in `handle_audio_chunk` running on the same window alongside `run_inference` (Task 5 ✓); UNKNOWN indicator (Task 5 ✓); no change to inference behavior — health wrapped in try/except, never blocks (Task 5 ✓). `SignalHealthCheck` base added early so Phase 1 only registers checks (Task 3 ✓).
- **Portability constraint (§3/§7):** `app/health/` imports only stdlib + NumPy; no tester/UI imports — verified by module contents in Tasks 2–4.
- **Placeholder scan:** no "TBD/TODO/handle edge cases" steps; every code step shows full code; test code is concrete. The minimal `decide()` is an intentional Phase 0 stub, fully specified.
- **Type consistency:** `HealthAnalysisPipeline(manager=...)`, `manager.run_checks(window, features)`, `decide(results) -> (HealthState, float, str)`, `SignalCheckResult(check_id, check_name, status, executed, execution_time, measurements, diagnostic_messages)`, `report.final_state` — names match across Tasks 2–5 and the `main.py` wiring.
