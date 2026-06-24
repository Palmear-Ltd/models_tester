# Phase 2b — Dedicated Health Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated "Signal Health Detail" panel to the tester showing every check's id/name, status (color-coded), and key measurements/diagnostics, refreshed each analysis window.

**Architecture:** A UI-agnostic `app/health/reporting.py` turns a `HealthReport` into plain display rows `(check_id, check_name, status, detail)` — pure functions, no Tkinter, fully testable. `main.py` adds a `ttk.Treeview` panel in the left frame and repopulates it from `report_rows(report)` inside the existing `_update_health_indicator`, so it updates on the same 0.5 s cadence as the indicator. Purely additive — no change to inference or to the health logic.

**Tech Stack:** Python 3.13, NumPy 2.x, Tkinter (ttk.Treeview), pytest. Run via `.venv/bin/python -m pytest`.

> **Commit policy:** Owner commits/pushes **manually** after review. **Do NOT run `git commit`/`git add`/`git push`.** Each task ends at "tests pass."

> **References:** spec `arch_update.md` §11 (Reporting). Design spec §5 Phase 2 ("dedicated health panel — per-check table with measurements, status, and overall state"). Builds on Phase 2a (11 checks via `default_pipeline`).

---

## Current State (end of Phase 2a)

- `default_pipeline()` runs 11 checks (7 time-domain + 4 frequency-domain); each `HealthReport.check_results` is a list of `SignalCheckResult` (`check_id`, `check_name`, `status: CheckStatus`, `category`, `measurements: list[Measurement]`, `diagnostic_messages: list[str]`). `Measurement(name, value, unit="")`.
- `main.py` `ModelsTesterApp`: a horizontal `main_pane` with `left_frame` (config / dashboard / controls / log) and `right_frame` (scrollable plots). `dash_frame` holds `self.health_label` (the "Signal Health: STATE" indicator) and ends with the energy bar (`main.py:280-285`); the control area follows (`main.py:287`).
- `_update_health_indicator(self, report)` (`main.py:700-715`) updates `self.health_label` and logs on state transitions; it is called from `handle_audio_chunk` every 0.5 s with `self.latest_health_report`.
- 64 tests pass.

## File Structure

**Create:**
- `app/health/reporting.py` — `check_row(result)`, `report_rows(report)`, `_format_value`, `_format_measurements`. Pure, UI-agnostic (imports only `app.health.models`).
- `tests/health/test_reporting.py`.

**Modify:**
- `main.py` — add the "Signal Health Detail" Treeview panel (setup in `_setup_ui`), a `_update_health_panel(report)` method, and a call to it from `_update_health_indicator`. Plus one import.

---

## Task 1: UI-agnostic report formatting

**Files:**
- Create: `app/health/reporting.py`
- Test: `tests/health/test_reporting.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/health/test_reporting.py`:

```python
from app.health.models import (
    CheckStatus,
    HealthReport,
    Measurement,
    SignalCheckResult,
)
from app.health.reporting import check_row, report_rows


def _result(cid, name, status, measurements=None, diag=None):
    return SignalCheckResult(
        check_id=cid,
        check_name=name,
        status=status,
        measurements=measurements or [],
        diagnostic_messages=diag or [],
    )


def test_check_row_shows_measurements_when_passing():
    r = _result(
        "T002", "Signal Energy", CheckStatus.PASS,
        measurements=[Measurement("rms", 0.21034)],
    )
    cid, name, status, detail = check_row(r)
    assert cid == "T002"
    assert name == "Signal Energy"
    assert status == "PASS"
    assert "rms=0.2103" in detail


def test_check_row_shows_diagnostics_when_not_passing():
    r = _result(
        "T001", "Flatline Detection", CheckStatus.FAIL,
        measurements=[Measurement("std", 0.0)],
        diag=["Flatline: dead signal"],
    )
    _, _, status, detail = check_row(r)
    assert status == "FAIL"
    assert detail == "Flatline: dead signal"


def test_measurement_unit_is_included():
    r = _result(
        "F001", "Spectral Shape", CheckStatus.PASS,
        measurements=[Measurement("spectral_centroid", 1002.0, "Hz")],
    )
    _, _, _, detail = check_row(r)
    assert "spectral_centroid=1002 Hz" in detail


def test_report_rows_one_per_check():
    report = HealthReport(
        timestamp=0.0,
        window_id="w",
        check_results=[
            _result("T001", "Flatline Detection", CheckStatus.PASS),
            _result("T002", "Signal Energy", CheckStatus.PASS),
        ],
    )
    rows = report_rows(report)
    assert len(rows) == 2
    assert rows[0][0] == "T001"
    assert rows[1][0] == "T002"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_reporting.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.health.reporting'`.

- [ ] **Step 3: Implement `reporting.py`**

Create `app/health/reporting.py`:

```python
"""UI-agnostic formatting of a HealthReport into display rows (spec §11).

Pure functions returning plain Python types — no Tkinter/UI imports — so the
health package stays portable. The tester's panel renders these rows; any other
front end can reuse them.
"""
from __future__ import annotations

from app.health.models import HealthReport, Measurement, SignalCheckResult


def _format_value(value: float) -> str:
    return f"{value:.4g}"


def _format_measurements(measurements: list[Measurement]) -> str:
    parts = []
    for m in measurements:
        unit = f" {m.unit}" if m.unit else ""
        parts.append(f"{m.name}={_format_value(m.value)}{unit}")
    return ", ".join(parts)


def check_row(result: SignalCheckResult) -> tuple[str, str, str, str]:
    """Return (check_id, check_name, status, detail) for one check result.

    `detail` is the diagnostic message(s) when the check is not PASS, otherwise a
    compact list of its measurements.
    """
    if result.diagnostic_messages:
        detail = "; ".join(result.diagnostic_messages)
    else:
        detail = _format_measurements(result.measurements)
    return (result.check_id, result.check_name, result.status.value, detail)


def report_rows(report: HealthReport) -> list[tuple[str, str, str, str]]:
    """One display row per check result, in pipeline execution order."""
    return [check_row(r) for r in report.check_results]
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_reporting.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Confirm the full suite still passes**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (68 total: 64 prior + 4 new).

---

## Task 2: Render the panel in the tester

**Files:**
- Modify: `main.py`

> Locate anchors by their quoted TEXT, not line numbers. Verify each anchor before editing.

- [ ] **Step 1: Add the import**

In `main.py`, find:

```python
from app.health.models import AudioWindow, HealthState
```

Add immediately AFTER it:

```python
from app.health.reporting import report_rows
```

- [ ] **Step 2: Build the panel in `_setup_ui`**

In `main.py`, find the end of the dashboard's energy indicator and the start of the control area:

```python
        self.energy_bar = ttk.Progressbar(energy_frame, orient="horizontal", length=200, mode="determinate", maximum=1.0)
        self.energy_bar.pack(side="left", padx=10, fill="x", expand=True)
        
        # --- Control Area (Left) ---
```

Insert the panel BETWEEN the energy bar and the control-area comment, so the block becomes:

```python
        self.energy_bar = ttk.Progressbar(energy_frame, orient="horizontal", length=200, mode="determinate", maximum=1.0)
        self.energy_bar.pack(side="left", padx=10, fill="x", expand=True)

        # --- Signal Health Detail (per-check breakdown) ---
        health_frame = ttk.LabelFrame(left_frame, text="Signal Health Detail", padding=8, style="Card.TLabelframe")
        health_frame.pack(fill="x", padx=10, pady=5)
        self.health_tree = ttk.Treeview(
            health_frame, columns=("status", "detail"), show="tree headings", height=11
        )
        self.health_tree.heading("#0", text="Check")
        self.health_tree.heading("status", text="Status")
        self.health_tree.heading("detail", text="Detail")
        self.health_tree.column("#0", width=180, stretch=False)
        self.health_tree.column("status", width=80, anchor="center", stretch=False)
        self.health_tree.column("detail", width=240)
        self.health_tree.tag_configure("PASS", foreground="green")
        self.health_tree.tag_configure("WARNING", foreground="orange")
        self.health_tree.tag_configure("FAIL", foreground="red")
        self.health_tree.tag_configure("NOT_EXECUTED", foreground="gray")
        self.health_tree.pack(fill="x")
        
        # --- Control Area (Left) ---
```

- [ ] **Step 3: Add the `_update_health_panel` method**

In `main.py`, find the `_update_health_indicator` method and add this method immediately BEFORE it (matching 4-space method indentation):

```python
    def _update_health_panel(self, report):
        tree = self.health_tree
        tree.delete(*tree.get_children())
        for check_id, name, status, detail in report_rows(report):
            tree.insert(
                "", "end", text=f"{check_id}  {name}", values=(status, detail), tags=(status,)
            )

```

- [ ] **Step 4: Call it from `_update_health_indicator`**

In `main.py`, find the end of `_update_health_indicator`:

```python
        # Log only on transitions to avoid flooding the log every 0.5s.
        if state != self._last_health_state:
            self._last_health_state = state
            self.log(f"Signal health {state.value}: {report.diagnostic_summary}")
```

Add immediately AFTER that block (still inside the method, 8-space indentation):

```python
        self._update_health_panel(report)
```

- [ ] **Step 5: Verify parse + additive diff**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('main.py parses OK')"`
Expected: `main.py parses OK`

Run: `git diff main.py | grep '^-' | grep -v '^---'`
Expected: NO output (purely additive — only insertions; no existing lines removed/modified).

- [ ] **Step 6: Headless check of the data path**

The Treeview itself needs a display, but the data feeding it is testable headlessly:

```bash
.venv/bin/python -c "
import numpy as np
from app.health.defaults import default_pipeline
from app.health.models import AudioWindow
from app.health.reporting import report_rows
SR, N = 44100, 110250
t = np.arange(N)/SR
sig = (0.3*np.sin(2*np.pi*1000*t)).astype(np.float32)
report = default_pipeline().analyze(AudioWindow(samples=sig, sample_rate=SR))
rows = report_rows(report)
print('rows:', len(rows))
for r in rows:
    print(r)
assert len(rows) == 11
print('panel data OK')
"
```
Expected: 11 rows printed (T001–T007, F001–F004) with status PASS and measurement details, then `panel data OK`.

- [ ] **Step 7: Confirm the suite still passes**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (68 total).

- [ ] **Step 8: Manual GUI verification (owner)**

(Requires full deps: `uv pip install --python .venv/bin/python -r requirements.txt`.) Launch `./.venv/bin/python launcher.py`, start a test, and confirm the new **"Signal Health Detail"** panel under the dashboard lists all 11 checks with color-coded status (green PASS / orange WARNING / red FAIL) and a Detail column showing measurements (or the diagnostic message for non-PASS rows). The panel should refresh live as audio plays; classification behavior unchanged.

---

## Phase 2b Done

The tester now shows a live per-check health panel (id/name, color-coded status, measurements/diagnostics) alongside the existing OK/WARNING/FAULT indicator. Hand back to the owner for review, manual test, and commit. Phase 2c adds the configuration-profile system (enable/disable checks, mandatory checks, dev/prod/diagnostic/minimal profiles).

---

## Self-Review

- **Spec coverage:** dedicated panel with per-check status + measurements (design §5 Phase 2) — Task 2 Treeview fed by `report_rows`. Overall state remains shown by the adjacent `health_label`. Reporting interface (spec §11) — `reporting.py` produces standardized display rows, UI-agnostic.
- **Placeholder scan:** no TBD/TODO; every step has full code and exact commands.
- **Type consistency:** `report_rows(report) -> list[(check_id, check_name, status, detail)]`; the Treeview insert uses `text=` for the `#0` (Check) column and `values=(status, detail)` for the two declared columns; tag names (`PASS`/`WARNING`/`FAIL`/`NOT_EXECUTED`) equal `CheckStatus.value`, so `tags=(status,)` colors each row. `reporting.py` imports only `app.health.models` (portable).
- **No regression:** panel is additive (Task 2 diff is insert-only); it reads `report.check_results` and never mutates the report or the inference path; refresh piggybacks on the existing per-window `_update_health_indicator` call.
