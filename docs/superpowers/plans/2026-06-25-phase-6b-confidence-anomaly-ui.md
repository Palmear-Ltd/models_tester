# Phase 6b — Surface Confidence & Anomaly (UI) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the report confidence on the Signal Health indicator and log an `[anomaly]` line when a window first becomes anomalous.

**Architecture:** Two small additions to `_update_health_indicator` in `main.py`: append `· conf X.XX` (from `report.confidence`) to the indicator text, and log the anomaly on its rising edge (tracked by a `_last_anomalous` flag, reset per test). Engine untouched (6a already populates `report.confidence`/`report.anomaly_result`).

**Tech Stack:** Tkinter; pytest (regression only).

> **Commit policy:** Owner commits manually. **Do NOT run git commit/add/push.** End at verification.

> **Testing reality:** Tkinter wiring — verified by `ast.parse` + `import main` + the unchanged 137-test suite + a manual GUI check.

## Current State (anchors)

- `main.py` `__init__` (≈lines 55-57): `self.runtime_monitor = RuntimeMonitor()` / `self._validating = False` / `self._validation_reports = []`.
- `main.py` `start_test` reset block (≈line 461): a bare `self.health_state_history = []` immediately followed by `self.runtime_monitor = RuntimeMonitor()` (distinct from the commented `__init__` initialiser `self.health_state_history = []  # (time, level: ...)`).
- `main.py` `_update_health_indicator`:
  ```python
        self.health_label.configure(
            text=f"Signal Health: {state.value}",
            foreground=colors.get(state, "gray"),
        )
        for event in events:
            self.log(f"[monitor] {event.message}: {report.diagnostic_summary}")
        self._update_health_panel(report)
  ```
- `report.confidence` (float) and `report.anomaly_result` (`None`, or has `.is_anomalous`, `.distance`, `.contributors` list of `(label, z)`) are populated by Phase 6a.
- 137 tests pass.

## File Structure
- Modify: `main.py` only.

---

## Task 1: Confidence on the indicator + anomaly log

**Files:** Modify `main.py`

- [ ] **Step 1: Initialise the anomaly edge flag in `__init__`**

In `main.py` `__init__`, find:

```python
        self._validating = False
        self._validation_reports = []
```

Replace with:

```python
        self._validating = False
        self._validation_reports = []
        self._last_anomalous = False
```

- [ ] **Step 2: Reset the flag in `start_test`**

In `main.py` `start_test`, find (the bare reset, followed by the monitor reset):

```python
        self.health_state_history = []
        self.runtime_monitor = RuntimeMonitor()
```

Replace with:

```python
        self.health_state_history = []
        self.runtime_monitor = RuntimeMonitor()
        self._last_anomalous = False
```

- [ ] **Step 3: Show confidence + log anomaly transitions in `_update_health_indicator`**

In `main.py` `_update_health_indicator`, find:

```python
        self.health_label.configure(
            text=f"Signal Health: {state.value}",
            foreground=colors.get(state, "gray"),
        )
        for event in events:
            self.log(f"[monitor] {event.message}: {report.diagnostic_summary}")
        self._update_health_panel(report)
```

Replace with:

```python
        self.health_label.configure(
            text=f"Signal Health: {state.value} · conf {report.confidence:.2f}",
            foreground=colors.get(state, "gray"),
        )
        for event in events:
            self.log(f"[monitor] {event.message}: {report.diagnostic_summary}")
        # Log an anomaly only on its rising edge (anomaly_result is None without a profile).
        anomaly = report.anomaly_result
        anomalous = anomaly is not None and anomaly.is_anomalous
        if anomalous and not self._last_anomalous:
            top_label, top_z = anomaly.contributors[0] if anomaly.contributors else ("", 0.0)
            self.log(f"[anomaly] distance {anomaly.distance:.1f} — {top_label} z={top_z:.1f}")
        self._last_anomalous = anomalous
        self._update_health_panel(report)
```

- [ ] **Step 4: Verify**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('parses OK')"` → `parses OK`
Run: `.venv/bin/python -c "import main; print('import OK')"` → `import OK` (keras FutureWarning is fine)
Run: `.venv/bin/python -m pytest tests/ -q` → 137 passed (UI-only change).
Run: `git diff main.py | grep '^-' | grep -v '^---'` → only the `health_label.configure` text line and the panel-update anchor lines show as replaced; confirm no inference/health logic removed.

- [ ] **Step 5: Manual GUI verification (owner)**

Launch `./.venv/bin/python launcher.py`. Start a test **with no calibration profile** → the indicator reads e.g. `Signal Health: OK · conf 1.00` and no `[anomaly]` lines appear. **Load a calibration profile** and run: the indicator's `conf` now tracks the anomaly distance (drops as the signal drifts from the profile), and the first window that crosses the anomaly threshold logs one `[anomaly] distance … — <check.measurement> z=…` line (not repeated every 0.5 s). Classification and the other panels are unchanged.

---

## Phase 6b Done — Phase 6 complete

The indicator now surfaces confidence and the log flags anomalies, completing Phase 6 (anomaly detection + confidence) and the Audio Signal Health Monitoring subsystem. Hand back to the owner for review, manual test, and commit.

---

## Self-Review

- **Spec coverage (§4):** indicator shows confidence (`· conf {report.confidence:.2f}`) — Step 3; `[anomaly]` logged once on transition via the `_last_anomalous` rising-edge flag (initialised in `__init__`, reset in `start_test`) — Steps 1/2/3; no new plots, engine untouched. Without a profile `report.anomaly_result is None` → `anomalous` is `False` → no log, confidence stays checks-based.
- **Placeholder scan:** no TBD/TODO; full code each step; concrete commands and expected output.
- **Type consistency:** `report.confidence` (float) and `report.anomaly_result` (`None` | `.is_anomalous`/`.distance`/`.contributors`) come from Phase 6a; `self._last_anomalous` created in `__init__`, reset in `start_test`, read+written in `_update_health_indicator`. `contributors[0]` is `(label, z)`; the empty-list guard prevents an IndexError.
