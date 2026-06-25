# Phase 4b — Runtime Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `RuntimeMonitor` that debounces the per-window health verdict (fault persistence + recovery) into a stable "runtime state" and emits transition events, and wire it so the tester's indicator shows the smoothed state and logs monitor events.

**Architecture:** A stateful, pure-stdlib `app/health/monitoring.py` consumes `HealthReport`s via `update(report) -> list[MonitoringEvent]`. It adopts a new state only after the raw per-window state has persisted for that state's threshold (recovery to OK / warn-persistence / fault-persistence), so transient blips don't flap. `main.py` holds a monitor (reset per test session), shows `runtime_monitor.runtime_state` on the indicator, and logs events; the panel still shows raw per-window/per-check detail.

**Tech Stack:** Python 3.13, stdlib (dataclasses); Tkinter wiring; pytest. `app/health/` stays NumPy/stdlib-only.

> **Commit policy:** Owner commits manually. **Do NOT run git commit/add/push.** End at "tests pass."

> **Design (agreed in the Phase 4 brainstorm):** separate stateful monitor that debounces the raw verdict; the deterministic checks/fusion are unchanged. Timeline plot is 4c.

## Current State (anchors)

- `app/health/models.py`: `HealthState` (OK/WARNING/FAULT/UNKNOWN); `HealthReport.final_state`, `.diagnostic_summary`.
- `main.py` `__init__` (≈lines 47-51): builds `self.health_pipeline = pipeline_for_profile(...)`, then `self.latest_health_report = None`, `self._last_health_state = None`. Health imports near the top include `from app.health.reporting import report_rows`.
- `main.py` `start_test` has a `# Reset History` block resetting `self.score_history`/`trigger_history`/`energy_history`.
- `main.py` `_update_health_indicator(self, report)` currently sets the label from `report.final_state` and logs raw-state transitions via `self._last_health_state`, then calls `self._update_health_panel(report)`.
- 110 tests pass.

## File Structure
- Create: `app/health/monitoring.py` (`MonitoringEvent`, `RuntimeMonitor`). Test: `tests/health/test_monitoring.py`.
- Modify: `main.py` (create/reset monitor; indicator shows runtime state + logs events).

---

## Task 1: RuntimeMonitor (debounce + events)

**Files:** Create `app/health/monitoring.py`; Test `tests/health/test_monitoring.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/health/test_monitoring.py`:

```python
from app.health.models import HealthState
from app.health.monitoring import MonitoringEvent, RuntimeMonitor


class _Rep:
    """Minimal HealthReport stand-in (RuntimeMonitor only reads final_state)."""

    def __init__(self, state, summary=""):
        self.final_state = state
        self.diagnostic_summary = summary


def _feed(monitor, states):
    events = []
    for s in states:
        events.extend(monitor.update(_Rep(s)))
    return events


def test_starts_ok():
    assert RuntimeMonitor().runtime_state is HealthState.OK


def test_single_fault_does_not_flap():
    m = RuntimeMonitor(fault_persistence=3)
    _feed(m, [HealthState.FAULT])
    assert m.runtime_state is HealthState.OK


def test_sustained_fault_escalates_and_emits_event():
    m = RuntimeMonitor(fault_persistence=3)
    events = _feed(m, [HealthState.FAULT, HealthState.FAULT, HealthState.FAULT])
    assert m.runtime_state is HealthState.FAULT
    assert any(e.event_type == "fault_detected" for e in events)


def test_transient_warning_does_not_escalate():
    m = RuntimeMonitor(warn_persistence=3)
    _feed(m, [HealthState.OK, HealthState.WARNING, HealthState.OK, HealthState.WARNING])
    assert m.runtime_state is HealthState.OK


def test_sustained_warning_escalates():
    m = RuntimeMonitor(warn_persistence=2)
    events = _feed(m, [HealthState.WARNING, HealthState.WARNING])
    assert m.runtime_state is HealthState.WARNING
    assert any(e.event_type == "warning_detected" for e in events)


def test_recovery_requires_sustained_ok():
    m = RuntimeMonitor(fault_persistence=2, recovery=3)
    _feed(m, [HealthState.FAULT, HealthState.FAULT])  # establish FAULT
    assert m.runtime_state is HealthState.FAULT
    _feed(m, [HealthState.OK, HealthState.OK])  # 2 OK < recovery 3
    assert m.runtime_state is HealthState.FAULT
    events = _feed(m, [HealthState.OK])  # 3rd OK
    assert m.runtime_state is HealthState.OK
    assert any(e.event_type == "fault_cleared" for e in events)


def test_event_carries_state_and_message():
    m = RuntimeMonitor(warn_persistence=1)
    (event,) = _feed(m, [HealthState.WARNING])
    assert isinstance(event, MonitoringEvent)
    assert event.state is HealthState.WARNING
    assert event.message
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_monitoring.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.health.monitoring'`.

- [ ] **Step 3: Implement `app/health/monitoring.py`**

```python
"""Runtime Monitoring (spec Ch. 8): debounce the per-window health verdict into a
stable runtime state and emit transition events.

A new state is adopted only after the raw per-window state has persisted for that
state's threshold (recovery windows for OK, warn/fault persistence for the rest),
so transient single-window blips do not flap the reported state. Pure stdlib;
the report is duck-typed (only ``final_state`` is read).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.health.models import HealthState


@dataclass
class MonitoringEvent:
    event_type: str  # fault_detected | warning_detected | fault_cleared | health_restored | signal_lost
    state: HealthState
    message: str


class RuntimeMonitor:
    """Debounces raw per-window verdicts into a stable runtime state."""

    def __init__(self, warn_persistence: int = 3, fault_persistence: int = 3, recovery: int = 5):
        self.warn_persistence = warn_persistence
        self.fault_persistence = fault_persistence
        self.recovery = recovery
        self.runtime_state = HealthState.OK
        self._streak_state = HealthState.OK
        self._streak_count = 0

    def _threshold(self, state: HealthState) -> int:
        return {
            HealthState.OK: self.recovery,
            HealthState.WARNING: self.warn_persistence,
            HealthState.FAULT: self.fault_persistence,
            HealthState.UNKNOWN: 1,
        }[state]

    def update(self, report) -> list[MonitoringEvent]:
        raw = report.final_state
        if raw is self._streak_state:
            self._streak_count += 1
        else:
            self._streak_state = raw
            self._streak_count = 1

        if raw is not self.runtime_state and self._streak_count >= self._threshold(raw):
            previous = self.runtime_state
            self.runtime_state = raw
            return [self._event(previous, raw)]
        return []

    def _event(self, previous: HealthState, new: HealthState) -> MonitoringEvent:
        if new is HealthState.FAULT:
            event_type, message = "fault_detected", "Fault detected"
        elif new is HealthState.WARNING:
            event_type, message = "warning_detected", "Warning detected"
        elif new is HealthState.OK:
            if previous is HealthState.FAULT:
                event_type, message = "fault_cleared", "Fault cleared"
            else:
                event_type, message = "health_restored", "Health restored"
        else:  # UNKNOWN
            event_type, message = "signal_lost", "Signal lost"
        return MonitoringEvent(event_type=event_type, state=new, message=message)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_monitoring.py -q`
Expected: PASS (7 passed).

---

## Task 2: Wire the monitor into the tester

**Files:** Modify `main.py`

- [ ] **Step 1: Import the monitor**

In `main.py`, find:

```python
from app.health.reporting import report_rows
```

Add immediately AFTER it:

```python
from app.health.monitoring import RuntimeMonitor
```

- [ ] **Step 2: Create the monitor in `__init__`**

In `main.py` `__init__`, find:

```python
        self.latest_health_report = None
        self._last_health_state = None
```

Replace with:

```python
        self.latest_health_report = None
        self._last_health_state = None
        self.runtime_monitor = RuntimeMonitor()
```

- [ ] **Step 3: Reset the monitor at the start of each test**

In `main.py` `start_test`, find:

```python
        # Reset History
        self.score_history = []
        self.trigger_history = []
        self.energy_history = []
```

Add immediately AFTER it:

```python
        self.runtime_monitor = RuntimeMonitor()
```

- [ ] **Step 4: Show the smoothed runtime state + log events**

In `main.py`, replace the `_update_health_indicator` method:

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
        # Log only on transitions to avoid flooding the log every 0.5s.
        if state != self._last_health_state:
            self._last_health_state = state
            self.log(f"Signal health {state.value}: {report.diagnostic_summary}")
        self._update_health_panel(report)
```

with:

```python
    def _update_health_indicator(self, report):
        colors = {
            HealthState.OK: "green",
            HealthState.WARNING: "orange",
            HealthState.FAULT: "red",
            HealthState.UNKNOWN: "gray",
        }
        # The monitor debounces the raw per-window verdict into a stable state.
        events = self.runtime_monitor.update(report)
        state = self.runtime_monitor.runtime_state
        self.health_label.configure(
            text=f"Signal Health: {state.value}",
            foreground=colors.get(state, "gray"),
        )
        for event in events:
            self.log(f"[monitor] {event.message}: {report.diagnostic_summary}")
        self._update_health_panel(report)
```

- [ ] **Step 5: Verify**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('parses OK')"` → `parses OK`
Run: `.venv/bin/python -c "import main; print('import OK')"` → `import OK`
Run: `.venv/bin/python -m pytest tests/ -q` → all pass (117 total: 110 + 7 monitor).

- [ ] **Step 6: Headless monitor demo (no display)**

```bash
.venv/bin/python -c "
from app.health.models import HealthState
from app.health.monitoring import RuntimeMonitor
m = RuntimeMonitor(fault_persistence=3, recovery=5)
seq = [HealthState.OK, HealthState.FAULT, HealthState.OK,        # transient fault -> ignored
       HealthState.FAULT, HealthState.FAULT, HealthState.FAULT]  # sustained -> FAULT
out = []
for s in seq:
    out += [(e.event_type) for e in m.update(type('R',(),{'final_state':s,'diagnostic_summary':''})())]
print('runtime state:', m.runtime_state.value, '| events:', out)
"
```
Expected: `runtime state: FAULT | events: ['fault_detected']` — the single transient fault is ignored; only the sustained run escalates.

- [ ] **Step 7: Manual GUI verification (owner)**

Launch `./.venv/bin/python launcher.py`, start a test. Confirm the **Signal Health** indicator no longer flickers on isolated bad windows — it changes only after a problem persists (and logs `[monitor] Fault detected: …` / `[monitor] Fault cleared: …` / etc. on transitions). The per-check panel still updates every window with the raw verdicts. Classification behavior unchanged.

---

## Phase 4b Done

The indicator now reflects a debounced runtime state (fault persistence + recovery) and logs monitor events, so isolated bad windows don't flap the display. Hand back to the owner for review, manual test, and commit. **Phase 4c** adds the health timeline plot.

> Noted, not addressed here (out of scope): the pipeline's stability history (4a) and the runtime monitor reset on test start, but not on a mid-test profile/calibration change — acceptable, since switching config mid-run is rare and the monitor simply continues smoothing.

---

## Self-Review

- **Spec coverage (Ch. 8):** debounced runtime state distinct from per-window state — `RuntimeMonitor.runtime_state` (Task 1); fault persistence (warn/fault thresholds) — `_threshold` + adopt-on-streak; recovery requires consecutive OK — `recovery` threshold on the OK state; monitoring events on transitions — `MonitoringEvent` + `_event` (fault_detected/warning_detected/fault_cleared/health_restored/signal_lost); event-driven (only on transitions) — `update` returns `[]` unless the state changes; indicator shows smoothed state + logs events — Task 2.
- **Placeholder scan:** no TBD/TODO; full code each step; concrete commands and expected outputs.
- **Type consistency:** `RuntimeMonitor(warn_persistence=3, fault_persistence=3, recovery=5)` with `.runtime_state` and `update(report) -> list[MonitoringEvent]`; `MonitoringEvent(event_type, state, message)`; the report is duck-typed (`.final_state`, `.diagnostic_summary`), so the monitor needs no `HealthReport` import. `main.py` creates `self.runtime_monitor` in `__init__`, resets it in `start_test`, and reads `runtime_monitor.runtime_state` / iterates events in `_update_health_indicator`. The old `_last_health_state` field is left set (harmless) but no longer drives the indicator.
