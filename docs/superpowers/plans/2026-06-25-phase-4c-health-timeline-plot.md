# Phase 4c — Health Timeline Plot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Health Timeline" strip-chart to the right pane showing the debounced runtime health state (OK / WARN / FAULT) over time.

**Architecture:** The plot grid grows from 3×2 to 4×2 (with a shorter bottom row) to add a full-width `ax_health`. `main.py` records `(time, level)` into a `health_state_history` list each window (level 0/1/2 from the runtime monitor's state), resets it per test, and `update_plots` draws it as a stepped line with OK/WARN/FAULT y-ticks. UI-only; no change to the health engine.

**Tech Stack:** matplotlib (GridSpec, step plot), Tkinter; pytest (regression only).

> **Commit policy:** Owner commits manually. **Do NOT run git commit/add/push.** End at verification.

> **Testing reality:** Tkinter/matplotlib layout — verified by `ast.parse` + `import main` + the unchanged 117-test suite + a manual GUI check. (The level mapping is a trivial inline dict; no separate unit.)

## Current State (anchors)

- `main.py` `_setup_ui` plot grid: `gs = self.plot_fig.add_gridspec(3, 2)` then five `add_subplot`s (`ax_wave` row0 span, `ax_spec`/`ax_energy` row1, `ax_time`/`ax_hist` row2), then `self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=right_frame)`.
- `main.py` `__init__` "History Data for Plots" block: `self.score_history = []`, `self.trigger_history = [] # (time, is_trigger)`, `self.energy_history = []  # (time, rms)`.
- `main.py` `start_test` reset block: `self.score_history = []` / `self.trigger_history = []` / `self.energy_history = []` then `self.runtime_monitor = RuntimeMonitor()` (added in 4b).
- `main.py` `_update_health_indicator` (post-4b): `events = self.runtime_monitor.update(report)` then `state = self.runtime_monitor.runtime_state`, configures the label, logs events, calls `self._update_health_panel(report)`.
- `main.py` `update_plots`: ends with the "Update Timeline" (`ax_time`) block, then `self.plot_canvas.draw_idle()`, then `except Exception as e: self.log(f"Plot Error: {e}")`.
- 117 tests pass.

## File Structure
- Modify: `main.py` only.

---

## Task 1: Add the Health Timeline plot

**Files:** Modify `main.py`

- [ ] **Step 1: Grow the grid and add `ax_health`**

In `main.py` `_setup_ui`, find:

```python
        gs = self.plot_fig.add_gridspec(3, 2)
        self.ax_wave = self.plot_fig.add_subplot(gs[0, :])      # waveform: full-width top
        self.ax_spec = self.plot_fig.add_subplot(gs[1, 0])      # spectrogram
        self.ax_energy = self.plot_fig.add_subplot(gs[1, 1])    # energy timeline
        self.ax_time = self.plot_fig.add_subplot(gs[2, 0])      # trigger timeline
        self.ax_hist = self.plot_fig.add_subplot(gs[2, 1])      # score distribution
```

Replace with:

```python
        gs = self.plot_fig.add_gridspec(4, 2, height_ratios=[2, 2, 2, 1.3])
        self.ax_wave = self.plot_fig.add_subplot(gs[0, :])      # waveform: full-width top
        self.ax_spec = self.plot_fig.add_subplot(gs[1, 0])      # spectrogram
        self.ax_energy = self.plot_fig.add_subplot(gs[1, 1])    # energy timeline
        self.ax_time = self.plot_fig.add_subplot(gs[2, 0])      # trigger timeline
        self.ax_hist = self.plot_fig.add_subplot(gs[2, 1])      # score distribution
        self.ax_health = self.plot_fig.add_subplot(gs[3, :])    # health timeline: full-width strip
```

- [ ] **Step 2: Initialise the history list in `__init__`**

In `main.py` `__init__`, find:

```python
        self.score_history = []
        self.trigger_history = [] # (time, is_trigger)
        self.energy_history = []  # (time, rms)
```

Replace with:

```python
        self.score_history = []
        self.trigger_history = [] # (time, is_trigger)
        self.energy_history = []  # (time, rms)
        self.health_state_history = []  # (time, level: 0=OK, 1=WARNING, 2=FAULT)
```

- [ ] **Step 3: Reset it in `start_test`**

In `main.py` `start_test`, find:

```python
        self.energy_history = []
        self.runtime_monitor = RuntimeMonitor()
```

Replace with:

```python
        self.energy_history = []
        self.health_state_history = []
        self.runtime_monitor = RuntimeMonitor()
```

- [ ] **Step 4: Record the runtime state each window**

In `main.py` `_update_health_indicator`, find:

```python
        events = self.runtime_monitor.update(report)
        state = self.runtime_monitor.runtime_state
        self.health_label.configure(
```

Replace with:

```python
        events = self.runtime_monitor.update(report)
        state = self.runtime_monitor.runtime_state
        level = {HealthState.OK: 0, HealthState.WARNING: 1, HealthState.FAULT: 2}.get(state, 0)
        self.health_state_history.append((time.time() - self.start_time, level))
        self.health_label.configure(
```

- [ ] **Step 5: Draw the timeline in `update_plots`**

In `main.py` `update_plots`, find:

```python
                self.ax_time.tick_params(labelsize=7)
                self.ax_time.grid(True)

            self.plot_canvas.draw_idle()
```

Replace with:

```python
                self.ax_time.tick_params(labelsize=7)
                self.ax_time.grid(True)

            # Update Health Timeline (debounced runtime state over time)
            if self.health_state_history:
                times, levels = zip(*self.health_state_history)
                self.ax_health.clear()
                self.ax_health.step(times, levels, where="post", color="purple")
                self.ax_health.set_title("Health Timeline", fontsize=9)
                self.ax_health.set_ylim(-0.2, 2.2)
                self.ax_health.set_yticks([0, 1, 2])
                self.ax_health.set_yticklabels(["OK", "WARN", "FAULT"], fontsize=7)
                self.ax_health.set_xlabel("Time (s)", fontsize=8)
                self.ax_health.tick_params(labelsize=7)
                self.ax_health.grid(True)

            self.plot_canvas.draw_idle()
```

- [ ] **Step 6: Verify**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('parses OK')"` → `parses OK`
Run: `.venv/bin/python -c "import main; print('import OK')"` → `import OK` (keras FutureWarning is fine)
Run: `.venv/bin/python -m pytest tests/ -q` → 117 passed (UI-only change; no test touches the plots).
Run: `git diff main.py | grep '^-' | grep -v '^---'` → only the two replaced lines (`add_gridspec(3, 2)` and the `draw_idle()`/`grid(True)` anchor lines) should appear as removed; confirm no inference/health logic removed.

- [ ] **Step 7: Manual GUI verification (owner)**

Launch `./.venv/bin/python launcher.py`, start a test. Confirm the right pane now shows six charts including a bottom **"Health Timeline"** strip with OK/WARN/FAULT y-ticks; it tracks the (debounced) Signal Health indicator over time, the grid still fits without scrolling, and resizing rescales it. Classification and the other charts are unchanged.

---

## Phase 4c Done — Phase 4 complete

The right pane now has a Health Timeline showing the runtime state over time, completing Phase 4 (stability checks + runtime monitoring + timeline). Hand back to the owner for review, manual test, and commit. Next: **Phase 5 — Startup Validation.**

---

## Self-Review

- **Spec coverage:** health timeline plot showing health state over time (design §5 Phase 4) — Task 1 adds `ax_health` + the `update_plots` strip-chart fed by `health_state_history`, which records the debounced `runtime_monitor.runtime_state` (consistent with the 4b indicator). Grid restructured 3×2 → 4×2 with a shorter bottom row so it stays laptop-fit (no scroll). UI-only; engine untouched.
- **Placeholder scan:** no TBD/TODO; full code each step; concrete commands and expected output.
- **Type consistency:** `self.health_state_history` created in `__init__`, reset in `start_test`, appended in `_update_health_indicator` (as `(time, level)`), read in `update_plots` — names match. `self.ax_health` created in `_setup_ui` and used only in `update_plots`. Level mapping uses `HealthState.OK/WARNING/FAULT` (already imported in `main.py`), default 0 for UNKNOWN. `self.start_time` (set in `start_test`) is the time origin, matching the other history series.
