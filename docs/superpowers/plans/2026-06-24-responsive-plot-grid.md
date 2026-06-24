# Responsive Plot Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the five stacked, fixed-size, individually-toolbarred plot figures (in a vertical scroll canvas) with one responsive `GridSpec` figure that fills the right pane and resizes with the window — no scrolling.

**Architecture:** A single `Figure(constrained_layout=True)` holds all five axes via a 3×2 grid (waveform full-width on top; spectrogram|energy; trigger|score-distribution), in one `FigureCanvasTkAgg` packed to fill `right_frame`. `update_plots` clears the axes and calls one `draw_idle()` per cycle instead of five `draw()` calls. No per-chart navigation toolbars.

**Tech Stack:** matplotlib (Figure/GridSpec/FigureCanvasTkAgg), Tkinter, pytest.

> **Commit policy:** Owner commits manually. **Do NOT run git commit/add/push.** End at verification.

> **Reference spec:** `docs/superpowers/specs/2026-06-24-responsive-plot-grid-design.md`.

> **Testing reality:** Tkinter/matplotlib layout — verified by `ast.parse`, `import main`, the unchanged 81-test suite, and a manual GUI check.

---

## Task 1: Replace the plot setup with one responsive grid figure

**Files:** Modify `main.py`

- [ ] **Step 1: Drop the now-unused toolbar import**

Find:
```python
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
```
Replace with:
```python
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
```

- [ ] **Step 2: Replace the plots-area block**

Replace the entire block from the comment `# --- Plots Area (Right) ---` through the last `add_plot_section(...)` line (`self.ax_time, self.canvas_time = add_plot_section("Trigger Timeline")`) with:

```python
        # --- Plots Area (Right): one responsive grid figure (no scroll) ---
        self.plot_fig = Figure(figsize=(7, 6), dpi=100, constrained_layout=True)
        gs = self.plot_fig.add_gridspec(3, 2)
        self.ax_wave = self.plot_fig.add_subplot(gs[0, :])      # waveform: full-width top
        self.ax_spec = self.plot_fig.add_subplot(gs[1, 0])      # spectrogram
        self.ax_energy = self.plot_fig.add_subplot(gs[1, 1])    # energy timeline
        self.ax_time = self.plot_fig.add_subplot(gs[2, 0])      # trigger timeline
        self.ax_hist = self.plot_fig.add_subplot(gs[2, 1])      # score distribution
        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=right_frame)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)
```

- [ ] **Step 3: Verify**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('parses OK')"` → `parses OK`
Run: `grep -c "add_plot_section\|canvas_wave\|canvas_hist\|canvas_spec\|canvas_energy\|canvas_time\|NavigationToolbar2Tk\|plot_frame" main.py` → expect `0` (all old plumbing gone; `update_plots` references in Task 2 still pending — see note). NOTE: this grep will be non-zero until Task 2 removes the `canvas_*.draw()` calls in `update_plots`; that's expected here.

---

## Task 2: Rewrite `update_plots` to draw the single canvas once

**Files:** Modify `main.py`

- [ ] **Step 1: Replace the `update_plots` try-block**

Replace the body from `try:` through the `except Exception as e:` / `self.log(f"Plot Error: {e}")` lines inside `update_plots` with:

```python
        try:
            if self.raw_audio_snapshot is not None:
                self.ax_wave.clear()
                self.ax_wave.plot(self.raw_audio_snapshot[::10])  # Decimate for speed
                self.ax_wave.set_title("Audio Waveform", fontsize=9)
                self.ax_wave.set_ylim(-1, 1)
                self.ax_wave.tick_params(labelsize=7)
                self.ax_wave.grid(True)

            if self.score_history:
                self.ax_hist.clear()
                self.ax_hist.hist(self.score_history, bins=10, range=(0, 1), color='skyblue', edgecolor='black')
                self.ax_hist.set_title("Score Distribution", fontsize=9)
                self.ax_hist.set_xlim(0, 1)
                self.ax_hist.tick_params(labelsize=7)
                self.ax_hist.grid(True)

            if self.current_spectrogram is not None:
                self.ax_spec.clear()
                self.ax_spec.imshow(self.current_spectrogram.T, aspect='auto', origin='lower', cmap='inferno')
                self.ax_spec.set_title("Mel Spectrogram", fontsize=9)
                self.ax_spec.set_ylabel("Mel Bands", fontsize=8)
                self.ax_spec.set_xlabel("Time Frames", fontsize=8)
                self.ax_spec.tick_params(labelsize=7)

            if self.energy_history:
                times, energy = zip(*self.energy_history)
                self.ax_energy.clear()
                self.ax_energy.plot(times, energy, 'g-')
                self.ax_energy.set_title("Energy Timeline (RMS)", fontsize=9)
                self.ax_energy.set_ylim(0, max(0.1, max(energy) * 1.2))  # Dynamic limit
                self.ax_energy.tick_params(labelsize=7)
                self.ax_energy.grid(True)

            if self.trigger_history:
                times, triggers = zip(*self.trigger_history)
                self.ax_time.clear()
                self.ax_time.plot(times, triggers, 'r-o', markersize=4)
                self.ax_time.set_title("Trigger Timeline (1=Pos, 0=Neg)", fontsize=9)
                self.ax_time.set_ylim(-0.1, 1.1)
                self.ax_time.set_xlabel("Time (s)", fontsize=8)
                self.ax_time.tick_params(labelsize=7)
                self.ax_time.grid(True)

            self.plot_canvas.draw_idle()
        except Exception as e:
            self.log(f"Plot Error: {e}")
```

- [ ] **Step 2: Verify**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('parses OK')"` → `parses OK`
Run: `grep -c "add_plot_section\|canvas_wave\|canvas_hist\|canvas_spec\|canvas_energy\|canvas_time\|NavigationToolbar2Tk\|plot_frame" main.py` → expect `0` (all old plot plumbing removed).
Run: `.venv/bin/python -c "import main; print('import OK')"` → `import OK` (keras FutureWarning is fine).
Run: `.venv/bin/python -m pytest tests/ -q` → 81 passed.

- [ ] **Step 3: Manual GUI verification (owner)**

Launch `./.venv/bin/python launcher.py`; the right pane shows all five charts at once (waveform full-width top; spectrogram|energy; trigger|score-distribution) with no scrollbar; resizing the window rescales the grid; during a test all charts update live; no per-chart toolbars.

---

## Self-Review

- **Spec coverage:** single responsive figure + GridSpec arrangement (§3) — Task 1; one `draw_idle()` per cycle replacing five `draw()` (§3/§4) — Task 2; removed scroll canvas / `add_plot_section` / 5 figures / toolbars (§4) — Task 1 Step 2 + import drop Step 1; smaller fonts for legibility (§3) — Task 2 fontsizes; data plotted unchanged (§5) — same plot calls preserved.
- **Placeholder scan:** none; full code in every step; the Task 1 Step 3 grep being non-zero is explicitly explained (cleared by Task 2).
- **Type consistency:** axis handles keep names `ax_wave`/`ax_spec`/`ax_energy`/`ax_time`/`ax_hist`; the new single `self.plot_canvas` (a FigureCanvasTkAgg) replaces the old per-chart `canvas_*`; `update_plots` (Task 2) uses exactly those handles + `self.plot_canvas.draw_idle()`.
