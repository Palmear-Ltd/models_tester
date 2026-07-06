# Responsive Plot Grid — Fit Charts on a Field Laptop

**Date:** 2026-06-24
**Status:** Approved (design); ready for implementation planning
**Scope:** UI/plot layout change in `main.py`. No change to what is plotted, to inference, or to the health subsystem.

## 1. Purpose

The right pane stacks **five separate matplotlib figures vertically inside a scrollable canvas**, each a fixed `Figure(figsize=(8, 2.6), dpi=100)` (≈800×260 px) with its **own navigation toolbar**. Five of those ≈ 1500+ px tall, so on a field laptop only 2–3 are visible and the user must scroll. Goal: make all charts visible at once, readable, and responsive to window size — no scrolling.

## 2. Current State

- `_setup_ui`: `right_frame` → `plot_container` → a vertical scroll `tk.Canvas` (`plot_canvas`) + `Scrollbar` → inner `plot_frame`.
- `add_plot_section(title)` builds, per chart: a section `Frame`, a `Figure(figsize=(8, 2.6), dpi=100)` with one `add_subplot(111)`, a `FigureCanvasTkAgg`, and a `NavigationToolbar2Tk`. Called 5×, producing `self.ax_wave/canvas_wave`, `ax_hist/canvas_hist`, `ax_spec/canvas_spec`, `ax_energy/canvas_energy`, `ax_time/canvas_time` for: Waveform, Score Distribution, Mel Spectrogram, Energy Timeline, Trigger Timeline.
- `update_plots` (every 500 ms): clears each `ax`, redraws, and calls each `canvas_*.draw()` (five separate draws).

## 3. Target Design

**One responsive figure.** Replace the scroll canvas + 5 figures with a single `Figure` (constrained layout) hosting all five axes via a `GridSpec`, in a single `FigureCanvasTkAgg` packed `fill="both", expand=True` directly in `right_frame` — so it resizes with the window and never scrolls. **No per-chart navigation toolbars.**

Layout (3 rows): waveform spans the full top row; the rest fill a 2×2 below.

```
┌─────────────────────────────────────────────┐
│  Waveform  (full width)                       │
├──────────────────────┬────────────────────────┤
│  Mel Spectrogram     │  Energy Timeline        │
├──────────────────────┼────────────────────────┤
│  Trigger Timeline    │  Score Distribution     │
└──────────────────────┴────────────────────────┘
```

- GridSpec: 3 rows × 2 cols; row 0 = waveform spanning both columns; row 1 = (spectrogram, energy); row 2 = (trigger, score distribution).
- Axis handles keep their existing names so `update_plots`/snapshot code is unchanged otherwise: `self.ax_wave`, `self.ax_spec`, `self.ax_energy`, `self.ax_time`, `self.ax_hist`.
- A single `self.plot_canvas` (the `FigureCanvasTkAgg`); `update_plots` clears the axes and calls `self.plot_canvas.draw_idle()` **once** per cycle.
- Modest default figure size with `constrained_layout=True` (or `figure.tight_layout`) so titles/labels don't overlap when packed small; the canvas widget expands to the pane. Use slightly smaller title/label fonts for legibility at reduced size.

## 4. Removed / Changed

- Remove: the vertical scroll `plot_canvas` (tk.Canvas) + scrollbar + inner `plot_frame` + their `<Configure>` bindings, the `add_plot_section` helper, the 5 per-chart figures, and all 5 `NavigationToolbar2Tk` instances and the per-chart `canvas_*` objects.
- Change: `update_plots` draws the single canvas once (replacing the five `canvas_*.draw()` calls); the per-axis `clear()`/plot logic is otherwise preserved.

## 5. Out of Scope

- No change to the data plotted, decimation, colormap, or analysis.
- No new charts; no chart removed (all five kept).
- No shared toolbar (dropped entirely; can be revisited if zoom/pan is ever needed).

## 6. Testing

Tkinter/matplotlib layout — automated coverage is limited:
- `main.py` parses (`ast.parse`) and imports cleanly.
- Full pytest suite still passes (81; regression guard for non-UI code).
- **Manual GUI verification (owner):** right pane shows all five charts at once with no scrollbar; resizing the window rescales the grid; waveform is full-width on top with the 2×2 below; live updates still work during a test (waveform, energy, trigger, spectrogram, score distribution all refresh); no per-chart toolbars.
