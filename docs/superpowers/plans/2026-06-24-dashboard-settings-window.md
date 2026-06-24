# Dashboard Settings Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move most configuration off the crowded main window into one tabbed Settings sub-window, leaving a compact run bar (input source + device/file + START/STOP + Settings) and the diagnosis thresholds on the main window.

**Architecture:** Rewrite the dead `app/ui/settings_dialog.py::SettingsDialog` into the real, modal, 3-tab (Model & Scaler / Acquisition & Output / Preprocessing) dialog that binds to the app's existing `tk` variables. Hoist the three vars that move into that (lazily built) dialog up to `__init__` so they exist before any test starts. Replace the left-pane "Configuration" card with a Run bar + Thresholds strip and rework `refresh_devices` for the new layout. No change to inference, acquisition, or the health subsystem.

**Tech Stack:** Python 3.13, Tkinter/ttk (Notebook, Combobox), pytest. Run via `.venv/bin/python -m pytest`.

> **Commit policy:** Owner commits/pushes **manually** after review. **Do NOT run `git commit`/`git add`/`git push`.** Each task ends at "verification passes."

> **Reference spec:** `docs/superpowers/specs/2026-06-24-dashboard-settings-window-design.md`.

> **Testing reality:** This is a Tkinter layout refactor; most of it is verified by `ast.parse`, module import, the unchanged 80-test suite (regression guard for non-UI code), and a manual GUI checklist. Only the dialog's importability/construction is unit-tested (no display needed for that).

---

## Current State

- `main.py` `ModelsTesterApp._setup_ui` builds a left-pane `config_frame` "Configuration" card (grid) holding: model path (+Browse), scaler path (+Browse), window duration, input source (mic/wav), device combo / wav Browse, output dir (+Browse), thresholds (score/susp/inf), user label, save-results toggle, inference mode. Then `dash_frame`, the Health Profile dropdown, the "Signal Health Detail" panel, a `ctrl_frame` with the **START TEST** button, and the status log.
- Several `tk` vars are created **inside `_setup_ui`**: `model_path_var` (line ~169), `scaler_path_var` (~175), `duration_var` (~183), `input_type_var` (~188), `device_var` (~199), `score_thresh_var` (~217), `susp_limit_var` (~221), `inf_limit_var` (~225). Others (`use_filter_var`, `low_cut_var`, `up_cut_var`, `sub_win_size_var`, `sub_hop_size_var`, `n_mels_var`, `seq_len_var`, `inference_mode_var`, `save_results_var`, `user_label_var`, `output_dir_var`) are in `__init__`.
- `refresh_devices` grids `device_combo`/`file_btn` into `config_frame` at `row=4`.
- File ▸ Settings calls `open_prep_settings`, which builds a modal Toplevel inline (filter/FFT/hop/mel/seq).
- `app/ui/settings_dialog.py::SettingsDialog` exists and is exported by `app/ui/__init__.py`, but is unused and references vars the app never defines (`enable_downsample_var`, `downsample_sr_var`, `use_pcen_var`). It is dead code.
- 80 tests pass.

## File Structure

**Modify:**
- `app/ui/settings_dialog.py` — rewrite `SettingsDialog` as a 3-tab modal dialog taking the `app`.
- `main.py` — hoist 3 vars to `__init__`; replace `config_frame` with Run bar + Thresholds; rework `refresh_devices`; remove the old `ctrl_frame` Start button; point `open_prep_settings` at the new dialog.

**Create:**
- `tests/test_settings_dialog.py` — headless import/construction test.

---

## Task 1: Hoist the moving vars into `__init__`

`model_path_var`, `scaler_path_var`, and `duration_var` move into the Settings dialog (built lazily when opened). They must exist before any test starts, so create them in `__init__`.

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add the three vars in `__init__`**

In `main.py`, find:

```python
        self.user_label_var = tk.StringVar(value="infested") # User expectation
        self.output_dir_var = tk.StringVar(value=self.default_output_dir)
```

Add immediately AFTER it (the `default_model_path`/`default_scaler_path` attributes are already set just above this point in `__init__`):

```python
        self.model_path_var = tk.StringVar(value=self.default_model_path)
        self.scaler_path_var = tk.StringVar(value=self.default_scaler_path)
        self.duration_var = tk.DoubleVar(value=2.5)
```

- [ ] **Step 2: Remove the inline creation in `_setup_ui` (model path)**

In `main.py` find and DELETE this single line (the widget on the next line keeps using `self.model_path_var`):

```python
        self.model_path_var = tk.StringVar(value=self.default_model_path)
```

- [ ] **Step 3: Remove the inline creation in `_setup_ui` (scaler path)**

Find and DELETE this single line:

```python
        self.scaler_path_var = tk.StringVar(value=self.default_scaler_path)
```

- [ ] **Step 4: Remove the inline creation in `_setup_ui` (duration)**

Find and DELETE this single line:

```python
        self.duration_var = tk.DoubleVar(value=2.5)
```

- [ ] **Step 5: Verify**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('main.py parses OK')"`
Expected: `main.py parses OK`

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 80 passed (no behavior change; vars just created earlier).

---

## Task 2: Rewrite `SettingsDialog` as the 3-tab window and wire it up

**Files:**
- Modify: `app/ui/settings_dialog.py`, `main.py`
- Test: `tests/test_settings_dialog.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_settings_dialog.py`:

```python
from app.ui.settings_dialog import SettingsDialog


class _FakeApp:
    """Minimal stand-in; SettingsDialog must construct without touching Tk."""

    root = None


def test_settings_dialog_constructs_without_display():
    dlg = SettingsDialog(_FakeApp())
    assert hasattr(dlg, "show")
    assert callable(dlg.show)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_settings_dialog.py -q`
Expected: FAIL — `SettingsDialog.__init__` currently requires a `config_vars` argument, so `SettingsDialog(_FakeApp())` raises `TypeError`.

- [ ] **Step 3: Rewrite `app/ui/settings_dialog.py`**

Replace the ENTIRE contents of `app/ui/settings_dialog.py` with:

```python
"""Tabbed Settings dialog for the model tester.

A single modal window consolidating all configuration that does not live on the
main run bar. It binds directly to the `tk` variables already on the app, so
edits take effect on the next test start (same as before) — there is no separate
apply/cancel buffering. Built lazily when opened.
"""

import tkinter as tk
from tkinter import ttk


class SettingsDialog:
    """Modal, 3-tab settings window bound to the app's tk variables."""

    def __init__(self, app):
        self.app = app
        self.window = None

    def show(self):
        app = self.app
        self.window = tk.Toplevel(app.root)
        self.window.title("Settings")
        self.window.geometry("480x420")
        self.window.transient(app.root)
        self.window.grab_set()

        notebook = ttk.Notebook(self.window)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self._build_model_tab(notebook)
        self._build_acquisition_tab(notebook)
        self._build_preprocessing_tab(notebook)

        ttk.Button(self.window, text="Close", command=self.window.destroy).pack(
            fill="x", padx=10, pady=(0, 10)
        )

    def _build_model_tab(self, notebook):
        app = self.app
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text="Model & Scaler")
        tab.columnconfigure(1, weight=1)

        ttk.Label(tab, text="Model Path (.tflite):").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.model_path_var).grid(row=0, column=1, padx=5, sticky="ew")
        ttk.Button(tab, text="Browse", command=app.load_model_dialog).grid(row=0, column=2)

        ttk.Label(tab, text="Scaler Path (.json/.npz):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.scaler_path_var).grid(row=1, column=1, padx=5, sticky="ew")
        ttk.Button(tab, text="Browse", command=app.load_scaler_dialog).grid(row=1, column=2)

    def _build_acquisition_tab(self, notebook):
        app = self.app
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text="Acquisition & Output")
        tab.columnconfigure(1, weight=1)

        ttk.Label(tab, text="Window Duration (sec):").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Spinbox(tab, from_=0.5, to=10.0, increment=0.5, textvariable=app.duration_var, width=12).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(tab, text="Output Dir:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.output_dir_var).grid(row=1, column=1, padx=5, sticky="ew")
        ttk.Button(tab, text="Browse", command=app.browse_output_dir).grid(row=1, column=2)

        ttk.Label(tab, text="User Label:").grid(row=2, column=0, sticky="w", pady=5)
        label_frame = ttk.Frame(tab)
        label_frame.grid(row=2, column=1, sticky="w", padx=5)
        ttk.Radiobutton(label_frame, text="Infested", variable=app.user_label_var, value="infested").pack(side="left", padx=5)
        ttk.Radiobutton(label_frame, text="Healthy", variable=app.user_label_var, value="healthy").pack(side="left", padx=5)

        ttk.Checkbutton(tab, text="Save results and audio", variable=app.save_results_var).grid(row=3, column=0, columnspan=3, sticky="w", pady=5)

        ttk.Label(tab, text="Inference Mode:").grid(row=4, column=0, sticky="w", pady=5)
        mode_frame = ttk.Frame(tab)
        mode_frame.grid(row=4, column=1, sticky="w", padx=5)
        ttk.Radiobutton(mode_frame, text="Sliding Window", variable=app.inference_mode_var, value="sliding").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text=f"Single Shot ({app.single_shot_duration_sec}s)", variable=app.inference_mode_var, value="single").pack(side="left", padx=5)

    def _build_preprocessing_tab(self, notebook):
        app = self.app
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text="Preprocessing")
        tab.columnconfigure(1, weight=1)

        ttk.Checkbutton(tab, text="Use Bandpass Filter", variable=app.use_filter_var).grid(row=0, column=0, columnspan=2, sticky="w", pady=5)

        ttk.Label(tab, text="Low Cut (Hz):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.low_cut_var, width=15).grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(tab, text="Up Cut (Hz):").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.up_cut_var, width=15).grid(row=2, column=1, sticky="w", padx=5)

        ttk.Label(tab, text="FFT Win Size (s):").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.sub_win_size_var, width=15).grid(row=3, column=1, sticky="w", padx=5)

        ttk.Label(tab, text="Hop Size (s):").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.sub_hop_size_var, width=15).grid(row=4, column=1, sticky="w", padx=5)

        ttk.Label(tab, text="Mel Bands:").grid(row=5, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.n_mels_var, width=15).grid(row=5, column=1, sticky="w", padx=5)

        ttk.Label(tab, text="Seq Len (frames):").grid(row=6, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.seq_len_var, width=15).grid(row=6, column=1, sticky="w", padx=5)
```

- [ ] **Step 4: Run to verify the test passes**

Run: `.venv/bin/python -m pytest tests/test_settings_dialog.py -q`
Expected: PASS (1 passed) — `SettingsDialog(_FakeApp())` constructs and exposes `show`.

- [ ] **Step 5: Point `open_prep_settings` at the new dialog**

In `main.py`, add this import near the other `app.*` imports at the top (after `from app.health.reporting import report_rows`):

```python
from app.ui import SettingsDialog
```

Then replace the ENTIRE `open_prep_settings` method body (from `def open_prep_settings(self):` through its final `ttk.Button(... command=settings_win.destroy ...)` line) with:

```python
    def open_prep_settings(self):
        SettingsDialog(self).show()
```

- [ ] **Step 6: Verify**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('main.py parses OK')"`
Expected: `main.py parses OK`

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 81 passed (80 + 1 new).

---

## Task 3: Replace the Configuration card with a Run bar + Thresholds strip

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Replace the `config_frame` block with the Run bar + Thresholds**

In `main.py` `_setup_ui`, replace the entire Configuration block — from the comment line `# --- Configuration Frame (Left) ---` through the end of the inference-mode block (the `ttk.Radiobutton(mode_frame, text=f"Single Shot ...")` line, the last line before the `# --- Dashboard Frame (Left) ---` comment) — with the following. (Open the file, find those two boundary lines, and replace everything between them inclusive.)

```python
        # --- Run Bar (Left): input source + device/file + controls ---
        run_frame = ttk.LabelFrame(left_frame, text="Run", padding=12, style="Card.TLabelframe")
        run_frame.pack(fill="x", padx=10, pady=5)
        run_frame.columnconfigure(1, weight=1)

        ttk.Label(run_frame, text="Input Source:").grid(row=0, column=0, sticky="w", pady=5)
        self.input_type_var = tk.StringVar(value="mic")
        input_frame = ttk.Frame(run_frame)
        input_frame.grid(row=0, column=1, columnspan=2, sticky="w", padx=5)
        ttk.Radiobutton(input_frame, text="Microphone", variable=self.input_type_var, value="mic", command=self.refresh_devices).pack(side="left", padx=5)
        ttk.Radiobutton(input_frame, text="Wav File", variable=self.input_type_var, value="file", command=self.refresh_devices).pack(side="left", padx=5)

        self.device_label = ttk.Label(run_frame, text="Device:")
        self.device_label.grid(row=1, column=0, sticky="w", pady=5)
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(run_frame, textvariable=self.device_var)
        self.device_combo.grid(row=1, column=1, padx=5, sticky="ew")
        self.file_btn = ttk.Button(run_frame, text="Browse", command=self.browse_wav_file)
        self.refresh_devices()

        controls_frame = ttk.Frame(run_frame)
        controls_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.start_btn = ttk.Button(controls_frame, text="START TEST", command=self.toggle_test, width=18, style="Primary.TButton")
        self.start_btn.pack(side="left")
        ttk.Button(controls_frame, text="⚙ Settings", command=self.open_prep_settings).pack(side="left", padx=8)

        # --- Thresholds strip (Left) ---
        limits_frame = ttk.LabelFrame(left_frame, text="Thresholds", padding=8, style="Card.TLabelframe")
        limits_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(limits_frame, text="Score Thresh:").pack(side="left", padx=2)
        self.score_thresh_var = tk.DoubleVar(value=0.5)
        ttk.Entry(limits_frame, textvariable=self.score_thresh_var, width=5).pack(side="left", padx=2)
        ttk.Label(limits_frame, text="Suspicious >=").pack(side="left", padx=2)
        self.susp_limit_var = tk.IntVar(value=17)
        ttk.Entry(limits_frame, textvariable=self.susp_limit_var, width=5).pack(side="left", padx=2)
        ttk.Label(limits_frame, text="Infested >").pack(side="left", padx=2)
        self.inf_limit_var = tk.IntVar(value=27)
        ttk.Entry(limits_frame, textvariable=self.inf_limit_var, width=5).pack(side="left", padx=2)
```

- [ ] **Step 2: Remove the now-duplicate Start button (`ctrl_frame`)**

The START button now lives in the run bar, so delete the old control area. In `main.py` find and DELETE this block:

```python
        # --- Control Area (Left) ---
        ctrl_frame = ttk.Frame(left_frame, padding=10)
        ctrl_frame.pack(fill="x")
        
        self.start_btn = ttk.Button(ctrl_frame, text="START TEST", command=self.toggle_test, width=20, style="Primary.TButton")
        self.start_btn.pack(pady=5)
```

(If the exact whitespace differs, delete the `ctrl_frame` creation, its `.pack`, and the `self.start_btn = ttk.Button(ctrl_frame, ...)` + its `.pack` — i.e. the four/five lines that build the old control area. Do not touch the status-log block that follows.)

- [ ] **Step 3: Rework `refresh_devices` for the run-bar layout**

In `main.py`, replace the ENTIRE `refresh_devices` method with:

```python
    def refresh_devices(self):
        input_type = self.input_type_var.get()
        if input_type == 'mic':
            self.file_btn.grid_remove()
            self.device_label.configure(text="Device:")
            self.device_combo.grid(row=1, column=1, padx=5, sticky="ew")
            try:
                devices = sd.query_devices()
                input_devices = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]
                self.device_combo['values'] = input_devices
                if input_devices:
                    self.device_combo.current(0)
            except Exception as e:
                self.log(f"Error listing devices: {e}")
        else:
            self.device_label.configure(text="Wav File:")
            self.device_combo.grid(row=1, column=1, padx=5, sticky="ew")
            self.file_btn.grid(row=1, column=2, padx=5)
```

- [ ] **Step 4: Verify parse + additive-to-logic diff**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('main.py parses OK')"`
Expected: `main.py parses OK`

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: 81 passed (UI-only change; no test touches the relayout).

- [ ] **Step 5: Headless smoke check (import + class construction, no display)**

Run:
```bash
.venv/bin/python -c "
import ast, main  # importing main must not construct the GUI
from app.ui import SettingsDialog
print('imports OK; SettingsDialog:', SettingsDialog.__name__)
"
```
Expected: `imports OK; SettingsDialog: SettingsDialog`. (Importing `main` defines the class but does not instantiate `ModelsTesterApp`, so no display is needed.)

- [ ] **Step 6: Manual GUI verification (owner)**

(Requires full deps.) Launch `./.venv/bin/python launcher.py`. Confirm:
- The main left pane shows: **Run** card (Input Source mic/wav, device dropdown or wav Browse, **START TEST** + **⚙ Settings**), **Thresholds** strip, Dashboard, Health Profile, Signal Health Detail, log. The old "Configuration" card is gone.
- Switching **Microphone ↔ Wav File** swaps the device dropdown for the wav file Browse correctly.
- **⚙ Settings** (and File ▸ Settings) opens the tabbed window: Model & Scaler / Acquisition & Output / Preprocessing. Editing a value (e.g. model path, duration, low cut) and starting a test behaves exactly as before.
- A mic test and a wav-file test both run and classify as before; the health indicator/panel still update.

---

## Dashboard Settings Window Done

Configuration now lives in one tabbed Settings window; the main window is a clean run/monitor surface with a run bar + thresholds. The dormant `SettingsDialog` is now the real implementation. Hand back to the owner for review, manual test, and commit. (Phase 3 — Calibration — resumes after this.)

---

## Self-Review

- **Spec coverage:** run bar with mic/wav + device/file selector + START (§3.1) — Task 3; thresholds strip on main (§3.2) — Task 3; tabbed Settings window Model&Scaler / Acquisition&Output / Preprocessing (§4) — Task 2; reuse existing `tk` vars, apply-on-next-test (§5) — dialog binds app vars, no new state; retire dormant `SettingsDialog` by making it the real implementation (§5) — Task 2 rewrite; `refresh_devices` behavior preserved (§5) — Task 3 Step 3; Settings opened from button + File menu (§3) — Task 2 Step 5 + Task 3 Step 1. Out-of-scope items (no new settings, no inference change) respected.
- **Placeholder scan:** no TBD/TODO; every code step shows full code; verification commands are concrete. The one fuzzy instruction (boundary-span replacement in Task 3 Step 1 / Step 2) names exact boundary lines and full replacement text — unavoidable for a large contiguous block, and the implementer reads the file.
- **Type consistency:** `SettingsDialog(app)` with `.show()` (Task 2) matches the call site `SettingsDialog(self).show()` (Task 2 Step 5) and the test (`SettingsDialog(_FakeApp())`); the dialog references only vars that exist on the app after Task 1 (`model_path_var`, `scaler_path_var`, `duration_var`) plus `__init__` vars and methods (`load_model_dialog`, `load_scaler_dialog`, `browse_output_dir`, `single_shot_duration_sec`, `root`). `refresh_devices` grids into `run_frame` row 1 — matching where Task 3 Step 1 places `device_combo`/`file_btn`. `self.start_btn` is created once (run bar) after Task 3 Step 2 removes the old one.
- **Var-existence safety:** Task 1 guarantees `model_path_var`/`scaler_path_var`/`duration_var` exist from `__init__`, so starting a test without ever opening Settings still works.
