# Dashboard Refactor — Configuration in a Tabbed Settings Window

**Date:** 2026-06-24
**Status:** Approved (design); ready for implementation planning
**Scope:** UI reorganization of the model tester (`main.py`). No change to inference, the health subsystem, or acquisition behavior.

## 1. Purpose

The tester's left pane is crowded: a large "Configuration" card sits above the Dashboard, Health Profile, Signal Health panel, Start button, and log. This refactor moves most configuration into a dedicated **tabbed Settings sub-window**, leaving the main window as a clean run/monitor surface with a compact **run bar** and the **diagnosis thresholds** for quick iteration.

Today configuration is split across two places — the left-pane "Configuration" card and a separate modal "Preprocessing Pipeline Configuration" dialog (File ▸ Settings ▸ `open_prep_settings`) — and there is a dormant, unused `app/ui/settings_dialog.py::SettingsDialog`. This refactor consolidates all of it into one window.

## 2. Current State

- **Left pane** (`left_frame`): `config_frame` "Configuration" (model path, scaler path, window duration, input source mic/wav, device combo / wav Browse, output dir, thresholds, user label, save-results toggle, inference mode) → `dash_frame` (counts, result, Signal Health indicator, energy bar) → Health Profile dropdown → "Signal Health Detail" Treeview → Start button → status log.
- **Right pane** (`right_frame`): scrollable matplotlib plots.
- **Existing sub-window:** `open_prep_settings` builds a modal `Toplevel` inline with filter/FFT/hop/mel/seq settings. `app/ui/settings_dialog.py::SettingsDialog` exists but is not wired in (dead/duplicate).
- All settings are held in `tk` variables on `ModelsTesterApp` (e.g. `model_path_var`, `scaler_path_var`, `score_thresh_var`, …) and read at test start (`load_resources`, `start_test`).

## 3. Main Window After Refactor (left pane)

1. **Run bar** (compact, top): input-source toggle **Microphone ⟷ Wav File**; mic → device dropdown, wav → file path + **Browse**; **START / STOP** button. (Reuses the existing `refresh_devices` show/hide logic, relocated — not rebuilt.)
2. **Thresholds strip**: Score / Suspicious / Infested entries (kept for quick tuning).
3. **Dashboard**: POSITIVE/NEGATIVE counts, result label, Signal Health indicator, energy bar.
4. **Health Profile** dropdown + **Signal Health Detail** panel.
5. **Status log**.

Right pane (plots) unchanged. A **"⚙ Settings"** button is added (run bar or near Start), in addition to the existing File ▸ Settings menu entry.

## 4. Settings Window (tabbed, modal)

A single modal `Toplevel` with a `ttk.Notebook`:

- **Model & Scaler** — model path (Browse), scaler path (Browse).
- **Acquisition & Output** — window duration, output dir (Browse), user label (infested/healthy), save-results toggle, inference mode (sliding / single-shot).
- **Preprocessing** — bandpass filter on/off, low cut, up cut, FFT window size, hop size, mel bands, seq len. (The current `open_prep_settings` content; `is_one_shot_model` conditionals preserved.)

A **Close** button dismisses it. The window edits the **same `tk` variables** already on `ModelsTesterApp`, so edits take effect on the next test start exactly as today (no new state, no apply/cancel buffering).

### Item placement summary

| Item | Location |
| --- | --- |
| Input source (mic/wav), device / wav file selector | Run bar (main) |
| START / STOP | Run bar (main) |
| Score / Suspicious / Infested thresholds | Thresholds strip (main) |
| Model path, Scaler path | Settings ▸ Model & Scaler |
| Window duration, Output dir, User label, Save results, Inference mode | Settings ▸ Acquisition & Output |
| Filter on/off, low/up cut, FFT win, hop, mel bands, seq len | Settings ▸ Preprocessing |

## 5. Implementation Approach

- Build the Settings window as one method/class that hosts the existing `tk` vars in three notebook tabs. Prefer to **make `app/ui/settings_dialog.py::SettingsDialog` the real implementation** (wire it up and extend it to cover all three tabs), removing the inline `open_prep_settings` body — or, if that class's shape doesn't fit cleanly, replace it. Either way, no dead duplicate remains.
- The File ▸ Settings menu item and the new "⚙ Settings" button both open this window.
- Relocating the input-source/device/wav widgets must preserve `refresh_devices` behavior (mic → device combo populated; wav → file path + Browse).
- No change to `start_test` / `load_resources` / `run_inference` / the health pipeline. Behavior is identical; only widget parents/placement change.

## 6. Out of Scope

- No new settings or new functionality; no change to checks/inference/calibration.
- No restyling beyond what the relayout requires.
- Persisting settings to disk between runs (possible future; not now).

## 7. Testing

This is largely Tkinter layout, so automated coverage is limited:
- `main.py` must parse (`ast.parse`) and import cleanly.
- If any pure helper is extracted (e.g. a function returning the tab/field structure), unit-test it headlessly.
- **Manual GUI verification (owner):** main window shows the run bar (mic/wav + device/file selector), thresholds, dashboard, health profile + panel, log; the Settings button/menu opens the tabbed window; editing a setting there and starting a test behaves exactly as before; switching mic↔wav shows device dropdown vs file Browse correctly.
