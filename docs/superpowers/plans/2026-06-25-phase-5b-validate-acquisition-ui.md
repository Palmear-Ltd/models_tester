# Phase 5b — Validate Acquisition UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Validate Acquisition" button that assesses ~20 s of the live health stream during a running test and shows a PASS / WARNING / FAIL `StartupResult`.

**Architecture:** The tester already runs the health pipeline every window during a test (`handle_audio_chunk` → `analyze` → `latest_health_report`). The button flips a `_validating` flag; `handle_audio_chunk` then collects the next ~40 reports and, once gathered, calls the headless `run_validation` (5a) and shows the result in a message box. No audio-thread changes; no effect on normal testing.

**Tech Stack:** Tkinter (messagebox), the `app/health/startup` engine; pytest (regression only).

> **Commit policy:** Owner commits manually. **Do NOT run git commit/add/push.** End at verification.

> **Design note (refinement of the brainstormed "self-starting capture"):** validation **rides the live test** rather than starting its own bounded acquisition — much simpler and reuses the running pipeline (the tester is already monitoring health live). Therefore it **requires a test to be running**; if idle, the button explains how to use it. Reference spec: `docs/superpowers/specs/2026-06-25-phase-5-startup-validation-design.md` (§4).

> **Testing reality:** the validation engine (`run_validation`) is already unit-tested in 5a; this is Tkinter wiring — verified by `ast.parse` + `import main` + the unchanged 128-test suite + a manual GUI check.

## Current State (anchors)

- `main.py` top: `SAMPLE_RATE = 44100` constant; health imports include `from app.health.reporting import report_rows` and `from app.health.monitoring import RuntimeMonitor`; `from tkinter import ttk, filedialog, messagebox, simpledialog`.
- `__init__`: `self.runtime_monitor = RuntimeMonitor()` (line ~53). `self.calibration_profile` exists (None unless loaded). `self.is_running` flag exists.
- Run-bar controls in `_setup_ui`:
  ```python
        self.start_btn = ttk.Button(controls_frame, text="START TEST", command=self.toggle_test, width=18, style="Primary.TButton")
        self.start_btn.pack(side="left")
        ttk.Button(controls_frame, text="⚙ Settings", command=self.open_prep_settings).pack(side="left", padx=8)
  ```
- `handle_audio_chunk` health block:
  ```python
        # Audio signal health monitoring — additive; never blocks or alters inference.
        try:
            window = AudioWindow(samples=self.session_buffer, sample_rate=SAMPLE_RATE)
            self.latest_health_report = self.health_pipeline.analyze(window)
            self._update_health_indicator(self.latest_health_report)
        except Exception as e:
            self.log(f"Health monitoring error: {e}")
  ```
- 128 tests pass.

## File Structure
- Modify: `main.py` only.

---

## Task 1: Validate Acquisition button + live collection

**Files:** Modify `main.py`

- [ ] **Step 1: Import the validator and add the window constant**

In `main.py`, find:

```python
from app.health.monitoring import RuntimeMonitor
```

Add immediately AFTER it:

```python
from app.health.startup import StartupDecision, run_validation
```

Then find:

```python
SAMPLE_RATE = 44100  # acquisition sample rate (Hz); the whole pipeline runs at this rate
```

Add immediately AFTER it:

```python
VALIDATION_WINDOWS = 40  # ~20 s of validation at 0.5 s per window
```

- [ ] **Step 2: Add validation state in `__init__`**

In `main.py` `__init__`, find:

```python
        self.runtime_monitor = RuntimeMonitor()
```

Add immediately AFTER it:

```python
        self._validating = False
        self._validation_reports = []
```

- [ ] **Step 3: Add the Validate button to the run bar**

In `main.py` `_setup_ui`, find:

```python
        ttk.Button(controls_frame, text="⚙ Settings", command=self.open_prep_settings).pack(side="left", padx=8)
```

Add immediately AFTER it:

```python
        ttk.Button(controls_frame, text="Validate Acquisition", command=self.validate_acquisition).pack(side="left", padx=8)
```

- [ ] **Step 4: Collect reports during validation in `handle_audio_chunk`**

In `main.py` `handle_audio_chunk`, find:

```python
            self.latest_health_report = self.health_pipeline.analyze(window)
            self._update_health_indicator(self.latest_health_report)
        except Exception as e:
            self.log(f"Health monitoring error: {e}")
```

Replace with:

```python
            self.latest_health_report = self.health_pipeline.analyze(window)
            self._update_health_indicator(self.latest_health_report)
            if self._validating and self.latest_health_report is not None:
                self._validation_reports.append(self.latest_health_report)
                if len(self._validation_reports) >= VALIDATION_WINDOWS:
                    self._validating = False
                    self._show_validation_result()
        except Exception as e:
            self.log(f"Health monitoring error: {e}")
```

- [ ] **Step 5: Add the handler + result methods**

In `main.py`, add these two methods immediately before `def _update_health_indicator(self, report):`:

```python
    def validate_acquisition(self):
        if not self.is_running:
            messagebox.showinfo(
                "Validate Acquisition",
                "Start a test first, then click Validate to assess ~20 s of the live signal.",
            )
            return
        self._validation_reports = []
        self._validating = True
        self.log("Validating acquisition over the next ~20 s ...")

    def _show_validation_result(self):
        result = run_validation(
            self._validation_reports,
            sample_rate=SAMPLE_RATE,
            input_ready=True,
            calibration_loaded=self.calibration_profile is not None,
        )
        self.log(f"[validation] {result.summary}")
        show = {
            StartupDecision.PASS: messagebox.showinfo,
            StartupDecision.WARNING: messagebox.showwarning,
            StartupDecision.FAIL: messagebox.showerror,
        }[result.decision]
        show("Acquisition Validation", f"{result.decision.value}\n\n{result.summary}")

```

- [ ] **Step 6: Verify**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('parses OK')"` → `parses OK`
Run: `.venv/bin/python -c "import main; print('import OK')"` → `import OK` (keras FutureWarning is fine)
Run: `.venv/bin/python -m pytest tests/ -q` → 128 passed (UI-only change).
Run: `git diff main.py | grep '^-' | grep -v '^---'` → only the health-block anchor lines should show as replaced (the two `self.latest_health_report`/`_update_health_indicator` lines and the `except` line) — confirm no inference/health logic removed.

- [ ] **Step 7: Headless wiring check**

```bash
.venv/bin/python -c "
import main
for m in ('validate_acquisition', '_show_validation_result'):
    assert callable(getattr(main.ModelsTesterApp, m)), m
print('handlers present; VALIDATION_WINDOWS =', main.VALIDATION_WINDOWS)
from app.health.startup import run_validation, StartupDecision
print('validator wired OK')
"
```
Expected: `handlers present; VALIDATION_WINDOWS = 40` then `validator wired OK`.

- [ ] **Step 8: Manual GUI verification (owner)**

Launch `./.venv/bin/python launcher.py`. Click **Validate Acquisition** while idle → an info box explains to start a test first. **Start a test**, then click **Validate Acquisition** → after ~20 s a result box appears: a clean live signal → **PASS** (info), a disconnected/faulty input → **FAIL** (error) with the window tally + top failing checks, a healthy signal with no calibration loaded → **PASS/WARNING** with a "No calibration profile loaded" note. The log shows `[validation] …`. Normal testing and classification are unaffected.

---

## Phase 5b Done — Phase 5 complete

The tester can now validate the acquisition chain on demand: press the button during a test, and after ~20 s get a PASS/WARNING/FAIL verdict over the live health stream. This completes Phase 5. Hand back to the owner for review, manual test, and commit. Next: **Phase 6 — anomaly detection + confidence/reporting polish.**

---

## Self-Review

- **Spec coverage (§4):** "Validate Acquisition" button — Step 3; bounded ~20 s assessment of the live signal feeding the pipeline — Steps 2/4 collect `VALIDATION_WINDOWS` reports; calls `run_validation(..., input_ready=True, calibration_loaded=self.calibration_profile is not None)` and shows the `StartupResult` color-coded (info/warning/error) with tally + top checks — Step 5; never blocks normal START TEST — it only collects when `_validating` and resets itself. Refinement (rides the live test, requires a running test) is documented at the top.
- **Placeholder scan:** no TBD/TODO; full code each step; concrete commands and expected output.
- **Type consistency:** `run_validation(reports, *, sample_rate, input_ready, calibration_loaded)` and `StartupDecision` (PASS/WARNING/FAIL) imported from `app.health.startup` (defined in 5a); `self._validating`/`self._validation_reports` created in `__init__`, set in `validate_acquisition`, consumed in `handle_audio_chunk` and `_show_validation_result`; `VALIDATION_WINDOWS` defined once at module level. `messagebox` already imported.
