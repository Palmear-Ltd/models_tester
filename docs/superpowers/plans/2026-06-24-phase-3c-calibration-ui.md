# Phase 3c — Calibration UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface calibration in the tester: a calibration column in the Signal Health panel, the ability to load a calibration profile (and show the active one), and a "Generate from recordings…" button that builds one via the 3a CLI.

**Architecture:** `report_rows` (UI-agnostic) gains a per-check calibration verdict drawn from `report.calibration_evaluation`; the panel renders it in a new column. `main.py` holds an optional loaded `CalibrationProfile`, rebuilds the pipeline via `pipeline_for_profile(profile_name, calibration_profile=...)`, shows the active profile, and wires Settings controls to load/generate one (reusing `calibrate.run`). No change to the calibration engine (3a/3b).

**Tech Stack:** Python 3.13, Tkinter/ttk, pytest. `calibrate.run` (soundfile/librosa) for generation.

> **Commit policy:** Owner commits manually. **Do NOT run git commit/add/push.** End at "tests pass."

> **References:** spec `docs/superpowers/specs/2026-06-23-...-design.md` §5 Phase 3; 3b design. Builds on `app/health/calibration.py` (`load_profile`), `app/health/calibration_eval.py` (`CalibrationEvaluation`, `MeasurementDeviation`), `app/health/config.py` (`pipeline_for_profile(name, calibration_profile=None)`), `calibrate.py` (`run(input, output, profile_id, sensor_info="")`).

> **Testing reality:** the panel-row formatting (`report_rows`) is unit-tested headlessly; the Tkinter wiring is verified by `ast.parse` + `import main` + the unchanged suite + a manual GUI checklist.

## Current State (key anchors)

- `app/health/reporting.py`: `report_rows(report)` returns `list[(check_id, name, status, detail)]` (4-tuples); `check_row` returns a 4-tuple.
- `main.py` `__init__` (lines 44-45): `self.profile_var = tk.StringVar(value="development")` then `self.health_pipeline = pipeline_for_profile(self.profile_var.get())`.
- Health Profile dropdown in `profile_frame` (lines ~246-258).
- Signal Health panel `self.health_tree` with `columns=("status", "detail")` (lines ~260-276); `_update_health_panel` (lines ~617-623) unpacks `for check_id, name, status, detail in report_rows(report)`.
- `_on_profile_change` (lines ~610-615) rebuilds `pipeline_for_profile(profile)`.
- `app/ui/settings_dialog.py` `_build_model_tab` has Model Path (row 0) and Scaler Path (row 1).
- `main.py` imports `from tkinter import ttk, filedialog, messagebox`.
- 100 tests pass.

---

## Task 1: Calibration verdict in report rows + panel column

**Files:** Modify `app/health/reporting.py`, `main.py`; Test `tests/health/test_reporting.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/health/test_reporting.py`:

```python
from app.health.calibration_eval import CalibrationEvaluation, MeasurementDeviation  # noqa: E402


def test_report_rows_blank_calibration_when_none():
    report = HealthReport(
        timestamp=0.0, window_id="w",
        check_results=[_result("T001", "Flatline", CheckStatus.PASS)],
    )
    rows = report_rows(report)
    assert rows[0][4] == ""  # 5th element = calibration verdict


def test_report_rows_marks_calibration_deviation_per_check():
    ev = CalibrationEvaluation(
        deviations=[MeasurementDeviation("T002", "rms", 9.9, 0.1, 0.3, CheckStatus.FAIL)],
        warn_count=0, fault_count=1,
    )
    report = HealthReport(
        timestamp=0.0, window_id="w",
        check_results=[
            _result("T001", "Flatline", CheckStatus.PASS),
            _result("T002", "Signal Energy", CheckStatus.PASS),
        ],
        calibration_evaluation=ev,
    )
    by_id = {r[0]: r for r in report_rows(report)}
    assert by_id["T002"][4] == "FAIL"
    assert by_id["T001"][4] == ""
```

(`HealthReport`, `CheckStatus`, `_result`, `report_rows` are already imported/defined at the top of `test_reporting.py`.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_reporting.py -q`
Expected: FAIL — `IndexError: tuple index out of range` (rows are 4-tuples; `[4]` doesn't exist).

- [ ] **Step 3: Extend `report_rows` in `app/health/reporting.py`**

Replace the `report_rows` function with:

```python
def report_rows(report: HealthReport) -> list[tuple[str, str, str, str, str]]:
    """One display row per check: (check_id, name, status, detail, calibration).

    `calibration` is the worst calibration verdict for that check from
    `report.calibration_evaluation` ("FAIL" beats "WARNING"), or "" when the
    check has no calibration deviation (or no profile is loaded).
    """
    cal_by_check: dict[str, str] = {}
    evaluation = report.calibration_evaluation
    if evaluation is not None:
        for d in evaluation.deviations:
            verdict = d.verdict.value  # "FAIL" or "WARNING"
            if verdict == "FAIL" or d.check_id not in cal_by_check:
                cal_by_check[d.check_id] = verdict
    rows = []
    for r in report.check_results:
        check_id, name, status, detail = check_row(r)
        rows.append((check_id, name, status, detail, cal_by_check.get(check_id, "")))
    return rows
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_reporting.py -q`
Expected: PASS. (Existing `test_report_rows_one_per_check` still passes — it indexes `[0]`/`[0][0]`.)

- [ ] **Step 5: Add the calibration column to the panel (`main.py` `_setup_ui`)**

Find:

```python
        self.health_tree = ttk.Treeview(
            health_frame, columns=("status", "detail"), show="tree headings", height=11
        )
        self.health_tree.heading("#0", text="Check")
        self.health_tree.heading("status", text="Status")
        self.health_tree.heading("detail", text="Detail")
        self.health_tree.column("#0", width=180, stretch=False)
        self.health_tree.column("status", width=80, anchor="center", stretch=False)
        self.health_tree.column("detail", width=240)
```

Replace with:

```python
        self.health_tree = ttk.Treeview(
            health_frame, columns=("status", "cal", "detail"), show="tree headings", height=11
        )
        self.health_tree.heading("#0", text="Check")
        self.health_tree.heading("status", text="Status")
        self.health_tree.heading("cal", text="Cal")
        self.health_tree.heading("detail", text="Detail")
        self.health_tree.column("#0", width=170, stretch=False)
        self.health_tree.column("status", width=70, anchor="center", stretch=False)
        self.health_tree.column("cal", width=60, anchor="center", stretch=False)
        self.health_tree.column("detail", width=210)
```

- [ ] **Step 6: Update `_update_health_panel` (`main.py`) to render the calibration column**

Find:

```python
    def _update_health_panel(self, report):
        tree = self.health_tree
        tree.delete(*tree.get_children())
        for check_id, name, status, detail in report_rows(report):
            tree.insert(
                "", "end", text=f"{check_id}  {name}", values=(status, detail), tags=(status,)
            )
```

Replace with:

```python
    def _update_health_panel(self, report):
        tree = self.health_tree
        tree.delete(*tree.get_children())
        for check_id, name, status, detail, cal in report_rows(report):
            tree.insert(
                "", "end", text=f"{check_id}  {name}",
                values=(status, cal, detail), tags=(status,)
            )
```

- [ ] **Step 7: Verify**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('parses OK')"` → `parses OK`
Run: `.venv/bin/python -m pytest tests/ -q` → 102 passed (100 + 2 new).

---

## Task 2: Load a calibration profile + show the active one

**Files:** Modify `main.py`

- [ ] **Step 1: Add calibration state and rebuild the initial pipeline with it**

In `main.py` `__init__`, find:

```python
        self.profile_var = tk.StringVar(value="development")
        self.health_pipeline = pipeline_for_profile(self.profile_var.get())
```

Replace with:

```python
        self.profile_var = tk.StringVar(value="development")
        self.calibration_profile = None
        self.calibration_profile_path_var = tk.StringVar(value="")
        self.health_pipeline = pipeline_for_profile(
            self.profile_var.get(), calibration_profile=self.calibration_profile
        )
```

- [ ] **Step 2: Add the rebuild + load helpers and Browse handler**

In `main.py`, add these methods immediately before `def _on_profile_change(self, event=None):`:

```python
    def _rebuild_health_pipeline(self):
        self.health_pipeline = pipeline_for_profile(
            self.profile_var.get(), calibration_profile=self.calibration_profile
        )
        self._last_health_state = None

    def _load_calibration_profile(self, path):
        from app.health.calibration import load_profile

        try:
            self.calibration_profile = load_profile(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration profile:\n{e}")
            return
        self.calibration_profile_path_var.set(path)
        self._rebuild_health_pipeline()
        p = self.calibration_profile
        self.cal_label.configure(text=f"Calibration: {p.profile_id} ({p.window_count} win)")
        self.log(f"Loaded calibration profile '{p.profile_id}' ({p.window_count} windows)")

    def browse_calibration_profile(self):
        path = filedialog.askopenfilename(
            title="Select calibration profile", filetypes=[("Calibration profile", "*.json")]
        )
        if path:
            self._load_calibration_profile(path)

```

- [ ] **Step 3: Use the rebuild helper in `_on_profile_change`**

Replace the body of `_on_profile_change`:

```python
    def _on_profile_change(self, event=None):
        profile = self.profile_var.get()
        self.health_pipeline = pipeline_for_profile(profile)
        self._last_health_state = None
        count = len(self.health_pipeline.manager.checks)
        self.log(f"Health profile: {profile} ({count} checks active)")
```

with:

```python
    def _on_profile_change(self, event=None):
        self._rebuild_health_pipeline()
        profile = self.profile_var.get()
        count = len(self.health_pipeline.manager.checks)
        self.log(f"Health profile: {profile} ({count} checks active)")
```

- [ ] **Step 4: Add the active-profile label next to the Health Profile dropdown**

In `main.py` `_setup_ui`, find:

```python
        self.profile_combo.pack(side="left", padx=5)
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_profile_change)
```

Add immediately AFTER it:

```python
        self.cal_label = ttk.Label(profile_frame, text="Calibration: none")
        self.cal_label.pack(side="left", padx=12)
```

- [ ] **Step 5: Verify**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('parses OK')"` → `parses OK`
Run: `.venv/bin/python -c "import main; print('import OK')"` → `import OK`
Run: `.venv/bin/python -m pytest tests/ -q` → 102 passed.

---

## Task 3: Settings — load/generate calibration profile

**Files:** Modify `main.py`, `app/ui/settings_dialog.py`

- [ ] **Step 1: Add the `simpledialog` import**

In `main.py`, find:

```python
from tkinter import ttk, filedialog, messagebox
```

Replace with:

```python
from tkinter import ttk, filedialog, messagebox, simpledialog
```

- [ ] **Step 2: Add the generate handler**

In `main.py`, add this method immediately before `def browse_calibration_profile(self):`:

```python
    def generate_calibration_profile(self):
        import calibrate

        folder = filedialog.askdirectory(title="Select folder of HEALTHY recordings")
        if not folder:
            return
        out = filedialog.asksaveasfilename(
            title="Save calibration profile", defaultextension=".json",
            filetypes=[("Calibration profile", "*.json")],
        )
        if not out:
            return
        profile_id = simpledialog.askstring(
            "Calibration profile id", "Profile id:", initialvalue="piezo"
        )
        if not profile_id:
            return
        self.log(f"Generating calibration profile from {folder} ...")
        self.root.update_idletasks()
        try:
            profile = calibrate.run(folder, out, profile_id=profile_id)
        except Exception as e:
            messagebox.showerror("Error", f"Calibration failed:\n{e}")
            return
        self.log(f"Calibration profile saved: {out} ({profile.window_count} windows)")
        self._load_calibration_profile(out)

```

- [ ] **Step 3: Add the calibration controls to the Settings "Model & Scaler" tab**

In `app/ui/settings_dialog.py` `_build_model_tab`, find:

```python
        ttk.Label(tab, text="Scaler Path (.json/.npz):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.scaler_path_var).grid(row=1, column=1, padx=5, sticky="ew")
        ttk.Button(tab, text="Browse", command=app.load_scaler_dialog).grid(row=1, column=2)
```

Add immediately AFTER it:

```python

        ttk.Label(tab, text="Calibration Profile (.json):").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.calibration_profile_path_var).grid(row=2, column=1, padx=5, sticky="ew")
        ttk.Button(tab, text="Browse", command=app.browse_calibration_profile).grid(row=2, column=2)
        ttk.Button(
            tab, text="Generate from recordings…", command=app.generate_calibration_profile
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=5)
```

- [ ] **Step 4: Verify parse + import + suite**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); ast.parse(open('app/ui/settings_dialog.py').read()); print('parses OK')"` → `parses OK`
Run: `.venv/bin/python -c "import main; print('import OK')"` → `import OK`
Run: `.venv/bin/python -m pytest tests/ -q` → 102 passed.

- [ ] **Step 5: Headless wiring check (no display)**

```bash
.venv/bin/python -c "
import inspect, main
for m in ('_rebuild_health_pipeline','_load_calibration_profile','browse_calibration_profile','generate_calibration_profile'):
    assert callable(getattr(main.ModelsTesterApp, m)), m
print('handlers present')
import calibrate
assert hasattr(calibrate, 'run')
print('calibrate.run available')
"
```
Expected: `handlers present` then `calibrate.run available`.

- [ ] **Step 6: Manual GUI verification (owner)**

Launch `./.venv/bin/python launcher.py`. In **⚙ Settings ▸ Model & Scaler**: **Browse** a calibration JSON (e.g. one made earlier with `calibrate.py`) → the main window shows "Calibration: <id> (<n> win)" and a log line; run a mic/wav test → the **Cal** column in the Signal Health panel shows FAIL/WARNING on any deviating checks, and the overall indicator reflects calibration escalation. **Generate from recordings…** → pick a folder of healthy WAVs, a save path, and an id → it builds, logs the window count, and auto-loads the new profile. With no profile loaded, behavior is unchanged and the Cal column stays blank.

---

## Phase 3c Done — Phase 3 complete

Calibration is now fully GUI-driven: load or generate a profile, see the active one, and watch sensor-relative deviations in the panel's Cal column influence the verdict. This completes Phase 3 (3a generation + 3b evaluation + 3c UI). Hand back to the owner for review, manual test, and commit. Next: **Phase 4 — Stability checks + Runtime Monitoring + timeline plot.**

---

## Self-Review

- **Spec coverage:** calibration column in the panel — Task 1; load a profile + show the active one — Task 2; "Generate profile" action — Task 3 (reuses `calibrate.run`); pipeline rebuilt with the loaded profile via `pipeline_for_profile(name, calibration_profile=...)` — Task 2. With no profile, behavior unchanged (Cal blank, pipeline built with `calibration_profile=None`).
- **Placeholder scan:** no TBD/TODO; full code each step; concrete commands. The synchronous generate may briefly block on large sets — acceptable for a one-time calibration, noted in the manual step.
- **Type consistency:** `report_rows(report) -> list[(check_id, name, status, detail, cal)]` (now 5-tuple) consumed by `_update_health_panel`'s 5-way unpack and the 3-value Treeview `values=(status, cal, detail)` against `columns=("status","cal","detail")`; `check_row` unchanged (4-tuple). `_rebuild_health_pipeline`/`_load_calibration_profile`/`browse_calibration_profile`/`generate_calibration_profile` defined in `main.py` and referenced by `_on_profile_change`, the Settings tab, and each other consistently. `self.calibration_profile_path_var`/`self.cal_label`/`self.calibration_profile` created in `__init__`/`_setup_ui` before use. `pipeline_for_profile(name, calibration_profile=None)` and `calibrate.run(input, output, profile_id, sensor_info="")` match their definitions.
