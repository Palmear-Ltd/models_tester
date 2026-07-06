# Phase 2c — Configuration & Profile System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a centralized configuration system (spec §4.7) — a check registry, an enable/disable + mandatory + per-check-params hierarchy, and named profiles (development / production / diagnostic / minimal) — plus a profile selector in the tester that rebuilds the active check set live.

**Architecture:** A new pure `app/health/config.py` holds a `REGISTRY` of all checks (id → class, config-group, mandatory flag), a `HealthConfig` (category enables + per-check `CheckConfig`) with an `is_active(spec)` resolution rule (mandatory always on; else category-enabled AND check-enabled), and `build_manager`/`build_pipeline`/`pipeline_for_profile`. `main.py` gains a profile dropdown that swaps `self.health_pipeline` to `pipeline_for_profile(selected)`; the Phase 2b panel then reflects the active checks. No change to check logic, fusion, or inference.

**Tech Stack:** Python 3.13, stdlib only for config (dataclasses), Tkinter (ttk.Combobox), pytest. Run via `.venv/bin/python -m pytest`.

> **Commit policy:** Owner commits/pushes **manually** after review. **Do NOT run `git commit`/`git add`/`git push`.** Each task ends at "tests pass."

> **References:** spec `arch_update.md` §4.7 (configuration hierarchy, mandatory checks, profiles §4.7.7). Design spec §5 Phase 2. Builds on Phase 2a/2b (11 checks; `default_pipeline`).

> **Scope note:** Calibration-derived thresholds are §4.7.5's "calibration" source — deferred to Phase 3. Phase 2c uses MANUAL params only. Profiles are code-defined; file/YAML loading is a trivial future add and out of scope here. `diagnostic` differs from `development` only in logging verbosity (a Phase 6 concern), so in 2c it has the same check set.

---

## Current State (end of Phase 2b)

- `app/health/checks/time_domain.py`: `FlatlineCheck` T001, `SignalEnergyCheck` T002, `PeakAmplitudeCheck` T003, `ClippingCheck` T004, `CrestFactorCheck` T005, `DCOffsetCheck` T006, `ZeroCrossingRateCheck` T007. Each takes thresholds via `__init__` kwargs.
- `app/health/checks/frequency_domain.py`: `SpectralShapeCheck` F001, `SpectralFlatnessCheck` F002, `BandEnergyDistributionCheck` F003, `ElectricalHumCheck` F004.
- `app/health/manager.py`: `SignalCheckManager` — `register(check)`, `.checks` property.
- `app/health/pipeline.py`: `HealthAnalysisPipeline(manager=None)`; `.manager` attribute holds the manager.
- `app/health/defaults.py`: `default_manager()`/`default_pipeline()` register all 11. (Kept as-is; still covered by `tests/health/test_defaults.py`.)
- `main.py`: imports `from app.health.defaults import default_pipeline` (line 20); `self.health_pipeline = default_pipeline()` (line 43). A "Signal Health Detail" Treeview panel sits in `_setup_ui` just before the `# --- Control Area (Left) ---` comment.
- 68 tests pass.

## File Structure

**Create:**
- `app/health/config.py` — `CheckSpec`, `REGISTRY`, `CheckConfig`, `HealthConfig`, `build_manager`, `build_pipeline`, `PROFILES`, `config_for_profile`, `pipeline_for_profile`. Pure (no NumPy/UI).
- `tests/health/test_config.py`.

**Modify:**
- `main.py` — add `self.profile_var`, switch the pipeline import/construction to `pipeline_for_profile`, add a profile `ttk.Combobox` and an `_on_profile_change` handler.

---

## Task 1: Config model, registry & builder

**Files:**
- Create: `app/health/config.py`
- Test: `tests/health/test_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/health/test_config.py`:

```python
import pytest

from app.health.checks.time_domain import SignalEnergyCheck
from app.health.config import (
    REGISTRY,
    CheckConfig,
    HealthConfig,
    build_manager,
)

ALL_IDS = {
    "T001", "T002", "T003", "T004", "T005", "T006", "T007",
    "F001", "F002", "F003", "F004",
}


def _ids(manager):
    return {c.check_id for c in manager.checks}


def test_registry_has_all_eleven_checks():
    assert len(REGISTRY) == 11
    assert {s.check_id for s in REGISTRY} == ALL_IDS


def test_default_config_activates_everything():
    assert _ids(build_manager(HealthConfig())) == ALL_IDS


def test_mandatory_checks_run_even_when_categories_disabled():
    cfg = HealthConfig(categories={"time_domain": False, "frequency_domain": False})
    # Only the mandatory checks (Flatline, Signal Energy, Clipping) survive.
    assert _ids(build_manager(cfg)) == {"T001", "T002", "T004"}


def test_disabling_a_category_drops_its_non_mandatory_checks():
    cfg = HealthConfig(categories={"time_domain": True, "frequency_domain": False})
    ids = _ids(build_manager(cfg))
    assert "F001" not in ids and "F004" not in ids
    assert {"T001", "T002", "T003", "T004", "T005", "T006", "T007"} <= ids


def test_disabling_an_individual_check():
    cfg = HealthConfig(checks={"F004": CheckConfig(enabled=False)})
    ids = _ids(build_manager(cfg))
    assert "F004" not in ids
    assert len(ids) == 10


def test_check_params_override_constructor_defaults():
    cfg = HealthConfig(checks={"T002": CheckConfig(params={"min_rms_fault": 0.5})})
    manager = build_manager(cfg)
    energy = next(c for c in manager.checks if c.check_id == "T002")
    assert isinstance(energy, SignalEnergyCheck)
    assert energy.min_rms_fault == 0.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.health.config'`.

- [ ] **Step 3: Implement `config.py`**

Create `app/health/config.py`:

```python
"""Centralized configuration for the Audio Signal Health Monitoring subsystem
(spec §4.7).

A `REGISTRY` lists every available check with its config group and whether it is
mandatory. A `HealthConfig` enables/disables checks at the category and individual
level and supplies per-check parameter overrides; `build_manager` resolves it into
a populated `SignalCheckManager`. Pure stdlib — no NumPy/UI imports.

Threshold *params* here are manual overrides; calibration-derived thresholds are a
Phase 3 concern.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.health.checks.base import SignalHealthCheck
from app.health.checks.frequency_domain import (
    BandEnergyDistributionCheck,
    ElectricalHumCheck,
    SpectralFlatnessCheck,
    SpectralShapeCheck,
)
from app.health.checks.time_domain import (
    ClippingCheck,
    CrestFactorCheck,
    DCOffsetCheck,
    FlatlineCheck,
    PeakAmplitudeCheck,
    SignalEnergyCheck,
    ZeroCrossingRateCheck,
)
from app.health.manager import SignalCheckManager
from app.health.pipeline import HealthAnalysisPipeline


@dataclass(frozen=True)
class CheckSpec:
    """Static metadata describing one available check."""

    check_id: str
    factory: type  # a SignalHealthCheck subclass
    category: str  # config group: "time_domain" | "frequency_domain"
    mandatory_default: bool = False


# Canonical registry of every available check. Mandatory checks (spec §4.7.6)
# protect against catastrophic acquisition failures and cannot be disabled.
REGISTRY: list[CheckSpec] = [
    CheckSpec("T001", FlatlineCheck, "time_domain", mandatory_default=True),
    CheckSpec("T002", SignalEnergyCheck, "time_domain", mandatory_default=True),
    CheckSpec("T003", PeakAmplitudeCheck, "time_domain"),
    CheckSpec("T004", ClippingCheck, "time_domain", mandatory_default=True),
    CheckSpec("T005", CrestFactorCheck, "time_domain"),
    CheckSpec("T006", DCOffsetCheck, "time_domain"),
    CheckSpec("T007", ZeroCrossingRateCheck, "time_domain"),
    CheckSpec("F001", SpectralShapeCheck, "frequency_domain"),
    CheckSpec("F002", SpectralFlatnessCheck, "frequency_domain"),
    CheckSpec("F003", BandEnergyDistributionCheck, "frequency_domain"),
    CheckSpec("F004", ElectricalHumCheck, "frequency_domain"),
]


@dataclass
class CheckConfig:
    """Per-check configuration overlay."""

    enabled: bool = True
    mandatory: bool = False
    params: dict = field(default_factory=dict)


@dataclass
class HealthConfig:
    """Resolved configuration controlling which checks run and how."""

    profile: str = "development"
    categories: dict = field(default_factory=dict)  # group name -> enabled
    checks: dict = field(default_factory=dict)  # check_id -> CheckConfig

    def is_active(self, spec: CheckSpec) -> bool:
        cc: Optional[CheckConfig] = self.checks.get(spec.check_id)
        mandatory = spec.mandatory_default or (cc.mandatory if cc else False)
        if mandatory:
            return True
        if not self.categories.get(spec.category, True):
            return False
        return cc.enabled if cc else True


def build_manager(config: HealthConfig) -> SignalCheckManager:
    """Register every active check (in registry order) into a fresh manager."""
    manager = SignalCheckManager()
    for spec in REGISTRY:
        if config.is_active(spec):
            cc = config.checks.get(spec.check_id)
            params = cc.params if cc else {}
            manager.register(spec.factory(**params))
    return manager


def build_pipeline(config: HealthConfig) -> HealthAnalysisPipeline:
    return HealthAnalysisPipeline(manager=build_manager(config))
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_config.py -q`
Expected: PASS (6 passed).

---

## Task 2: Named profiles

**Files:**
- Modify: `app/health/config.py` (append profiles)
- Test: `tests/health/test_config.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/health/test_config.py`:

```python
import numpy as np  # noqa: E402

from app.health.config import (  # noqa: E402
    PROFILES,
    config_for_profile,
    pipeline_for_profile,
)
from app.health.models import AudioWindow, HealthState  # noqa: E402


def _profile_ids(name):
    return {c.check_id for c in build_manager(config_for_profile(name)).checks}


def test_profiles_listed():
    assert set(PROFILES) == {"development", "production", "diagnostic", "minimal"}


def test_minimal_profile_is_mandatory_only():
    assert _profile_ids("minimal") == {"T001", "T002", "T004"}


def test_production_profile_excludes_electrical_hum():
    ids = _profile_ids("production")
    assert "F004" not in ids
    assert len(ids) == 10


def test_development_and_diagnostic_have_all_eleven():
    assert len(_profile_ids("development")) == 11
    assert len(_profile_ids("diagnostic")) == 11


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        config_for_profile("bogus")


def test_pipeline_for_profile_minimal_faults_on_silence():
    report = pipeline_for_profile("minimal").analyze(
        AudioWindow(samples=np.zeros(110250, dtype=np.float32), sample_rate=44100)
    )
    assert report.final_state is HealthState.FAULT
    assert len(report.check_results) == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/health/test_config.py -q`
Expected: FAIL with `ImportError: cannot import name 'PROFILES'`.

- [ ] **Step 3: Append the profiles to `config.py`**

Append to `app/health/config.py`:

```python


PROFILES = ("development", "production", "diagnostic", "minimal")


def _profile_configs() -> dict:
    return {
        # All checks enabled (categories default to enabled when unset).
        "development": HealthConfig(profile="development"),
        # Same check set as development; differs only in logging verbosity (Phase 6).
        "diagnostic": HealthConfig(profile="diagnostic"),
        # Recommended set: everything except environment-dependent electrical hum.
        "production": HealthConfig(
            profile="production", checks={"F004": CheckConfig(enabled=False)}
        ),
        # Resource-constrained: mandatory checks only (categories disabled).
        "minimal": HealthConfig(
            profile="minimal",
            categories={"time_domain": False, "frequency_domain": False},
        ),
    }


def config_for_profile(name: str) -> HealthConfig:
    configs = _profile_configs()
    if name not in configs:
        raise KeyError(f"Unknown health profile: {name!r} (known: {sorted(configs)})")
    return configs[name]


def pipeline_for_profile(name: str) -> HealthAnalysisPipeline:
    return build_pipeline(config_for_profile(name))
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/health/test_config.py -q`
Expected: PASS (12 passed).

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (80 total: 68 prior + 12 new).

---

## Task 3: Profile selector in the tester

**Files:**
- Modify: `main.py`

> Locate anchors by quoted TEXT, not line numbers. Verify each before editing.

- [ ] **Step 1: Switch the pipeline import**

In `main.py`, find:

```python
from app.health.defaults import default_pipeline
from app.health.models import AudioWindow, HealthState
```

Replace the first line so the two lines become:

```python
from app.health.config import PROFILES, pipeline_for_profile
from app.health.models import AudioWindow, HealthState
```

- [ ] **Step 2: Build the initial pipeline from a profile + store the profile var**

In `main.py` `__init__`, find:

```python
        self.health_pipeline = default_pipeline()
        self.latest_health_report = None
        self._last_health_state = None
```

Replace with:

```python
        self.profile_var = tk.StringVar(value="development")
        self.health_pipeline = pipeline_for_profile(self.profile_var.get())
        self.latest_health_report = None
        self._last_health_state = None
```

- [ ] **Step 3: Add the profile dropdown above the health panel**

In `main.py` `_setup_ui`, find the panel comment:

```python
        # --- Signal Health Detail (per-check breakdown) ---
```

Insert immediately BEFORE it (8-space indentation):

```python
        # Health monitoring profile selector
        profile_frame = ttk.Frame(left_frame)
        profile_frame.pack(fill="x", padx=10, pady=(5, 0))
        ttk.Label(profile_frame, text="Health Profile:").pack(side="left")
        self.profile_combo = ttk.Combobox(
            profile_frame,
            textvariable=self.profile_var,
            values=list(PROFILES),
            state="readonly",
            width=14,
        )
        self.profile_combo.pack(side="left", padx=5)
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_profile_change)

```

- [ ] **Step 4: Add the `_on_profile_change` handler**

In `main.py`, add this method immediately BEFORE the `def _update_health_panel(self, report):` method (4-space indent for `def`, blank line after):

```python
    def _on_profile_change(self, event=None):
        profile = self.profile_var.get()
        self.health_pipeline = pipeline_for_profile(profile)
        self._last_health_state = None
        count = len(self.health_pipeline.manager.checks)
        self.log(f"Health profile: {profile} ({count} checks active)")

```

- [ ] **Step 5: Verify parse + additive diff**

Run: `.venv/bin/python -c "import ast; ast.parse(open('main.py').read()); print('main.py parses OK')"`
Expected: `main.py parses OK`

Run: `git diff main.py | grep '^-' | grep -v '^---'`
Expected: only the single replaced import line (`from app.health.defaults import default_pipeline`) and the replaced `self.health_pipeline = default_pipeline()` line show as removed; plus the pre-existing Phase 2a `sample_rate=44100` line (already in the working tree). No inference/feature/model logic removed.

- [ ] **Step 6: Headless check of profile switching**

```bash
.venv/bin/python -c "
from app.health.config import pipeline_for_profile, PROFILES
for p in PROFILES:
    n = len(pipeline_for_profile(p).manager.checks)
    print(p, n)
assert len(pipeline_for_profile('minimal').manager.checks) == 3
assert len(pipeline_for_profile('development').manager.checks) == 11
print('profiles OK')
"
```
Expected: prints each profile with its count (development 11, production 10, diagnostic 11, minimal 3) then `profiles OK`.

- [ ] **Step 7: Confirm the suite still passes**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (80 total).

- [ ] **Step 8: Manual GUI verification (owner)**

(Requires full deps.) Launch `./.venv/bin/python launcher.py`; the new **"Health Profile"** dropdown sits above the Signal Health Detail panel. Start a test, switch the profile to **Minimal** → the panel should drop to 3 rows (Flatline, Signal Energy, Clipping) and the log notes "Health profile: minimal (3 checks active)"; switch to **Development** → 11 rows. Classification behavior unchanged.

---

## Phase 2c Done

Signal health is now configuration-driven: a registry + `HealthConfig` hierarchy with mandatory enforcement, four named profiles, and a live profile selector in the tester. This completes Phase 2 (frequency checks + panel + config). Hand back to the owner for review, manual test, and commit. Phase 3 introduces calibration (Layer 2): generating a `CalibrationProfile` from healthy recordings and switching thresholds from manual to calibration-derived.

---

## Self-Review

- **Spec coverage (§4.7):** configuration hierarchy global/category/check (§4.7.1–4.7.4) — `HealthConfig.categories` + per-check `CheckConfig`, resolved by `is_active`; mandatory checks (§4.7.6) — `mandatory_default` on T001/T002/T004, always active; configuration profiles (§4.7.7) — `development`/`production`/`diagnostic`/`minimal`. Manual threshold params (§4.7.5 manual source) via `CheckConfig.params`; calibration source deferred to Phase 3 (noted). File/YAML storage (§4.7.8) intentionally out of scope (profiles are code-defined).
- **Placeholder scan:** no TBD/TODO; every step has full code and exact commands.
- **Type consistency:** `CheckSpec(check_id, factory, category, mandatory_default)`; `HealthConfig.is_active(spec)`; `build_manager(config) -> SignalCheckManager`; `build_pipeline`/`pipeline_for_profile -> HealthAnalysisPipeline` (whose `.manager.checks` the UI counts); `config_for_profile(name) -> HealthConfig`; `PROFILES` tuple matches `_profile_configs()` keys. `main.py` uses `pipeline_for_profile` (Task 3) defined in Task 2 and `PROFILES` for the dropdown.
- **No regression:** `default_pipeline`/`defaults.py` untouched (its tests stay green); `main.py` default profile "development" = all 11 checks = prior behavior. UI change is additive (dropdown + handler); the panel refresh path is unchanged.
