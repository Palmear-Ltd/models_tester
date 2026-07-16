import pytest

from app.health.checks.time_domain import SignalEnergyCheck
from app.health.config import (
    REGISTRY,
    CheckConfig,
    HealthConfig,
    build_manager,
)

ALL_IDS = {
    "T001", "T002", "T003", "T004", "T005", "T006", "T007", "T008", "T009",
    "F001", "F002", "F003", "F004",
    "S001", "S002", "S003", "S004",
}


def _ids(manager):
    return {c.check_id for c in manager.checks}


def test_registry_has_all_eleven_checks():
    assert len(REGISTRY) == 17
    assert {s.check_id for s in REGISTRY} == ALL_IDS


def test_default_config_activates_everything():
    assert _ids(build_manager(HealthConfig())) == ALL_IDS


def test_mandatory_checks_run_even_when_categories_disabled():
    cfg = HealthConfig(categories={"time_domain": False, "frequency_domain": False, "stability": False})
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
    assert len(ids) == 16


def test_check_params_override_constructor_defaults():
    cfg = HealthConfig(checks={"T002": CheckConfig(params={"min_rms_fault": 0.5})})
    manager = build_manager(cfg)
    energy = next(c for c in manager.checks if c.check_id == "T002")
    assert isinstance(energy, SignalEnergyCheck)
    assert energy.min_rms_fault == 0.5


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
    assert len(ids) == 16


def test_development_and_diagnostic_have_all_eleven():
    assert len(_profile_ids("development")) == 17
    assert len(_profile_ids("diagnostic")) == 17


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        config_for_profile("bogus")


def test_pipeline_for_profile_minimal_faults_on_silence():
    report = pipeline_for_profile("minimal").analyze(
        AudioWindow(samples=np.zeros(110250, dtype=np.float32), sample_rate=44100)
    )
    assert report.final_state is HealthState.FAULT
    assert len(report.check_results) == 3


# ---------------------------------------------------------------------------
# Persisted check-threshold overrides (app/health/check_thresholds.json)
# ---------------------------------------------------------------------------

from app.health.checks.time_domain import ClickTransientCheck  # noqa: E402
from app.health.config import load_check_thresholds  # noqa: E402


def test_load_check_thresholds_reads_shipped_repo_file():
    # No path given -> resolves the repo-committed app/health/check_thresholds.json.
    thresholds = load_check_thresholds()
    assert thresholds["T009"] == {"warn_count": 15, "fault_count": 30}


def test_load_check_thresholds_falls_back_when_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    thresholds = load_check_thresholds(str(missing))
    # Never raises; falls back to the shipped-in-code defaults.
    assert thresholds["T009"] == {"warn_count": 15, "fault_count": 30}


def test_load_check_thresholds_falls_back_on_malformed_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    thresholds = load_check_thresholds(str(bad))
    assert thresholds["T009"] == {"warn_count": 15, "fault_count": 30}


def test_load_check_thresholds_reads_custom_file(tmp_path):
    custom = tmp_path / "custom.json"
    custom.write_text('{"T009": {"warn_count": 7, "fault_count": 21}}', encoding="utf-8")
    thresholds = load_check_thresholds(str(custom))
    assert thresholds["T009"] == {"warn_count": 7, "fault_count": 21}


def test_profile_configs_apply_check_thresholds_with_no_call_site_changes():
    # config_for_profile("development") / pipeline_for_profile(...) take no new
    # argument -- the persisted thresholds must be picked up automatically.
    cfg = config_for_profile("development")
    assert cfg.checks["T009"].params == {"warn_count": 15, "fault_count": 30}
    manager = build_manager(cfg)
    click_check = next(c for c in manager.checks if c.check_id == "T009")
    assert isinstance(click_check, ClickTransientCheck)
    assert click_check.warn_count == 15
    assert click_check.fault_count == 30


def test_profile_configs_check_thresholds_do_not_clobber_existing_overrides():
    # production disables F004 via an explicit CheckConfig(enabled=False) --
    # applying persisted thresholds must not stomp on that (F004 has no
    # persisted threshold entry, but this guards the merge logic generally:
    # an explicit profile-level CheckConfig for a check_id must survive).
    cfg = config_for_profile("production")
    assert cfg.checks["F004"].enabled is False
