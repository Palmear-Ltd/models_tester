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
    "S001", "S002", "S003",
}


def _ids(manager):
    return {c.check_id for c in manager.checks}


def test_registry_has_all_eleven_checks():
    assert len(REGISTRY) == 16
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
    assert len(ids) == 15


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
    assert len(ids) == 15


def test_development_and_diagnostic_have_all_eleven():
    assert len(_profile_ids("development")) == 16
    assert len(_profile_ids("diagnostic")) == 16


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        config_for_profile("bogus")


def test_pipeline_for_profile_minimal_faults_on_silence():
    report = pipeline_for_profile("minimal").analyze(
        AudioWindow(samples=np.zeros(110250, dtype=np.float32), sample_rate=44100)
    )
    assert report.final_state is HealthState.FAULT
    assert len(report.check_results) == 3
