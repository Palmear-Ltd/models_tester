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

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from app.health.checks.frequency_domain import (
    BandEnergyDistributionCheck,
    ElectricalHumCheck,
    SpectralFlatnessCheck,
    SpectralShapeCheck,
)
from app.health.checks.time_domain import (
    ClickTransientCheck,
    ClippingCheck,
    CrestFactorCheck,
    DCOffsetCheck,
    DropoutSegmentCheck,
    FlatlineCheck,
    PeakAmplitudeCheck,
    SignalEnergyCheck,
    ZeroCrossingRateCheck,
)
from app.health.checks.stability import (
    DropoutRecurrenceCheck,
    EnergyStabilityCheck,
    LongTermNoiseFloorCheck,
    SpectralStabilityCheck,
)
from app.health.manager import SignalCheckManager
from app.health.pipeline import HealthAnalysisPipeline


@dataclass(frozen=True)
class CheckSpec:
    """Static metadata describing one available check."""

    check_id: str
    factory: type  # a SignalHealthCheck subclass
    category: str  # config group: "time_domain" | "frequency_domain" | "stability"
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
    CheckSpec("T008", DropoutSegmentCheck, "time_domain"),
    CheckSpec("T009", ClickTransientCheck, "time_domain"),
    CheckSpec("F001", SpectralShapeCheck, "frequency_domain"),
    CheckSpec("F002", SpectralFlatnessCheck, "frequency_domain"),
    CheckSpec("F003", BandEnergyDistributionCheck, "frequency_domain"),
    CheckSpec("F004", ElectricalHumCheck, "frequency_domain"),
    CheckSpec("S001", EnergyStabilityCheck, "stability"),
    CheckSpec("S002", SpectralStabilityCheck, "stability"),
    CheckSpec("S003", LongTermNoiseFloorCheck, "stability"),
    CheckSpec("S004", DropoutRecurrenceCheck, "stability"),
]


@dataclass
class CheckConfig:
    """Per-check configuration overlay."""

    enabled: bool = True
    mandatory: bool = False
    params: dict[str, object] = field(default_factory=dict)


@dataclass
class HealthConfig:
    """Resolved configuration controlling which checks run and how."""

    profile: str = "development"
    categories: dict[str, bool] = field(default_factory=dict)  # group name -> enabled
    checks: dict[str, "CheckConfig"] = field(default_factory=dict)  # check_id -> CheckConfig

    def is_active(self, spec: CheckSpec) -> bool:
        cc: Optional[CheckConfig] = self.checks.get(spec.check_id)
        mandatory = spec.mandatory_default or (cc.mandatory if cc else False)
        if mandatory:
            return True
        if not self.categories.get(spec.category, True):
            return False
        return cc.enabled if cc else True


# Persisted per-check threshold overrides (spec: docs/superpowers/specs/
# 2026-07-16-rootcause-threshold-recalibration-design.md). This JSON file is
# the source of truth at runtime; the dict below is only the shipped-in-code
# fallback used if the file is missing/unreadable/malformed -- loading must
# NEVER raise, mirroring app/decision/threshold.py's default_config() idiom
# (frozen JSON if present, else a hardcoded default). Only entries that
# differ from a check's class-constructor defaults need to be present in the
# file; both currently agree (the constructor defaults were updated to match
# as the primary fix -- this file lets them be retuned without a code change).
DEFAULT_CHECK_THRESHOLDS_PATH = os.path.join(
    os.path.dirname(__file__), "check_thresholds.json"
)

_SHIPPED_CHECK_THRESHOLD_DEFAULTS: dict[str, dict] = {
    "T009": {"warn_count": 15, "fault_count": 30},
}


def load_check_thresholds(path: Optional[str] = None) -> dict[str, dict]:
    """Load per-check constructor-param overrides from a persisted JSON config.

    ``{"T009": {"warn_count": 15, "fault_count": 30}, ...}`` -- only checks
    that need a non-default value need an entry. Falls back to the shipped
    defaults (never raises) if the file is absent, unreadable, or malformed.
    """
    resolved = path if path is not None else DEFAULT_CHECK_THRESHOLDS_PATH
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError):
        pass
    return {k: dict(v) for k, v in _SHIPPED_CHECK_THRESHOLD_DEFAULTS.items()}


def _apply_check_thresholds(config: HealthConfig, thresholds: dict[str, dict]) -> HealthConfig:
    """Merge persisted threshold overrides into a HealthConfig in place.

    An explicit profile-level CheckConfig (e.g. production's F004
    enabled=False) always wins for fields it sets; params from the persisted
    file only fill in what the profile didn't already specify.
    """
    for check_id, params in thresholds.items():
        existing = config.checks.get(check_id)
        if existing is not None:
            existing.params = {**params, **existing.params}
        else:
            config.checks[check_id] = CheckConfig(params=dict(params))
    return config


def build_manager(config: HealthConfig) -> SignalCheckManager:
    """Register every active check (in registry order) into a fresh manager."""
    manager = SignalCheckManager()
    for spec in REGISTRY:
        if config.is_active(spec):
            cc = config.checks.get(spec.check_id)
            params = cc.params if cc else {}
            manager.register(spec.factory(**params))
    return manager


def build_pipeline(config: HealthConfig, calibration_profile=None, anomaly_p: float = 0.001) -> HealthAnalysisPipeline:
    return HealthAnalysisPipeline(
        manager=build_manager(config), calibration_profile=calibration_profile, anomaly_p=anomaly_p
    )


PROFILES = ("development", "production", "diagnostic", "minimal")


def _profile_configs() -> dict:
    configs = {
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
            categories={"time_domain": False, "frequency_domain": False, "stability": False},
        ),
    }
    thresholds = load_check_thresholds()
    for config in configs.values():
        _apply_check_thresholds(config, thresholds)
    return configs


def config_for_profile(name: str) -> HealthConfig:
    configs = _profile_configs()
    if name not in configs:
        raise KeyError(f"Unknown health profile: {name!r} (known: {sorted(configs)})")
    return configs[name]


def pipeline_for_profile(name: str, calibration_profile=None, anomaly_p: float = 0.001) -> HealthAnalysisPipeline:
    return build_pipeline(
        config_for_profile(name), calibration_profile=calibration_profile, anomaly_p=anomaly_p
    )
