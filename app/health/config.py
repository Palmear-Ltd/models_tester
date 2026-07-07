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
            categories={"time_domain": False, "frequency_domain": False, "stability": False},
        ),
    }


def config_for_profile(name: str) -> HealthConfig:
    configs = _profile_configs()
    if name not in configs:
        raise KeyError(f"Unknown health profile: {name!r} (known: {sorted(configs)})")
    return configs[name]


def pipeline_for_profile(name: str, calibration_profile=None, anomaly_p: float = 0.001) -> HealthAnalysisPipeline:
    return build_pipeline(
        config_for_profile(name), calibration_profile=calibration_profile, anomaly_p=anomaly_p
    )
