"""Factories that assemble the standard Phase 1 health-monitoring pipeline.

Phase 2 replaces this with the configuration-profile system; for now it simply
registers the seven time-domain checks with their default manual thresholds.
"""
from __future__ import annotations

from app.health.checks.base import SignalHealthCheck
from app.health.checks.time_domain import (
    ClippingCheck,
    CrestFactorCheck,
    DCOffsetCheck,
    FlatlineCheck,
    PeakAmplitudeCheck,
    SignalEnergyCheck,
    ZeroCrossingRateCheck,
)
from app.health.checks.frequency_domain import (
    BandEnergyDistributionCheck,
    ElectricalHumCheck,
    SpectralFlatnessCheck,
    SpectralShapeCheck,
)
from app.health.manager import SignalCheckManager
from app.health.pipeline import HealthAnalysisPipeline


def default_time_domain_checks() -> list[SignalHealthCheck]:
    """The seven time-domain checks (T001–T007) with default thresholds."""
    return [
        FlatlineCheck(),
        SignalEnergyCheck(),
        PeakAmplitudeCheck(),
        ClippingCheck(),
        CrestFactorCheck(),
        DCOffsetCheck(),
        ZeroCrossingRateCheck(),
    ]


def default_frequency_domain_checks() -> list[SignalHealthCheck]:
    """The four frequency-domain checks (F001–F004) with default thresholds."""
    return [
        SpectralShapeCheck(),
        SpectralFlatnessCheck(),
        BandEnergyDistributionCheck(),
        ElectricalHumCheck(),
    ]


def default_manager() -> SignalCheckManager:
    manager = SignalCheckManager()
    for check in default_time_domain_checks() + default_frequency_domain_checks():
        manager.register(check)
    return manager


def default_pipeline() -> HealthAnalysisPipeline:
    return HealthAnalysisPipeline(manager=default_manager())
