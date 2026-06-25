"""Stability Signal Health Checks (spec §4.10, S001–S003).

These read the pipeline's per-window measurement history from
``features["history"]`` — a list of ``{check_id: {measurement_name: value}}``
snapshots (oldest -> newest, prior windows only) — and flag intermittent or
gradual changes. All SUPPORTING and stateless; they PASS when there is too little
history or the source measurement is absent. Thresholds are provisional manual
defaults. Pure NumPy.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.health.checks.base import SignalHealthCheck
from app.health.models import (
    AudioWindow,
    CheckCategory,
    CheckStatus,
    Measurement,
    SignalCheckResult,
)


def _series(history, check_id: str, measurement: str) -> list[float]:
    """Pull a measurement's values across the history snapshots (skipping gaps)."""
    values = []
    for snap in history:
        v = snap.get(check_id, {}).get(measurement)
        if v is not None:
            values.append(float(v))
    return values


def _coefficient_of_variation(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    return float(arr.std() / mean) if mean > 0 else 0.0


class EnergyStabilityCheck(SignalHealthCheck):
    """S001 — flag unstable signal energy over recent windows (loose contacts, etc.)."""

    check_id = "S001"
    check_name = "Energy Stability"
    category = CheckCategory.SUPPORTING

    def __init__(self, min_samples: int = 5, max_variation: float = 0.5):
        self.min_samples = min_samples
        self.max_variation = max_variation

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        series = _series(features.get("history", []), "T002", "rms")
        if len(series) < self.min_samples:
            return self._result(0.0, CheckStatus.PASS, [])
        cv = _coefficient_of_variation(series)
        if cv > self.max_variation:
            return self._result(
                cv, CheckStatus.WARNING,
                [f"Energy variation {cv:.2f} over recent windows"],
            )
        return self._result(cv, CheckStatus.PASS, [])

    def _result(self, cv, status, diagnostics) -> SignalCheckResult:
        return SignalCheckResult(
            check_id=self.check_id, check_name=self.check_name, status=status,
            measurements=[Measurement("energy_cv", cv)], diagnostic_messages=diagnostics,
        )


class SpectralStabilityCheck(SignalHealthCheck):
    """S002 — flag drifting spectral centroid over recent windows (sensor ageing)."""

    check_id = "S002"
    check_name = "Spectral Stability"
    category = CheckCategory.SUPPORTING

    def __init__(self, min_samples: int = 5, max_variation: float = 0.3):
        self.min_samples = min_samples
        self.max_variation = max_variation

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        series = _series(features.get("history", []), "F001", "spectral_centroid")
        if len(series) < self.min_samples:
            return self._result(0.0, CheckStatus.PASS, [])
        cv = _coefficient_of_variation(series)
        if cv > self.max_variation:
            return self._result(
                cv, CheckStatus.WARNING,
                [f"Spectral centroid variation {cv:.2f} over recent windows"],
            )
        return self._result(cv, CheckStatus.PASS, [])

    def _result(self, cv, status, diagnostics) -> SignalCheckResult:
        return SignalCheckResult(
            check_id=self.check_id, check_name=self.check_name, status=status,
            measurements=[Measurement("centroid_cv", cv)], diagnostic_messages=diagnostics,
        )


class LongTermNoiseFloorCheck(SignalHealthCheck):
    """S003 — flag an elevated background noise floor (estimated from recent RMS)."""

    check_id = "S003"
    check_name = "Long-Term Noise Floor"
    category = CheckCategory.SUPPORTING

    def __init__(self, min_samples: int = 5, max_noise_floor: float = 0.05):
        self.min_samples = min_samples
        self.max_noise_floor = max_noise_floor

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        series = _series(features.get("history", []), "T002", "rms")
        if len(series) < self.min_samples:
            return self._result(0.0, CheckStatus.PASS, [])
        noise_floor = float(np.percentile(series, 25))
        if noise_floor > self.max_noise_floor:
            return self._result(
                noise_floor, CheckStatus.WARNING,
                [f"Background noise floor {noise_floor:.3f} elevated"],
            )
        return self._result(noise_floor, CheckStatus.PASS, [])

    def _result(self, noise_floor, status, diagnostics) -> SignalCheckResult:
        return SignalCheckResult(
            check_id=self.check_id, check_name=self.check_name, status=status,
            measurements=[Measurement("noise_floor", noise_floor)],
            diagnostic_messages=diagnostics,
        )
