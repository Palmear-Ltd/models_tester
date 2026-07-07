"""Frequency-domain Signal Health Checks (spec §4.9, F001–F004).

Each check consumes the shared spectrum from feature preparation
(`features["power_spectrum"]`, `features["freqs"]`) and never recomputes an FFT.
Pure NumPy. Thresholds are provisional manual defaults; Phase 3 replaces them with
calibration-derived values. A window with no spectral energy (silence/non-finite)
PASSES here — the time-domain critical checks (Flatline, Signal Energy) own that
failure, so the frequency layer must not double-report it.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from app.health.checks.base import SignalHealthCheck
from app.health.models import (
    AudioWindow,
    CheckCategory,
    CheckStatus,
    Measurement,
    SignalCheckResult,
)


def _usable_spectrum(features: dict[str, Any]) -> Optional[tuple]:
    """Return (freqs, power, total_power) or None when there is no usable energy."""
    freqs = features.get("freqs")
    power = features.get("power_spectrum")
    if freqs is None or power is None or power.size == 0:
        return None
    total = float(power.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    return freqs, power, total


class SpectralShapeCheck(SignalHealthCheck):
    """F001 — verify overall frequency response (centroid, bandwidth, roll-off)."""

    check_id = "F001"
    check_name = "Spectral Shape"
    category = CheckCategory.PRIMARY

    def __init__(
        self,
        centroid_range: tuple = (50.0, 12000.0),
        bandwidth_range: tuple = (0.0, 12000.0),
        rolloff_fraction: float = 0.85,
        rolloff_range: tuple = (0.0, 20000.0),
    ):
        self.centroid_range = centroid_range
        self.bandwidth_range = bandwidth_range
        self.rolloff_fraction = rolloff_fraction
        self.rolloff_range = rolloff_range

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        spec = _usable_spectrum(features)
        if spec is None:
            return SignalCheckResult(
                check_id=self.check_id,
                check_name=self.check_name,
                status=CheckStatus.PASS,
                measurements=[
                    Measurement("spectral_centroid", 0.0, "Hz"),
                    Measurement("spectral_bandwidth", 0.0, "Hz"),
                    Measurement("spectral_rolloff", 0.0, "Hz"),
                ],
            )
        freqs, power, total = spec
        centroid = float((freqs * power).sum() / total)
        bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * power).sum() / total))
        cumulative = np.cumsum(power)
        # searchsorted (side='left') -> first bin whose cumulative power reaches the
        # fraction; clamp guards the (unreachable here) case of fraction >= 1.0.
        idx = int(np.searchsorted(cumulative, self.rolloff_fraction * total))
        rolloff = float(freqs[min(idx, freqs.size - 1)])

        diagnostics = []
        if not (self.centroid_range[0] <= centroid <= self.centroid_range[1]):
            diagnostics.append(f"Spectral centroid {centroid:.0f} Hz outside expected range")
        if not (self.bandwidth_range[0] <= bandwidth <= self.bandwidth_range[1]):
            diagnostics.append(f"Spectral bandwidth {bandwidth:.0f} Hz outside expected range")
        if not (self.rolloff_range[0] <= rolloff <= self.rolloff_range[1]):
            diagnostics.append(f"Spectral roll-off {rolloff:.0f} Hz outside expected range")
        status = CheckStatus.WARNING if diagnostics else CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[
                Measurement("spectral_centroid", centroid, "Hz"),
                Measurement("spectral_bandwidth", bandwidth, "Hz"),
                Measurement("spectral_rolloff", rolloff, "Hz"),
            ],
            diagnostic_messages=diagnostics,
        )


class SpectralFlatnessCheck(SignalHealthCheck):
    """F002 — measure how noise-like the spectrum is (geometric/arithmetic mean)."""

    check_id = "F002"
    check_name = "Spectral Flatness"
    category = CheckCategory.SUPPORTING

    def __init__(self, maximum_flatness: float = 0.6):
        self.maximum_flatness = maximum_flatness

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        spec = _usable_spectrum(features)
        if spec is None:
            return SignalCheckResult(
                check_id=self.check_id,
                check_name=self.check_name,
                status=CheckStatus.PASS,
                measurements=[Measurement("spectral_flatness", 0.0)],
            )
        _, power, _ = spec
        p = power + 1e-20
        geometric = float(np.exp(np.mean(np.log(p))))
        arithmetic = float(np.mean(p))
        flatness = geometric / arithmetic if arithmetic > 0 else 0.0
        diagnostics = []
        if flatness > self.maximum_flatness:
            status = CheckStatus.WARNING
            diagnostics.append(f"Spectral flatness {flatness:.3f} indicates broadband noise")
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[Measurement("spectral_flatness", flatness)],
            diagnostic_messages=diagnostics,
        )


class BandEnergyDistributionCheck(SignalHealthCheck):
    """F003 — report the energy distribution across frequency bands.

    Phase 2a is measurement-only (always PASS): it characterises the band
    distribution for the panel and for calibration. Phase 3 adds calibrated
    thresholds on the expected distribution.
    """

    check_id = "F003"
    check_name = "Band Energy Distribution"
    category = CheckCategory.PRIMARY

    def __init__(self, bands: tuple = (
        (0.0, 500.0),
        (500.0, 2000.0),
        (2000.0, 8000.0),
        (8000.0, 22050.0),
    )):
        self.bands = bands

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        spec = _usable_spectrum(features)
        measurements = []
        if spec is None:
            for lo, hi in self.bands:
                measurements.append(
                    Measurement(f"band_{int(lo)}_{int(hi)}_ratio", 0.0)
                )
            return SignalCheckResult(
                check_id=self.check_id,
                check_name=self.check_name,
                status=CheckStatus.PASS,
                measurements=measurements,
            )
        freqs, power, total = spec
        for lo, hi in self.bands:
            mask = (freqs >= lo) & (freqs < hi)
            ratio = float(power[mask].sum()) / total
            measurements.append(Measurement(f"band_{int(lo)}_{int(hi)}_ratio", ratio))
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=CheckStatus.PASS,
            measurements=measurements,
        )


class ElectricalHumCheck(SignalHealthCheck):
    """F004 — detect mains-frequency interference (fundamental + harmonics)."""

    check_id = "F004"
    check_name = "Electrical Hum Detection"
    category = CheckCategory.SUPPORTING

    def __init__(
        self,
        fundamental_frequency: float = 60.0,
        harmonic_count: int = 3,
        analysis_bandwidth: float = 2.0,
        max_hum_ratio: float = 0.1,
    ):
        self.fundamental_frequency = fundamental_frequency
        self.harmonic_count = harmonic_count
        self.analysis_bandwidth = analysis_bandwidth
        self.max_hum_ratio = max_hum_ratio

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        spec = _usable_spectrum(features)
        if spec is None:
            return SignalCheckResult(
                check_id=self.check_id,
                check_name=self.check_name,
                status=CheckStatus.PASS,
                measurements=[
                    Measurement("hum_ratio", 0.0),
                    Measurement("hum_energy", 0.0),
                ],
            )
        freqs, power, total = spec
        hum_energy = 0.0
        for k in range(1, self.harmonic_count + 1):
            target = self.fundamental_frequency * k
            mask = np.abs(freqs - target) <= self.analysis_bandwidth
            hum_energy += float(power[mask].sum())
        hum_ratio = hum_energy / total
        diagnostics = []
        if hum_ratio > self.max_hum_ratio:
            status = CheckStatus.WARNING
            diagnostics.append(
                f"Mains hum ratio {hum_ratio:.3f} around {self.fundamental_frequency:.0f} Hz"
            )
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[
                Measurement("hum_ratio", hum_ratio),
                Measurement("hum_energy", hum_energy),
            ],
            diagnostic_messages=diagnostics,
        )
