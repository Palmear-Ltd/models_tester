"""Time-domain Signal Health Checks (spec §4.8, T001–T007).

Each check operates directly on the waveform (NumPy only) and reports a status
plus measurements. Thresholds are provisional manual defaults; Phase 3 replaces
them with calibration-derived values.
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


class FlatlineCheck(SignalHealthCheck):
    """T001 — detect complete loss of signal (disconnected sensor, dead cable)."""

    check_id = "T001"
    check_name = "Flatline Detection"
    category = CheckCategory.CRITICAL

    def __init__(self, min_std: float = 1e-5, min_peak_to_peak: float = 1e-4):
        self.min_std = min_std
        self.min_peak_to_peak = min_peak_to_peak

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        x = window.samples
        # A non-finite or empty window is a broken signal, not a healthy one.
        if x.size == 0 or not np.all(np.isfinite(x)):
            return SignalCheckResult(
                check_id=self.check_id,
                check_name=self.check_name,
                status=CheckStatus.FAIL,
                measurements=[
                    Measurement("std", 0.0),
                    Measurement("peak_to_peak", 0.0),
                ],
                diagnostic_messages=["Flatline: empty or non-finite (NaN/Inf) signal"],
            )
        std = float(np.std(x))
        p2p = float(x.max() - x.min())
        diagnostics = []
        # A true flatline shows no variation by BOTH measures; requiring both to be
        # below threshold avoids false alarms on legitimately quiet signals.
        if std < self.min_std and p2p < self.min_peak_to_peak:
            status = CheckStatus.FAIL
            diagnostics.append(
                f"Flatline: std={std:.2e}, peak-to-peak={p2p:.2e} below minimum"
            )
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[
                Measurement("std", std),
                Measurement("peak_to_peak", p2p),
            ],
            diagnostic_messages=diagnostics,
        )


class SignalEnergyCheck(SignalHealthCheck):
    """T002 — verify the signal carries enough (but not excessive) energy."""

    check_id = "T002"
    check_name = "Signal Energy"
    category = CheckCategory.CRITICAL

    def __init__(
        self,
        min_rms_fault: float = 1e-4,
        min_rms_warn: float = 1e-3,
        max_rms_warn: float = 0.7,
        max_rms_fault: float = 0.9,
    ):
        self.min_rms_fault = min_rms_fault
        self.min_rms_warn = min_rms_warn
        self.max_rms_warn = max_rms_warn
        self.max_rms_fault = max_rms_fault

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        x = window.samples
        rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
        diagnostics = []
        if rms < self.min_rms_fault or rms > self.max_rms_fault:
            status = CheckStatus.FAIL
            diagnostics.append(f"RMS energy {rms:.2e} outside acceptable range")
        elif rms < self.min_rms_warn or rms > self.max_rms_warn:
            status = CheckStatus.WARNING
            diagnostics.append(f"RMS energy {rms:.2e} outside expected range")
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[Measurement("rms", rms)],
            diagnostic_messages=diagnostics,
        )


class ClippingCheck(SignalHealthCheck):
    """T004 — detect ADC/amplifier saturation via the fraction of clipped samples."""

    check_id = "T004"
    check_name = "Clipping Detection"
    category = CheckCategory.CRITICAL

    def __init__(
        self,
        clipping_threshold: float = 0.99,
        warning_ratio: float = 0.001,
        fault_ratio: float = 0.01,
    ):
        self.clipping_threshold = clipping_threshold
        self.warning_ratio = warning_ratio
        self.fault_ratio = fault_ratio

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        x = window.samples
        n = x.size
        clipped = int(np.count_nonzero(np.abs(x) >= self.clipping_threshold))
        ratio = clipped / n if n else 0.0
        diagnostics = []
        if ratio >= self.fault_ratio:
            status = CheckStatus.FAIL
            diagnostics.append(f"Clipping ratio {ratio:.3%} indicates saturation")
        elif ratio >= self.warning_ratio:
            status = CheckStatus.WARNING
            diagnostics.append(f"Clipping ratio {ratio:.3%} elevated")
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[
                Measurement("clipping_ratio", ratio),
                Measurement("clipped_samples", float(clipped)),
            ],
            diagnostic_messages=diagnostics,
        )


class PeakAmplitudeCheck(SignalHealthCheck):
    """T003 — verify the peak amplitude is within the expected operating range."""

    check_id = "T003"
    check_name = "Peak Amplitude"
    category = CheckCategory.PRIMARY

    def __init__(self, min_peak: float = 1e-3, max_peak: float = 0.99):
        self.min_peak = min_peak
        self.max_peak = max_peak

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        x = window.samples
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        diagnostics = []
        if peak < self.min_peak or peak > self.max_peak:
            status = CheckStatus.WARNING
            diagnostics.append(f"Peak amplitude {peak:.3e} outside expected range")
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[Measurement("peak_amplitude", peak)],
            diagnostic_messages=diagnostics,
        )


class CrestFactorCheck(SignalHealthCheck):
    """T005 — evaluate waveform dynamics via peak-to-RMS ratio."""

    check_id = "T005"
    check_name = "Crest Factor"
    category = CheckCategory.SUPPORTING

    def __init__(self, min_crest: float = 1.2, max_crest: float = 50.0):
        self.min_crest = min_crest
        self.max_crest = max_crest

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        x = window.samples
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
        crest = peak / rms if rms > 0 else 0.0
        diagnostics = []
        # A zero RMS (dead signal) is handled by FlatlineCheck; skip here.
        if rms > 0 and (crest < self.min_crest or crest > self.max_crest):
            status = CheckStatus.WARNING
            diagnostics.append(f"Crest factor {crest:.2f} outside expected range")
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[Measurement("crest_factor", crest)],
            diagnostic_messages=diagnostics,
        )


class DCOffsetCheck(SignalHealthCheck):
    """T006 — detect a constant acquisition bias (DC offset)."""

    check_id = "T006"
    check_name = "DC Offset"
    category = CheckCategory.PRIMARY

    def __init__(
        self, max_dc_offset_warn: float = 0.02, max_dc_offset_fault: float = 0.1
    ):
        self.max_dc_offset_warn = max_dc_offset_warn
        self.max_dc_offset_fault = max_dc_offset_fault

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        x = window.samples
        dc = float(np.mean(x)) if x.size else 0.0
        magnitude = abs(dc)
        diagnostics = []
        if magnitude > self.max_dc_offset_fault:
            status = CheckStatus.FAIL
            diagnostics.append(f"DC offset {dc:.3f} exceeds fault limit")
        elif magnitude > self.max_dc_offset_warn:
            status = CheckStatus.WARNING
            diagnostics.append(f"DC offset {dc:.3f} elevated")
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[Measurement("dc_offset", dc)],
            diagnostic_messages=diagnostics,
        )


class ZeroCrossingRateCheck(SignalHealthCheck):
    """T007 — flag excessive high-frequency content via zero-crossing rate."""

    check_id = "T007"
    check_name = "Zero Crossing Rate"
    category = CheckCategory.SUPPORTING

    def __init__(self, max_zcr: float = 0.8):
        self.max_zcr = max_zcr

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        x = window.samples
        if x.size < 2:
            zcr = 0.0
        else:
            signs = np.signbit(x)
            crossings = int(np.count_nonzero(np.diff(signs)))
            zcr = crossings / (x.size - 1)
        diagnostics = []
        if zcr > self.max_zcr:
            status = CheckStatus.WARNING
            diagnostics.append(f"Zero-crossing rate {zcr:.2f} abnormally high")
        else:
            status = CheckStatus.PASS
        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[Measurement("zero_crossing_rate", zcr)],
            diagnostic_messages=diagnostics,
        )
