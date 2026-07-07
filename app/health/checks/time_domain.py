"""Time-domain Signal Health Checks (spec §4.8, T001–T007; T008–T009 add cable-fault
transient detection).

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


class DropoutSegmentCheck(SignalHealthCheck):
    """T008 — detect a brief envelope collapse mid-window (loose contact
    momentarily breaking circuit).

    S001 (inter-window RMS coefficient-of-variation) and T005 (crest factor)
    only weakly catch this: both operate on whole-window aggregates diluted
    by 2.5s of otherwise-normal signal around a single brief dropout. This
    check instead compares short (default 20ms) frames against a robust
    local reference (the window's median frame RMS).
    """

    check_id = "T008"
    check_name = "Dropout Segment Detection"
    category = CheckCategory.PRIMARY

    def __init__(
        self,
        frame_ms: float = 20.0,
        dropout_ratio: float = 0.15,
        min_event_ms: float = 30.0,
        fault_ratio: float = 0.15,
        ref_floor: float = 1e-4,
    ):
        self.frame_ms = frame_ms
        self.dropout_ratio = dropout_ratio
        self.min_event_ms = min_event_ms
        self.fault_ratio = fault_ratio
        self.ref_floor = ref_floor

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        x = window.samples
        frame_len = int(window.sample_rate * self.frame_ms / 1000)
        n_frames = x.size // frame_len if frame_len > 0 else 0
        if n_frames < 1:
            return SignalCheckResult(
                check_id=self.check_id,
                check_name=self.check_name,
                status=CheckStatus.PASS,
                measurements=[],
                diagnostic_messages=[],
            )

        frames = x[: n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        ref = float(np.median(frame_rms))

        # No local "normal" level to be intermittent relative to; total/near-total
        # silence is T001/T002's job, not this check's.
        if ref < self.ref_floor:
            return SignalCheckResult(
                check_id=self.check_id,
                check_name=self.check_name,
                status=CheckStatus.PASS,
                measurements=[],
                diagnostic_messages=[],
            )

        dropout_mask = frame_rms < self.dropout_ratio * ref

        # Vectorized run-length encoding via np.diff on a zero-padded boolean array.
        padded = np.concatenate(([False], dropout_mask, [False])).astype(np.int8)
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        run_lengths = ends - starts

        dropout_event_count = int(len(run_lengths))
        max_dropout_run_ms = (
            float(run_lengths.max()) * self.frame_ms if dropout_event_count else 0.0
        )
        dropout_frame_ratio = (
            float(run_lengths.sum()) / n_frames if dropout_event_count else 0.0
        )

        diagnostics: list[str] = []
        for start, end, length in zip(starts, ends, run_lengths):
            run_ms = float(length) * self.frame_ms
            boundary = start == 0 or end == n_frames
            msg = f"Dropout: {run_ms:.0f}ms gap at frame {start}"
            if boundary:
                msg += " (at window boundary)"
            diagnostics.append(msg)

        if dropout_frame_ratio >= self.fault_ratio:
            status = CheckStatus.FAIL
        elif dropout_event_count >= 1 and max_dropout_run_ms >= self.min_event_ms:
            status = CheckStatus.WARNING
        else:
            status = CheckStatus.PASS
            diagnostics = []

        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[
                Measurement("dropout_event_count", float(dropout_event_count)),
                Measurement("max_dropout_run_ms", max_dropout_run_ms, unit="ms"),
                Measurement("dropout_frame_ratio", dropout_frame_ratio),
            ],
            diagnostic_messages=diagnostics,
        )


class ClickTransientCheck(SignalHealthCheck):
    """T009 — detect sample-domain discontinuities (contact bounce/arcing
    clicks). Distinct from a dropout (a click is a sharp spike, not an
    envelope collapse).
    """

    check_id = "T009"
    check_name = "Click Transient Detection"
    category = CheckCategory.PRIMARY

    def __init__(
        self,
        click_k: float = 8.0,
        merge_gap: int = 3,
        warn_count: int = 3,
        fault_count: int = 15,
    ):
        self.click_k = click_k
        self.merge_gap = merge_gap
        self.warn_count = warn_count
        self.fault_count = fault_count

    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        x = window.samples.astype(np.float64)
        d = np.diff(x)
        if d.size == 0:
            return SignalCheckResult(
                check_id=self.check_id,
                check_name=self.check_name,
                status=CheckStatus.PASS,
                measurements=[],
                diagnostic_messages=[],
            )

        sigma = 1.4826 * float(np.median(np.abs(d - np.median(d))))

        # A flat/silent signal has zero variation in its first difference;
        # that's Flatline's job, not this check's.
        if sigma <= 0:
            return SignalCheckResult(
                check_id=self.check_id,
                check_name=self.check_name,
                status=CheckStatus.PASS,
                measurements=[],
                diagnostic_messages=[],
            )

        click_mask = np.abs(d) > self.click_k * sigma
        idxs = np.where(click_mask)[0]

        if idxs.size == 0:
            click_count = 0
        else:
            gaps = np.diff(idxs)
            group_starts = np.concatenate(([True], gaps > self.merge_gap))
            click_count = int(np.count_nonzero(group_starts))

        click_rate = click_count / window.window_duration if window.window_duration > 0 else 0.0

        diagnostics: list[str] = []
        if click_count >= self.fault_count:
            status = CheckStatus.FAIL
            diagnostics.append(
                f"Click transients: {click_count} events (rate {click_rate:.1f}/s) indicate dense clicking/crackle"
            )
        elif click_count >= self.warn_count:
            status = CheckStatus.WARNING
            diagnostics.append(
                f"Click transients: {click_count} events (rate {click_rate:.1f}/s)"
            )
        else:
            status = CheckStatus.PASS

        return SignalCheckResult(
            check_id=self.check_id,
            check_name=self.check_name,
            status=status,
            measurements=[
                Measurement("click_count", float(click_count)),
                Measurement("click_rate", click_rate, unit="/s"),
            ],
            diagnostic_messages=diagnostics,
        )
