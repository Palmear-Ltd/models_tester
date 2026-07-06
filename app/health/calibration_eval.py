"""Calibration Evaluation (spec §3.8): compare a window's measurements against a
CalibrationProfile's per-measurement normal range and flag deviations.

Percentile-based: PASS inside [p5, p95] (widened by a tolerance), FAULT well
outside the observed [min, max] range, WARNING in between. Pure stdlib.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.health.models import CheckStatus


@dataclass
class MeasurementDeviation:
    check_id: str
    measurement: str
    value: float
    p5: float
    p95: float
    verdict: CheckStatus  # PASS / WARNING / FAIL (FAIL = calibration FAULT verdict)


@dataclass
class CalibrationEvaluation:
    deviations: list = field(default_factory=list)  # list[MeasurementDeviation]
    warn_count: int = 0
    fault_count: int = 0
    summary: str = ""


def _verdict(value, stats, warn_margin_frac, fault_margin, eps) -> CheckStatus:
    span = stats.p95 - stats.p5
    warn_lo = stats.p5 - warn_margin_frac * span
    warn_hi = stats.p95 + warn_margin_frac * span
    rng = stats.maximum - stats.minimum
    fault_lo = stats.minimum - fault_margin * rng - eps
    fault_hi = stats.maximum + fault_margin * rng + eps
    if value < fault_lo or value > fault_hi:
        return CheckStatus.FAIL
    if value < warn_lo or value > warn_hi:
        return CheckStatus.WARNING
    return CheckStatus.PASS


def evaluate_calibration(
    results, profile, *, warn_margin_frac=0.1, fault_margin=0.5, eps=1e-9
) -> CalibrationEvaluation:
    deviations = []
    warn_count = 0
    fault_count = 0
    for result in results:
        check_stats = profile.statistics.get(result.check_id)
        if not check_stats:
            continue
        for m in result.measurements:
            stats = check_stats.get(m.name)
            if stats is None:
                continue
            verdict = _verdict(float(m.value), stats, warn_margin_frac, fault_margin, eps)
            if verdict is CheckStatus.PASS:
                continue
            deviations.append(
                MeasurementDeviation(
                    check_id=result.check_id,
                    measurement=m.name,
                    value=float(m.value),
                    p5=stats.p5,
                    p95=stats.p95,
                    verdict=verdict,
                )
            )
            if verdict is CheckStatus.FAIL:
                fault_count += 1
            else:
                warn_count += 1
    if deviations:
        parts = [
            f"{d.check_id}.{d.measurement}={d.value:.4g} (cal {d.p5:.4g}..{d.p95:.4g})"
            for d in deviations
        ]
        summary = "calibration deviations: " + "; ".join(parts)
    else:
        summary = ""
    return CalibrationEvaluation(
        deviations=deviations,
        warn_count=warn_count,
        fault_count=fault_count,
        summary=summary,
    )
