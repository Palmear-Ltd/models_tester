"""Simple session-level decision baselines, evaluated alongside SPRT for comparison.

Includes a faithful reproduction of main.py's existing count-band rule (the control the
new method has to beat), plus mean/median/EWMA-threshold alternatives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

FinalState = Literal["HEALTHY", "INFESTED"]
ThreeWayState = Literal["HEALTHY", "SUSPICIOUS", "INFESTED"]


@dataclass(frozen=True)
class BaselineResult:
    final_state: FinalState
    statistic: float


def mean_score(scores: Sequence[float]) -> float:
    return (sum(scores) / len(scores)) if scores else 0.0


def median_score(scores: Sequence[float]) -> float:
    ordered = sorted(scores)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def ewma_peak_score(scores: Sequence[float], span: float = 5.0) -> float:
    """EWMA-smoothed score, reporting the peak smoothed value reached in the session."""
    if not scores:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    smoothed = scores[0]
    peak = smoothed
    for s in scores[1:]:
        smoothed = alpha * s + (1.0 - alpha) * smoothed
        peak = max(peak, smoothed)
    return peak


def mean_threshold(scores: Sequence[float], threshold: float) -> BaselineResult:
    stat = mean_score(scores)
    return BaselineResult(final_state="INFESTED" if stat > threshold else "HEALTHY", statistic=stat)


def median_threshold(scores: Sequence[float], threshold: float) -> BaselineResult:
    stat = median_score(scores)
    return BaselineResult(final_state="INFESTED" if stat > threshold else "HEALTHY", statistic=stat)


def ewma_peak(scores: Sequence[float], threshold: float, span: float = 5.0) -> BaselineResult:
    stat = ewma_peak_score(scores, span)
    return BaselineResult(final_state="INFESTED" if stat > threshold else "HEALTHY", statistic=stat)


def current_method(
    scores: Sequence[float],
    score_thresh: float,
    susp_limit: int,
    inf_limit: int,
) -> ThreeWayState:
    """Reproduction of main.py's calculate_diagnosis (main.py:533-558): threshold each
    window's score into a positive/negative event, then compare the raw positive count
    against two absolute count bands. This is the control baseline the new method must
    beat, not a target to preserve."""
    pos = sum(1 for s in scores if s > score_thresh)
    if pos < susp_limit:
        return "HEALTHY"
    if pos <= inf_limit:
        return "SUSPICIOUS"
    return "INFESTED"
