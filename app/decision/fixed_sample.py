"""Fixed-sample-size decision rule: empirical quantile thresholds on a session-level
aggregate statistic (mean/median/EWMA-peak score), calibrated directly from labeled
session data.

Real recording sessions here are a fixed ~20s (40 windows), not open-ended — so an SPRT-
style "keep sampling until confident" method is structurally the wrong fit (it can only
ever use the 40 windows it's given, and abstains whenever that's not enough evidence).
This instead asks a fixed-sample question: "given exactly N windows, what statistic value
separates the classes at the target false-positive/false-negative rate?", and answers it
from the empirical class-conditional distribution of that statistic across the labeled
corpus, rather than an asymptotic or parametric assumption.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

ThreeWayState = Literal["HEALTHY", "SUSPICIOUS", "INFESTED"]


@dataclass(frozen=True)
class QuantileThresholds:
    t_low: float  # statistic below this -> HEALTHY (target false-negative rate = beta)
    t_high: float  # statistic above this -> INFESTED (target false-positive rate = alpha)
    alpha: float
    beta: float


def fit_quantile_thresholds(
    healthy_stats: Sequence[float],
    infested_stats: Sequence[float],
    alpha: float,
    beta: float,
) -> QuantileThresholds:
    """Empirical quantile thresholds from labeled session-level statistics.

    `t_high` is the (1-alpha) quantile of the healthy class's statistic: only `alpha`
    fraction of truly healthy sessions score above it, so crossing it decides INFESTED at
    a false-positive rate of approximately `alpha`. `t_low` is the beta quantile of the
    infested class's statistic, the symmetric construction for HEALTHY at a false-negative
    rate of approximately `beta`.

    If the two classes are well separated enough that `t_low` would land above `t_high`,
    there's no need for an ambiguous band — both collapse to their midpoint, making this a
    single clean cutoff.
    """
    if len(healthy_stats) < 2 or len(infested_stats) < 2:
        raise ValueError("need at least 2 samples per class to fit quantile thresholds")
    if not (0.0 < alpha < 1.0) or not (0.0 < beta < 1.0):
        raise ValueError(f"alpha and beta must be in (0, 1); got alpha={alpha}, beta={beta}")

    t_high = float(np.quantile(np.asarray(healthy_stats, dtype=np.float64), 1.0 - alpha))
    t_low = float(np.quantile(np.asarray(infested_stats, dtype=np.float64), beta))
    if t_low >= t_high:
        mid = (t_low + t_high) / 2.0
        t_low = t_high = mid
    return QuantileThresholds(t_low=t_low, t_high=t_high, alpha=alpha, beta=beta)


def classify(stat: float, thresholds: QuantileThresholds) -> ThreeWayState:
    if stat > thresholds.t_high:
        return "INFESTED"
    if stat < thresholds.t_low:
        return "HEALTHY"
    return "SUSPICIOUS"
