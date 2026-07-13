"""Class-conditional score likelihoods for SPRT evidence, fit via method of moments.

Pure NumPy — deliberately scipy-free so this stays importable in the project's fast
numpy+pytest test venv (no TFLite/librosa/scipy required to run these tests).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def fit_beta_params(scores: Sequence[float], eps: float = 1e-6) -> tuple[float, float]:
    """Closed-form method-of-moments fit of a Beta(a, b) distribution to `scores`.

    Scores are clipped to (eps, 1-eps) before fitting so degenerate 0/1 values (silence,
    saturation) don't break the moment equations. Variance is floored/capped relative to
    the theoretical max mean*(1-mean) so near-constant score streams still yield a valid,
    if very peaked, fit rather than a non-positive shape parameter.
    """
    x = np.clip(np.asarray(scores, dtype=np.float64), eps, 1.0 - eps)
    if x.size < 2:
        raise ValueError("fit_beta_params needs at least 2 scores")
    mean = float(np.mean(x))
    var = float(np.var(x, ddof=1))
    max_var = mean * (1.0 - mean)
    var = min(var, max_var * 0.99)
    var = max(var, max_var * 1e-4)
    common = max_var / var - 1.0
    a = mean * common
    b = (1.0 - mean) * common
    return max(a, eps), max(b, eps)


def _log_beta_pdf(x: float, a: float, b: float) -> float:
    """log of the Beta(a, b) PDF at x, via the log-gamma function (numerically stable)."""
    log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    return log_norm + (a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x)


@dataclass(frozen=True)
class FittedLikelihood:
    """Per-class Beta(a, b) score densities fit from labeled calibration recordings."""

    healthy_beta: tuple[float, float]
    infested_beta: tuple[float, float]

    def to_json(self) -> str:
        return json.dumps(
            {
                "healthy_beta": list(self.healthy_beta),
                "infested_beta": list(self.infested_beta),
            }
        )

    @staticmethod
    def from_json(text: str) -> "FittedLikelihood":
        data = json.loads(text)
        return FittedLikelihood(
            healthy_beta=tuple(data["healthy_beta"]),
            infested_beta=tuple(data["infested_beta"]),
        )


def fitted_llr_increment(score: float, fitted: FittedLikelihood, eps: float = 1e-6) -> float:
    """log( f_infested(score) / f_healthy(score) ) under the fitted class-conditional Betas."""
    p = min(max(score, eps), 1.0 - eps)
    log_f1 = _log_beta_pdf(p, *fitted.infested_beta)
    log_f0 = _log_beta_pdf(p, *fitted.healthy_beta)
    return log_f1 - log_f0
