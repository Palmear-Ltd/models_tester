"""Anomaly detection (spec Ch. 6 / Phase 7c): full-covariance Mahalanobis distance
of a window's measurements from the calibration profile, thresholded by a
chi-square critical value and turned into a confidence.

The profile is duck-typed: ``profile.feature_index`` (ordered [check_id, name]
pairs), ``profile.mean_vector`` (length D), ``profile.covariance`` (D x D). Results
are duck-typed (``check_id`` + ``measurements`` of ``name``/``value``). Pure NumPy
plus the stdlib chi-square helper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.health.chi2 import chi2_ppf


@dataclass
class AnomalyResult:
    distance: float          # Mahalanobis distance d
    threshold: float         # sqrt(chi2 critical value) at (1 - p, df)
    is_anomalous: bool
    contributors: list       # top (label, contribution-to-d^2) by |contribution|
    # Distance-from-profile normality, not a probability of good health: 1.0 at the
    # profile centroid, 0.5 at the threshold, 0.0 at twice the threshold.
    confidence: float


def detect_anomaly(results, profile, *, p: float = 0.001):
    feature_index = getattr(profile, "feature_index", None)
    mean_vector = getattr(profile, "mean_vector", None)
    covariance = getattr(profile, "covariance", None)
    if not feature_index or not mean_vector or not covariance:
        return None  # no covariance model (e.g. v1 profile) -> unchanged behavior

    # Map this window's measurements by (check_id, name).
    value_by_key = {}
    for result in results:
        for m in result.measurements:
            value_by_key[(result.check_id, m.name)] = float(m.value)

    keys = [tuple(k) for k in feature_index]
    mu_full = np.asarray(mean_vector, dtype=np.float64)
    cov_full = np.asarray(covariance, dtype=np.float64)

    # Subselect the dimensions actually present (and finite) this window.
    idx = [
        i for i, key in enumerate(keys)
        if key in value_by_key and np.isfinite(value_by_key[key])
    ]
    if not idx:
        return None

    x = np.array([value_by_key[keys[i]] for i in idx], dtype=np.float64)
    mu = mu_full[idx]
    cov = cov_full[np.ix_(idx, idx)]

    # Regularize for singular / ill-conditioned covariance.
    d = cov.shape[0]
    ridge = 1e-9 * (np.trace(cov) / d if d else 1.0)
    cov = cov + (ridge if ridge > 0 else 1e-12) * np.eye(d)

    diff = x - mu
    try:
        sol = np.linalg.solve(cov, diff)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(cov) @ diff

    per_dim = diff * sol  # each dim's contribution to d^2
    d2 = float(max(0.0, np.sum(per_dim)))
    distance = float(np.sqrt(d2))

    crit2 = chi2_ppf(1.0 - p, d)
    threshold = float(np.sqrt(crit2))
    is_anomalous = d2 > crit2
    confidence = max(0.0, 1.0 - distance / (2.0 * threshold)) if threshold > 0 else 0.0

    order = np.argsort(-np.abs(per_dim))[:3]
    contributors = [
        (f"{keys[idx[i]][0]}.{keys[idx[i]][1]}", float(per_dim[i])) for i in order
    ]

    return AnomalyResult(
        distance=distance,
        threshold=threshold,
        is_anomalous=is_anomalous,
        contributors=contributors,
        confidence=confidence,
    )
