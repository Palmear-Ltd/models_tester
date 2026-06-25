"""Anomaly detection (spec Ch. 6 / Phase 6): a holistic distance of a window's
measurements from the calibration profile, turned into a confidence.

Diagonal Mahalanobis (RMS z-distance) — calibration profiles store only
per-measurement stats, so this treats measurements as independent. Pure NumPy.
The profile is duck-typed: ``profile.statistics[check_id][name]`` with ``.mean`` /
``.std``; results are duck-typed (``check_id`` + ``measurements`` of ``name``/``value``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AnomalyResult:
    distance: float
    threshold: float
    is_anomalous: bool
    contributors: list[tuple[str, float]]  # top (label, z) by |z|
    # Distance-from-profile normality, not a probability of good health: 1.0 at the
    # profile centroid, decaying with distance. (State is owned by checks/calibration.)
    confidence: float


def detect_anomaly(results, profile, *, threshold: float = 3.0):
    statistics = profile.statistics
    deviations = []  # (label, z)
    for result in results:
        check_stats = statistics.get(result.check_id)
        if not check_stats:
            continue
        for measurement in result.measurements:
            stat = check_stats.get(measurement.name)
            if stat is None or stat.std <= 0:
                continue
            z = (float(measurement.value) - stat.mean) / stat.std
            if not np.isfinite(z):  # skip NaN/Inf measurements rather than poison the distance
                continue
            deviations.append((f"{result.check_id}.{measurement.name}", z))

    if not deviations:
        return None

    z_values = np.array([z for _, z in deviations], dtype=np.float64)
    distance = float(np.sqrt(np.mean(z_values ** 2)))
    is_anomalous = distance > threshold
    # Linear decay: 1.0 at the centroid, 0.5 at the threshold, 0.0 at twice the threshold.
    confidence = max(0.0, 1.0 - distance / (2 * threshold))
    contributors = sorted(deviations, key=lambda item: -abs(item[1]))[:3]
    return AnomalyResult(
        distance=distance,
        threshold=threshold,
        is_anomalous=is_anomalous,
        contributors=contributors,
        confidence=confidence,
    )
