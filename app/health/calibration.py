"""Calibration: build a statistical reference profile from healthy recordings.

Runs the health pipeline over windows of known-good audio and records, per check
measurement, the statistical reference values (spec §6). Stores statistics only —
decision thresholds are derived from these in Phase 3b. Pure stdlib + NumPy; WAV
loading lives in the `calibrate.py` CLI to keep this module portable.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Iterator

import numpy as np

from app.health.models import AudioWindow


@dataclass
class MeasurementStats:
    count: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    p5: float
    p95: float


@dataclass
class CalibrationProfile:
    profile_id: str
    version: int = 1
    sensor_info: str = ""
    sample_rate: int = 44100
    window_seconds: float = 2.5
    interval_seconds: float = 0.5
    created: str = ""
    window_count: int = 0
    # statistics[check_id][measurement_name] -> MeasurementStats
    statistics: dict[str, dict[str, MeasurementStats]] = field(default_factory=dict)


def compute_stats(values) -> MeasurementStats:
    arr = np.asarray(values, dtype=np.float64)
    return MeasurementStats(
        count=int(arr.size),
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        minimum=float(np.min(arr)),
        maximum=float(np.max(arr)),
        p5=float(np.percentile(arr, 5)),
        p95=float(np.percentile(arr, 95)),
    )


def iter_windows(samples, sample_rate, window_seconds=2.5, hop_seconds=0.5) -> Iterator[np.ndarray]:
    """Yield consecutive fixed-length windows (last partial window is dropped)."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    win = int(round(window_seconds * sample_rate))
    hop = int(round(hop_seconds * sample_rate))
    if win <= 0 or hop <= 0 or x.size < win:
        return
    for start in range(0, x.size - win + 1, hop):
        yield x[start:start + win]


def generate_profile(
    signals,
    sample_rate,
    *,
    profile_id,
    sensor_info="",
    pipeline=None,
    window_seconds=2.5,
    hop_seconds=0.5,
) -> CalibrationProfile:
    """Run the pipeline over every window of every signal and accumulate stats.

    `signals` is an iterable of 1-D arrays (each a full recording). `pipeline`
    defaults to the development profile (all checks).
    """
    if pipeline is None:
        from app.health.config import pipeline_for_profile

        pipeline = pipeline_for_profile("development")

    collected: dict[str, dict[str, list[float]]] = {}
    window_count = 0
    for signal in signals:
        for window in iter_windows(signal, sample_rate, window_seconds, hop_seconds):
            report = pipeline.analyze(
                AudioWindow(samples=window, sample_rate=sample_rate)
            )
            window_count += 1
            for result in report.check_results:
                per_check = collected.setdefault(result.check_id, {})
                for m in result.measurements:
                    per_check.setdefault(m.name, []).append(float(m.value))

    statistics = {
        cid: {name: compute_stats(vals) for name, vals in meas.items()}
        for cid, meas in collected.items()
    }
    return CalibrationProfile(
        profile_id=profile_id,
        sensor_info=sensor_info,
        sample_rate=int(sample_rate),
        window_seconds=window_seconds,
        interval_seconds=hop_seconds,
        created=date.today().isoformat(),
        window_count=window_count,
        statistics=statistics,
    )


def save_profile(profile: CalibrationProfile, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(profile), f, indent=2)


def load_profile(path: str) -> CalibrationProfile:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["statistics"] = {
        cid: {name: MeasurementStats(**s) for name, s in meas.items()}
        for cid, meas in data.get("statistics", {}).items()
    }
    return CalibrationProfile(**data)
