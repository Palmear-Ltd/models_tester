"""Shared spectral representation computed once per analysis window (spec §3.6).

Frequency-domain checks consume this instead of each recomputing an FFT. Pure
NumPy. DC is removed and a Hann window applied before the rFFT to reduce spectral
leakage and stop the 0 Hz bin from dominating spectral statistics (DC is checked
separately by the time-domain DC Offset check).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.health.models import AudioWindow


def prepare_features(window: AudioWindow) -> dict[str, Any]:
    x = window.samples
    n = int(x.size)
    if n == 0:
        return {
            "sample_rate": int(window.sample_rate),
            "n": 0,
            "freqs": np.zeros(0, dtype=np.float64),
            "power_spectrum": np.zeros(0, dtype=np.float64),
        }
    xd = x.astype(np.float64)
    xd = xd - xd.mean()  # remove DC
    spectrum = np.fft.rfft(xd * np.hanning(n))
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / window.sample_rate)
    return {
        "sample_rate": int(window.sample_rate),
        "n": n,
        "freqs": freqs,
        "power_spectrum": power,
    }
