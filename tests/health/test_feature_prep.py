import numpy as np

from app.health.feature_prep import prepare_features
from app.health.models import AudioWindow

SR = 44100
N = 110250


def _win(x):
    return AudioWindow(samples=np.asarray(x, dtype=np.float32), sample_rate=SR)


def _sine(freq, n=N, amp=0.3):
    t = np.arange(n) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def test_prepare_features_keys_and_shapes():
    feats = prepare_features(_win(_sine(1000.0)))
    assert set(feats) >= {"freqs", "power_spectrum", "sample_rate", "n"}
    assert feats["freqs"].shape == feats["power_spectrum"].shape
    assert feats["freqs"].shape[0] == N // 2 + 1  # rfft length
    assert feats["sample_rate"] == SR


def test_power_spectrum_peaks_at_tone_frequency():
    feats = prepare_features(_win(_sine(1000.0)))
    peak_freq = feats["freqs"][int(np.argmax(feats["power_spectrum"]))]
    assert abs(peak_freq - 1000.0) < 25.0  # within one rFFT bin-ish


def test_dc_is_removed():
    # A signal with a large DC offset must not put all energy in the 0 Hz bin.
    feats = prepare_features(_win(_sine(1000.0) + 0.5))
    assert feats["power_spectrum"][0] < feats["power_spectrum"].max()


def test_silence_has_zero_power():
    feats = prepare_features(_win(np.zeros(N)))
    assert float(feats["power_spectrum"].sum()) == 0.0
