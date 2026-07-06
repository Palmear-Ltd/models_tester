import numpy as np

from app.health.checks.frequency_domain import (
    BandEnergyDistributionCheck,
    ElectricalHumCheck,
    SpectralFlatnessCheck,
    SpectralShapeCheck,
)
from app.health.feature_prep import prepare_features
from app.health.models import AudioWindow, CheckCategory, CheckStatus

SR = 44100
N = 110250
RNG = np.random.default_rng(0)


def _feats(x):
    return prepare_features(
        AudioWindow(samples=np.asarray(x, dtype=np.float32), sample_rate=SR)
    )


def _win(x):
    return AudioWindow(samples=np.asarray(x, dtype=np.float32), sample_rate=SR)


def _sine(freq, n=N, amp=0.3):
    t = np.arange(n) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def _measure(result, name):
    return next(m.value for m in result.measurements if m.name == name)


def test_spectral_shape_passes_on_sine_and_reports_centroid():
    check = SpectralShapeCheck()
    assert check.category is CheckCategory.PRIMARY
    x = _sine(1000.0)
    result = check.run(_win(x), _feats(x))
    assert result.status is CheckStatus.PASS
    assert abs(_measure(result, "spectral_centroid") - 1000.0) < 50.0


def test_spectral_shape_passes_on_silence():
    result = SpectralShapeCheck().run(_win(np.zeros(N)), _feats(np.zeros(N)))
    assert result.status is CheckStatus.PASS


def test_spectral_flatness_low_for_tone_high_for_noise():
    tone = _sine(1000.0)
    noise = RNG.uniform(-0.3, 0.3, N)
    flat_tone = SpectralFlatnessCheck().run(_win(tone), _feats(tone))
    flat_noise = SpectralFlatnessCheck().run(_win(noise), _feats(noise))
    assert flat_tone.status is CheckStatus.PASS
    assert _measure(flat_tone, "spectral_flatness") < _measure(
        flat_noise, "spectral_flatness"
    )


def test_spectral_flatness_warns_when_above_threshold():
    noise = RNG.uniform(-0.3, 0.3, N)
    result = SpectralFlatnessCheck(maximum_flatness=0.05).run(_win(noise), _feats(noise))
    assert result.status is CheckStatus.WARNING
    # Category is the check's class attribute (the manager stamps it onto results).
    assert SpectralFlatnessCheck().category is CheckCategory.SUPPORTING


def test_band_energy_reports_ratios_and_passes():
    x = _sine(1000.0)
    result = BandEnergyDistributionCheck().run(_win(x), _feats(x))
    assert result.status is CheckStatus.PASS
    assert result.category is CheckCategory.PRIMARY  # class attr (default on result is also PRIMARY)
    # A 1 kHz tone puts most energy in the 500–2000 Hz band.
    assert _measure(result, "band_500_2000_ratio") > 0.5


def test_band_energy_passes_on_silence():
    result = BandEnergyDistributionCheck().run(_win(np.zeros(N)), _feats(np.zeros(N)))
    assert result.status is CheckStatus.PASS


def test_electrical_hum_warns_on_mains_tone():
    hum = _sine(60.0, amp=0.3)
    result = ElectricalHumCheck(fundamental_frequency=60.0).run(_win(hum), _feats(hum))
    assert result.status is CheckStatus.WARNING
    assert ElectricalHumCheck().category is CheckCategory.SUPPORTING
    assert _measure(result, "hum_ratio") > 0.5


def test_electrical_hum_passes_on_clean_tone():
    x = _sine(1000.0)
    result = ElectricalHumCheck(fundamental_frequency=60.0).run(_win(x), _feats(x))
    assert result.status is CheckStatus.PASS
