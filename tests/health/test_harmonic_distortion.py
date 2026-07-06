import numpy as np

from app.health.checks.frequency_domain import (
    ElectricalHumCheck,
    HarmonicResonanceCheck,
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


def _pink_noise(seed, n=N, sr=SR):
    """Synthesize 1/f ("pink") noise by shaping white noise's FFT magnitude by
    1/sqrt(f) (power ~ 1/f) before an inverse transform. Real acoustic content
    (insect audio, ambient background) plausibly has this kind of non-flat,
    low-frequency-heavy noise floor -- unlike the flat white/uniform noise
    fixtures used elsewhere in this file, which don't exercise a non-flat
    local_floor reference at all.
    """
    rng = np.random.default_rng(seed)
    white = rng.normal(size=n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    scale = np.ones_like(freqs)
    nonzero = freqs > 0
    scale[nonzero] = 1.0 / np.sqrt(freqs[nonzero])
    if nonzero.any():
        scale[~nonzero] = scale[nonzero][0]
    pink = np.fft.irfft(spectrum * scale, n=n)
    return (pink / np.std(pink)).astype(np.float32)


def _measure(result, name):
    return next(m.value for m in result.measurements if m.name == name)


def test_harmonic_resonance_passes_on_clean_sine_tone():
    # F005 is measurement-only (always PASS, see class docstring): validation
    # against the real labeled corpus (test_data/audio_signal_health/) found
    # it fired *less* on real broken-mic recordings than on normal ones when
    # it gated status, so the owner demoted it permanently. This test just
    # confirms the thd_ratio measurement stays low for an undistorted tone.
    check = HarmonicResonanceCheck()
    assert check.category is CheckCategory.PRIMARY
    x = _sine(1000.0)
    result = check.run(_win(x), _feats(x))
    assert result.status is CheckStatus.PASS
    assert _measure(result, "thd_ratio") < 0.05


def test_harmonic_resonance_measures_elevated_thd_on_nonlinear_distortion():
    # A quadratic (second-harmonic) nonlinearity is the standard textbook model
    # of asymmetric analog distortion (e.g. a damaged capsule ringing without
    # hard-clipping). A symmetric tanh soft-clip was tried first and rejected:
    # it only produces odd harmonics, which plateau at thd_ratio ~0.15
    # (limited further by harmonic_count=5 only reaching the 3rd/5th) no
    # matter how hard it's driven. This quadratic fixture reliably lands
    # around thd_ratio ~0.8 -- status is always PASS (measurement-only), but
    # the elevated thd_ratio measurement is still expected to show through.
    x = _sine(1000.0, amp=1.0)
    y = x + 1.8 * x**2
    y = y - y.mean()
    result = HarmonicResonanceCheck().run(_win(y), _feats(y))
    assert result.status is CheckStatus.PASS
    assert _measure(result, "thd_ratio") > 0.5


def test_harmonic_resonance_passes_while_electrical_hum_warns_on_same_signal():
    # F005 always PASSes now (measurement-only), but this is still a useful
    # regression: F004 and F005 stay self-disambiguating on a pure mains-hum
    # signal (60/120/180 Hz, no other tonal content) -- F004 WARNs on it,
    # while F005's candidate band excludes 60 Hz (below f0_min=80) and masks
    # out the 120/180 Hz hum harmonics via hum_exclusion_bw, leaving no real
    # candidate peak, so its measurements stay near-zero rather than crashing
    # or reporting something misleading.
    x = _sine(60.0, amp=0.3) + _sine(120.0, amp=0.2) + _sine(180.0, amp=0.1)
    hum_result = ElectricalHumCheck(fundamental_frequency=60.0).run(_win(x), _feats(x))
    resonance_result = HarmonicResonanceCheck().run(_win(x), _feats(x))
    assert hum_result.status is CheckStatus.WARNING
    assert resonance_result.status is CheckStatus.PASS
    assert _measure(resonance_result, "thd_ratio") < 1e-6


def test_harmonic_resonance_passes_on_broadband_noise():
    # Broadband noise has no discernible tone. This is the check's
    # prominence_k=50 default at work -- see the class docstring for why 50
    # (not a naive "3x") is needed: f0 is chosen via argmax over ~9,800
    # candidate bins, so by extreme-value statistics the tallest of that many
    # roughly-i.i.d. noise bins sits many multiples above a narrow local
    # median even with no real tone present. Status is always PASS
    # (measurement-only); thd_ratio should still stay at 0 here.
    noise = RNG.normal(size=N)
    result = HarmonicResonanceCheck().run(_win(noise), _feats(noise))
    assert result.status is CheckStatus.PASS
    assert _measure(result, "thd_ratio") == 0.0


def test_harmonic_resonance_passes_on_silence():
    result = HarmonicResonanceCheck().run(_win(np.zeros(N)), _feats(np.zeros(N)))
    assert result.status is CheckStatus.PASS
    assert _measure(result, "thd_ratio") == 0.0
    assert _measure(result, "resonance_score") == 0.0


def test_harmonic_resonance_passes_on_non_finite_input():
    # A NaN-filled window produces a non-finite spectrum total that
    # _usable_spectrum rejects (returns None), matching the F001-F004
    # silence-handling early-out; the check itself doesn't touch window
    # samples directly, only the prepared spectrum.
    x = np.full(N, np.nan, dtype=np.float32)
    result = HarmonicResonanceCheck().run(_win(x), _feats(x))
    assert result.status is CheckStatus.PASS


def test_harmonic_resonance_passes_on_pink_noise_floor():
    # F005 is permanently measurement-only (see class docstring): real-corpus
    # validation showed it fired *less* on genuine broken-mic recordings than
    # on normal ones when it gated status, so status gating was removed
    # entirely rather than chasing threshold tuning against non-flat
    # real-world spectra. This test (previously an xfail pinned to a known
    # local_floor false-positive on non-flat/pink noise) now just confirms
    # F005 still runs cleanly -- without crashing and without misleading
    # status -- on a non-flat (1/f) noise floor, which the flat white/uniform
    # noise fixture above doesn't exercise.
    pink = _pink_noise(seed=0)
    result = HarmonicResonanceCheck().run(_win(pink), _feats(pink))
    assert result.status is CheckStatus.PASS
