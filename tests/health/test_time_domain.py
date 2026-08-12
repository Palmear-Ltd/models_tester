import numpy as np

from app.health.checks import time_domain
from app.health.checks.time_domain import (
    ClickSpectralMatchCheck,
    ClickTemplateConfig,
    ClickTransientCheck,
    ClippingCheck,
    CrestFactorCheck,
    DCOffsetCheck,
    DropoutSegmentCheck,
    FlatlineCheck,
    PeakAmplitudeCheck,
    SignalEnergyCheck,
    ZeroCrossingRateCheck,
    load_click_template,
)
from app.health.models import AudioWindow, CheckCategory, CheckStatus

SR = 44100
N = 110250  # 2.5 s


def _win(x):
    return AudioWindow(samples=np.asarray(x, dtype=np.float32), sample_rate=SR)


def _sine(freq=1000.0, n=N, amp=0.3):
    t = np.arange(n) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def _measure(result, name):
    return next(m.value for m in result.measurements if m.name == name)


def test_flatline_fails_on_silence():
    check = FlatlineCheck()
    assert check.category is CheckCategory.CRITICAL
    result = check.run(_win(np.zeros(N)), {})
    assert result.status is CheckStatus.FAIL
    assert result.diagnostic_messages


def test_flatline_passes_on_sine():
    result = FlatlineCheck().run(_win(_sine()), {})
    assert result.status is CheckStatus.PASS
    assert _measure(result, "std") > 0.0
    assert _measure(result, "peak_to_peak") > 0.0


def test_flatline_fails_on_nonfinite():
    x = _sine()
    x[100:200] = np.nan
    result = FlatlineCheck().run(_win(x), {})
    assert result.status is CheckStatus.FAIL
    assert result.diagnostic_messages


def test_signal_energy_fails_on_silence():
    result = SignalEnergyCheck().run(_win(np.zeros(N)), {})
    assert result.status is CheckStatus.FAIL


def test_signal_energy_passes_on_sine():
    result = SignalEnergyCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS
    assert _measure(result, "rms") > 0.1


def test_signal_energy_warns_on_low_signal():
    # RMS between min_rms_fault (1e-4) and min_rms_warn (1e-3) -> WARNING.
    result = SignalEnergyCheck().run(_win(np.full(N, 5e-4)), {})
    assert result.status is CheckStatus.WARNING


def test_clipping_fails_when_saturated():
    result = ClippingCheck().run(_win(np.ones(N)), {})
    assert result.status is CheckStatus.FAIL
    assert _measure(result, "clipping_ratio") == 1.0


def test_clipping_warns_on_occasional_clipping():
    # ~0.5% clipped: between warning_ratio (0.001) and fault_ratio (0.01) -> WARNING.
    x = _sine(amp=0.3)
    x[::200] = 1.0  # every 200th sample saturates -> ratio = 1/200 = 0.005
    result = ClippingCheck().run(_win(x), {})
    assert result.status is CheckStatus.WARNING


def test_clipping_passes_on_clean_sine():
    result = ClippingCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS


def test_peak_amplitude_passes_on_sine():
    result = PeakAmplitudeCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS
    assert _measure(result, "peak_amplitude") > 0.25


def test_peak_amplitude_warns_when_too_small():
    result = PeakAmplitudeCheck().run(_win(_sine(amp=1e-4)), {})
    assert result.status is CheckStatus.WARNING


def test_crest_factor_passes_on_sine():
    # A sine has crest factor ~1.41, inside the default [1.2, 50] band.
    result = CrestFactorCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS


def test_crest_factor_category_is_supporting():
    assert CrestFactorCheck().category is CheckCategory.SUPPORTING


def test_dc_offset_fails_on_large_bias():
    result = DCOffsetCheck().run(_win(_sine(amp=0.3) + 0.3), {})
    assert result.status is CheckStatus.FAIL
    assert abs(_measure(result, "dc_offset") - 0.3) < 0.01


def test_dc_offset_passes_on_centered_signal():
    result = DCOffsetCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS


def test_zcr_warns_on_alternating_signal():
    # Sign flips every sample -> ZCR ~1.0, above the default 0.8 warning bound.
    alt = np.tile([0.3, -0.3], N // 2).astype(np.float32)
    result = ZeroCrossingRateCheck().run(_win(alt), {})
    assert result.status is CheckStatus.WARNING


def test_zcr_passes_on_sine():
    result = ZeroCrossingRateCheck().run(_win(_sine(freq=1000.0, amp=0.3)), {})
    assert result.status is CheckStatus.PASS


# T008 DropoutSegmentCheck
# frame_len = int(44100 * 20 / 1000) = 882; N / frame_len = 110250 / 882 = 125 frames exactly.
FRAME_LEN = 882
N_FRAMES = N // FRAME_LEN


def test_dropout_category_is_primary():
    assert DropoutSegmentCheck().category is CheckCategory.PRIMARY


def test_dropout_warns_on_mid_window_gap():
    x = _sine(amp=0.3)
    # Zero out 5 contiguous frames (100 ms) starting well inside the window.
    x[60 * FRAME_LEN : 65 * FRAME_LEN] = 0.0
    result = DropoutSegmentCheck().run(_win(x), {})
    assert result.status is CheckStatus.WARNING
    assert _measure(result, "dropout_event_count") == 1


def test_dropout_passes_on_full_silence():
    result = DropoutSegmentCheck().run(_win(np.zeros(N)), {})
    assert result.status is CheckStatus.PASS
    assert result.measurements == []


def test_dropout_passes_on_clean_sine():
    result = DropoutSegmentCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS


def test_dropout_boundary_run_is_flagged_in_diagnostics():
    x = _sine(amp=0.3)
    # Zero out the first 5 frames -> run touches frame index 0.
    x[: 5 * FRAME_LEN] = 0.0
    result = DropoutSegmentCheck().run(_win(x), {})
    assert result.status is CheckStatus.WARNING
    assert any("boundary" in msg for msg in result.diagnostic_messages)


def test_dropout_fails_when_ratio_exceeds_fault_threshold():
    x = _sine(amp=0.3)
    # 20 of 125 frames dropped -> ratio 0.16 >= fault_ratio (0.15).
    x[: 20 * FRAME_LEN] = 0.0
    result = DropoutSegmentCheck().run(_win(x), {})
    assert result.status is CheckStatus.FAIL
    assert _measure(result, "dropout_frame_ratio") >= 0.15


# T009 ClickTransientCheck
#
# Defaults recalibrated 2026-07-16 (docs/superpowers/specs/
# 2026-07-16-rootcause-threshold-recalibration-design.md): the old warn_count=3/
# fault_count=15 sat well inside the noise floor of genuinely healthy piezo
# contact (a quarter of clean reference windows already cleared WARN). New
# defaults (warn_count=15, fault_count=30) are loaded from
# app/health/check_thresholds.json via app.health.config, NOT the bare class
# constructor default below -- these tests exercise the check directly with
# explicit values so they don't depend on that file's contents.
NEW_WARN_COUNT = 15
NEW_FAULT_COUNT = 30


def _click_burst(n_clicks, n=N, amp=0.3, start=1000, spike=1.0):
    """A sine carrier with `n_clicks` isolated single-sample spikes, spaced far
    enough apart (> merge_gap) to register as that many separate click events."""
    x = _sine(amp=amp, n=n)
    if n_clicks <= 0:
        return x
    spacing = max(4, (n - start - 1) // n_clicks)
    for i in range(n_clicks):
        idx = start + i * spacing
        if idx >= n:
            break
        x[idx] = spike
    return x


def test_click_category_is_primary():
    assert ClickTransientCheck().category is CheckCategory.PRIMARY


def test_click_warns_on_sparse_isolated_spikes():
    x = _sine(amp=0.3)
    for idx in (10000, 30000, 60000, 90000):  # far apart -> 4 separate events
        x[idx] = 1.0
    result = ClickTransientCheck(warn_count=4, fault_count=30).run(_win(x), {})
    assert result.status is CheckStatus.WARNING
    assert _measure(result, "click_count") == 4


def test_click_fails_on_dense_spikes():
    x = _sine(amp=0.3)
    for i in range(35):  # spaced far enough apart to stay separate events
        x[1000 + i * 3000] = 1.0
    result = ClickTransientCheck().run(_win(x), {})
    assert result.status is CheckStatus.FAIL
    assert _measure(result, "click_count") >= NEW_FAULT_COUNT


def test_click_passes_on_full_silence():
    result = ClickTransientCheck().run(_win(np.zeros(N)), {})
    assert result.status is CheckStatus.PASS
    assert result.measurements == []


def test_click_passes_on_clean_sine():
    result = ClickTransientCheck().run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS


# --- recalibration: synthetic click-bursts straddling old (3/15) vs new
# (15/30) thresholds, using the constructor defaults directly. ---


def test_click_passes_below_new_warn_threshold():
    # 10 clicks: PASS under the new default (warn_count=15), but would have
    # been WARNING under the old default (warn_count=3) -- this is the
    # concrete case the recalibration fixes.
    x = _click_burst(10)
    result = ClickTransientCheck().run(_win(x), {})
    assert result.status is CheckStatus.PASS
    assert _measure(result, "click_count") == 10


def test_click_warns_between_new_warn_and_fault_thresholds():
    # 20 clicks: between the new warn_count=15 and fault_count=30 -> WARNING.
    x = _click_burst(20)
    result = ClickTransientCheck().run(_win(x), {})
    assert result.status is CheckStatus.WARNING
    assert _measure(result, "click_count") == 20


def test_click_fails_at_or_above_new_fault_threshold():
    # 35 clicks: at/above the new fault_count=30 -> FAIL.
    x = _click_burst(35)
    result = ClickTransientCheck().run(_win(x), {})
    assert result.status is CheckStatus.FAIL
    assert _measure(result, "click_count") == 35


def test_click_new_defaults_are_15_and_30():
    check = ClickTransientCheck()
    assert check.warn_count == NEW_WARN_COUNT
    assert check.fault_count == NEW_FAULT_COUNT


# ---------------------------------------------------------------------------
# T010 ClickSpectralMatchCheck (docs/superpowers/specs/
# 2026-08-09-rootcause-click-template-refinement-design.md)
#
# Detection re-uses T009's algorithm (independent re-implementation, not a T009
# dependency); the new logic under test is: (1) ClickTemplateConfig loading/fallback
# never raises, (2) a window with too few clicks always PASSes regardless of match
# scores, (3) the match_fraction PASS/WARNING/FAIL banding. For (3) and (2), the real
# FFT/spectral-binning arithmetic (`_click_spectrum`) is already exercised at corpus
# scale by fit_click_template.py's decision-gate run (see rootcause.py's weight-table
# comment for those numbers) -- these unit tests monkeypatch `_click_spectrum` to return
# controlled, deterministic spectra so the check's own aggregation/threshold logic can be
# tested precisely, independent of real audio content.
# ---------------------------------------------------------------------------


def _template_config(n_bins=4, match_threshold=0.5):
    # A simple one-hot template; matching/non-matching fake spectra below are chosen
    # relative to it, not to any real click acoustics.
    template = np.array([1.0, 0.0, 0.0, 0.0])
    return ClickTemplateConfig(
        n_bins=n_bins, freq_max_hz=8000.0, template=template, match_threshold=match_threshold
    )


_MATCHING_SPEC = np.array([1.0, 0.0, 0.0, 0.0])  # dot with template above = 1.0 >= 0.5
_NONMATCHING_SPEC = np.array([0.0, 1.0, 0.0, 0.0])  # dot = 0.0 < 0.5


def _fake_spectrum_cycle(pattern):
    """Returns a fake `_click_spectrum` that yields `pattern` values in order, one per
    call (one call per detected click, in detection order), for monkeypatching."""
    it = iter(pattern)

    def _fake(x, peak_idx, sr, half_ms, n_bins, freq_max):
        return next(it)

    return _fake


def test_click_spectral_match_category_is_primary():
    assert ClickSpectralMatchCheck.category is CheckCategory.PRIMARY


def test_click_spectral_match_passes_on_full_silence():
    check = ClickSpectralMatchCheck(template_config=_template_config())
    result = check.run(_win(np.zeros(N)), {})
    assert result.status is CheckStatus.PASS
    assert result.measurements == []


def test_click_spectral_match_passes_on_clean_sine():
    check = ClickSpectralMatchCheck(template_config=_template_config())
    result = check.run(_win(_sine(amp=0.3)), {})
    assert result.status is CheckStatus.PASS


def test_click_spectral_match_passes_when_template_disabled():
    # ClickTemplateConfig() with no `template=` given is disabled (template=None) -- the
    # missing/malformed-file fallback. Even a dense click burst must always PASS.
    disabled = ClickTemplateConfig()
    assert disabled.enabled is False
    check = ClickSpectralMatchCheck(template_config=disabled)
    x = _click_burst(20)
    result = check.run(_win(x), {})
    assert result.status is CheckStatus.PASS


def test_click_spectral_match_passes_below_min_click_count(monkeypatch):
    # 3 clicks, all "non-matching" -- would be FAIL if scored, but min_click_count=5
    # (default) means there isn't enough evidence to judge, so it must PASS.
    monkeypatch.setattr(time_domain, "_click_spectrum", _fake_spectrum_cycle([_NONMATCHING_SPEC] * 3))
    check = ClickSpectralMatchCheck(template_config=_template_config())
    x = _click_burst(3)
    result = check.run(_win(x), {})
    assert result.status is CheckStatus.PASS
    assert _measure(result, "click_count") == 3
    assert _measure(result, "matched_click_count") == 0.0


def test_click_spectral_match_fails_when_mostly_unmatched(monkeypatch):
    # 10 clicks, 1 matches (fraction 0.1) -- below the default fault_max_fraction (0.2).
    pattern = [_MATCHING_SPEC] + [_NONMATCHING_SPEC] * 9
    monkeypatch.setattr(time_domain, "_click_spectrum", _fake_spectrum_cycle(pattern))
    check = ClickSpectralMatchCheck(template_config=_template_config(), min_click_count=5)
    x = _click_burst(10)
    result = check.run(_win(x), {})
    assert result.status is CheckStatus.FAIL
    assert _measure(result, "matched_click_count") == 1.0
    assert abs(_measure(result, "match_fraction") - 0.1) < 1e-9


def test_click_spectral_match_warns_on_mixed_match(monkeypatch):
    # 10 clicks, 4 match (fraction 0.4) -- between fault_max_fraction (0.2) and
    # warn_max_fraction (0.5).
    pattern = [_MATCHING_SPEC] * 4 + [_NONMATCHING_SPEC] * 6
    monkeypatch.setattr(time_domain, "_click_spectrum", _fake_spectrum_cycle(pattern))
    check = ClickSpectralMatchCheck(template_config=_template_config(), min_click_count=5)
    x = _click_burst(10)
    result = check.run(_win(x), {})
    assert result.status is CheckStatus.WARNING
    assert abs(_measure(result, "match_fraction") - 0.4) < 1e-9


def test_click_spectral_match_passes_when_mostly_matched(monkeypatch):
    # 10 clicks, 9 match (fraction 0.9) -- above warn_max_fraction (0.5).
    pattern = [_MATCHING_SPEC] * 9 + [_NONMATCHING_SPEC]
    monkeypatch.setattr(time_domain, "_click_spectrum", _fake_spectrum_cycle(pattern))
    check = ClickSpectralMatchCheck(template_config=_template_config(), min_click_count=5)
    x = _click_burst(10)
    result = check.run(_win(x), {})
    assert result.status is CheckStatus.PASS
    assert abs(_measure(result, "match_fraction") - 0.9) < 1e-9


def test_click_spectral_match_skips_edge_clicks_gracefully():
    # A click too close to the window edge for a +/-30ms snippet: _click_spectrum
    # returns None for it (real, unpatched code path) -- it must count toward
    # click_count but never toward matched_click_count, and must not raise.
    check = ClickSpectralMatchCheck(template_config=_template_config(), min_click_count=1)
    x = _sine(amp=0.3)
    x[0] = 1.0  # spike at sample 0 -- no room for a snippet before it
    result = check.run(_win(x), {})
    assert result.status in (CheckStatus.PASS, CheckStatus.WARNING, CheckStatus.FAIL)
    assert _measure(result, "click_count") >= 1
    assert _measure(result, "matched_click_count") == 0.0


# --- ClickTemplateConfig / load_click_template: load-JSON-with-fallback idiom ---


def test_click_template_config_disabled_by_default():
    assert ClickTemplateConfig().enabled is False
    assert ClickTemplateConfig().template is None


def test_load_click_template_falls_back_when_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    config = load_click_template(str(missing))
    assert isinstance(config, ClickTemplateConfig)
    assert config.enabled is False


def test_load_click_template_falls_back_on_malformed_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    config = load_click_template(str(bad))
    assert config.enabled is False


def test_load_click_template_falls_back_when_template_length_mismatches_n_bins(tmp_path):
    bad = tmp_path / "mismatched.json"
    bad.write_text(
        '{"n_bins": 40, "freq_max_hz": 8000.0, "template": [1.0, 2.0], "match_threshold": 0.1}',
        encoding="utf-8",
    )
    config = load_click_template(str(bad))
    assert config.enabled is False


def test_load_click_template_round_trips_valid_json(tmp_path):
    path = tmp_path / "click_template.json"
    path.write_text(
        '{"n_bins": 4, "freq_max_hz": 8000.0, "template": [1.0, 0.0, 0.0, 0.0], '
        '"match_threshold": 0.25, "fitted_against": "unit test", "fitted_date": "2026-08-09"}',
        encoding="utf-8",
    )
    config = load_click_template(str(path))
    assert config.enabled is True
    assert config.n_bins == 4
    assert config.freq_max_hz == 8000.0
    assert config.match_threshold == 0.25
    assert np.array_equal(config.template, np.array([1.0, 0.0, 0.0, 0.0]))


def test_load_click_template_reads_shipped_repo_file_if_present():
    # app/health/click_template.json is a fitted data file, not guaranteed present in
    # every checkout (e.g. before it's been generated by fit_click_template.py) -- only
    # assert on its shape/type if it happens to load, never that it must be enabled.
    config = load_click_template()
    assert isinstance(config, ClickTemplateConfig)
    if config.enabled:
        assert config.template.shape == (config.n_bins,)
