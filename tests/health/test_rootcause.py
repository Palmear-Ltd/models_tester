"""Tests for app/health/rootcause.py — sensor-link fault attribution.

Constructs SignalCheckResult objects directly for unit-level scoring tests (no
pipeline needed), plus one integration-style test against a real
HealthAnalysisPipeline-produced report (mirrors tests/health/test_pipeline.py).
"""
from __future__ import annotations

import numpy as np

from app.health.models import (
    AudioWindow,
    CheckStatus,
    Measurement,
    SignalCheckResult,
)
from app.health.rootcause import (
    MAX_SCORE,
    RootCause,
    RootCauseAssessment,
    assess,
    assess_many,
    assess_results,
)


def _result(check_id, status, measurements=None, executed=True, diagnostic_messages=None):
    return SignalCheckResult(
        check_id=check_id,
        check_name=check_id,
        status=status,
        executed=executed,
        measurements=measurements or [],
        diagnostic_messages=diagnostic_messages or [],
    )


# ---------------------------------------------------------------------------
# Basic attribution
# ---------------------------------------------------------------------------


def test_all_pass_results_yield_none():
    results = [
        _result("T001", CheckStatus.PASS),
        _result("T002", CheckStatus.PASS, measurements=[Measurement("rms", 0.05)]),
        _result("F004", CheckStatus.PASS),
    ]
    outcome = assess_results(results)
    assert outcome.primary_cause is RootCause.NONE
    assert outcome.confidence == 1.0
    assert outcome.contributing_check_ids == []


def test_t001_fail_alone_is_sensor_link():
    results = [_result("T001", CheckStatus.FAIL)]
    outcome = assess_results(results)
    assert outcome.primary_cause is RootCause.SENSOR_LINK
    assert "T001" in outcome.contributing_check_ids


def test_t008_warning_alone_is_sensor_link():
    results = [_result("T008", CheckStatus.WARNING)]
    outcome = assess_results(results)
    assert outcome.primary_cause is RootCause.SENSOR_LINK
    assert "T008" in outcome.contributing_check_ids


def test_t009_fail_alone_is_sensor_link():
    results = [_result("T009", CheckStatus.FAIL)]
    outcome = assess_results(results)
    assert outcome.primary_cause is RootCause.SENSOR_LINK
    assert "T009" in outcome.contributing_check_ids


def test_s004_warning_alone_is_sensor_link():
    results = [_result("S004", CheckStatus.WARNING)]
    outcome = assess_results(results)
    assert outcome.primary_cause is RootCause.SENSOR_LINK
    assert "S004" in outcome.contributing_check_ids


def test_checks_unrelated_to_sensor_link_do_not_contribute():
    # T002 (energy), T004 (clipping), T006 (DC offset), F004 (hum), S002/S003
    # (spectral drift/noise floor) are about loudness/environment/general
    # microphone aging, not a sensor-link fault -- none should score.
    results = [
        _result("T002", CheckStatus.FAIL, measurements=[Measurement("rms", 0.95)]),
        _result("T004", CheckStatus.FAIL),
        _result("T006", CheckStatus.FAIL),
        _result("F004", CheckStatus.WARNING),
        _result("S002", CheckStatus.WARNING),
        _result("S003", CheckStatus.WARNING),
    ]
    outcome = assess_results(results)
    assert outcome.primary_cause is RootCause.UNKNOWN
    assert outcome.contributing_check_ids == []


def test_t008_fail_plus_t009_fail_is_additive():
    # T008 FAIL contributes 3.0, T009 FAIL contributes 2.0 -- combined score
    # (and therefore confidence) must be strictly higher than either alone.
    t008_alone = assess_results([_result("T008", CheckStatus.FAIL)])
    combined = assess_results(
        [_result("T008", CheckStatus.FAIL), _result("T009", CheckStatus.FAIL)]
    )
    assert t008_alone.primary_cause is RootCause.SENSOR_LINK
    assert combined.primary_cause is RootCause.SENSOR_LINK
    assert combined.confidence > t008_alone.confidence
    assert set(combined.contributing_check_ids) == {"T008", "T009"}


def test_confidence_is_score_over_max_score():
    outcome = assess_results([_result("T001", CheckStatus.FAIL)])
    assert outcome.confidence == 4.0 / MAX_SCORE


def test_unmapped_non_pass_result_yields_unknown():
    # T007 is not in the sensor-link weight table.
    results = [_result("T007", CheckStatus.WARNING)]
    outcome = assess_results(results)
    assert outcome.primary_cause is RootCause.UNKNOWN
    assert outcome.contributing_check_ids == []


# ---------------------------------------------------------------------------
# Calibration / anomaly integration: additive to explanation only
# ---------------------------------------------------------------------------


class _FakeCalibrationEvaluation:
    def __init__(self, deviations):
        self.deviations = deviations
        self.warn_count = len(deviations)
        self.fault_count = 0
        self.summary = "calibration deviations: fake"


class _FakeAnomalyResult:
    def __init__(self, is_anomalous):
        self.is_anomalous = is_anomalous
        self.distance = 5.0
        self.threshold = 2.0
        self.contributors = []
        self.confidence = 0.1


def test_calibration_and_anomaly_never_change_score_only_explanation():
    results = [_result("T001", CheckStatus.FAIL)]

    plain = assess_results(results)
    augmented = assess_results(
        results,
        calibration_evaluation=_FakeCalibrationEvaluation([object()]),
        anomaly_result=_FakeAnomalyResult(is_anomalous=True),
    )

    assert augmented.primary_cause is plain.primary_cause
    assert augmented.confidence == plain.confidence
    assert augmented.contributing_check_ids == plain.contributing_check_ids
    # Only the tester-facing explanation may grow a qualifier sentence.
    assert augmented.explanation != plain.explanation
    assert len(augmented.explanation) > len(plain.explanation)


def test_calibration_and_anomaly_absent_by_default_is_unaffected():
    results = [_result("T001", CheckStatus.FAIL)]
    outcome = assess_results(results, calibration_evaluation=None, anomaly_result=None)
    assert outcome.primary_cause is RootCause.SENSOR_LINK


# ---------------------------------------------------------------------------
# assess(report): integration against a real pipeline
# ---------------------------------------------------------------------------


def test_assess_against_real_pipeline_report():
    from app.health.config import pipeline_for_profile

    sr = 44100
    n = int(2.5 * sr)
    silence = np.zeros(n, dtype=np.float32)
    window = AudioWindow(samples=silence, sample_rate=sr)
    report = pipeline_for_profile("development").analyze(window)

    outcome = assess(report)
    assert isinstance(outcome, RootCauseAssessment)
    # Silence trips FlatlineCheck (T001) -> SENSOR_LINK is the expected cause.
    assert outcome.primary_cause is RootCause.SENSOR_LINK
    assert "T001" in outcome.contributing_check_ids


# ---------------------------------------------------------------------------
# assess_many: scores sum across windows
# ---------------------------------------------------------------------------


class _FakeReport:
    def __init__(self, check_results):
        self.check_results = check_results
        self.calibration_evaluation = None
        self.anomaly_result = None


def test_assess_many_ignores_reports_with_no_executed_checks():
    not_executed = _FakeReport([_result("T001", CheckStatus.FAIL, executed=False)])
    healthy = _FakeReport([_result("T002", CheckStatus.PASS, measurements=[Measurement("rms", 0.05)])])

    outcome = assess_many([not_executed, healthy])
    assert outcome.primary_cause is RootCause.NONE


def test_assess_many_sums_scores_across_reports():
    # 3 windows x T008 WARNING (+2 each) = combined 6.0 over 3 windows
    # considered -> mean_score 2.0, comfortably over the session cutoff.
    # (Pre-recalibration this asserted confidence == min(1, 6.0/MAX_SCORE) --
    # i.e. the RAW SUM over MAX_SCORE. That was exactly the saturation bug:
    # MAX_SCORE is a per-window cap, so summing across many windows against
    # it saturates confidence at 1.0 almost immediately. assess_many now
    # scores the SESSION MEAN (score/window), which stays dimensionally
    # comparable to a single window's score.)
    reports = [_FakeReport([_result("T008", CheckStatus.WARNING)]) for _ in range(3)]
    outcome = assess_many(reports)
    assert outcome.primary_cause is RootCause.SENSOR_LINK
    assert outcome.confidence == min(1.0, 2.0 / MAX_SCORE)


def test_assess_many_no_windows_considered_is_unknown():
    outcome = assess_many([])
    assert outcome.primary_cause is RootCause.UNKNOWN
    assert outcome.contributing_check_ids == []


# ---------------------------------------------------------------------------
# Session-level decision: a persisted, data-driven cutoff replaces "any
# positive combined score" (docs/superpowers/specs/
# 2026-07-16-rootcause-threshold-recalibration-design.md, Bug 2). A ~40-window
# capture is scored by its MEAN per-window score (combined_score / windows
# considered), compared against RootCauseSessionConfig.cutoff -- not by the
# raw sum, which grows without bound as the session gets longer and saturated
# confidence at 1.0 on every recording regardless of true fault rate.
# ---------------------------------------------------------------------------


def _healthy_window():
    return _FakeReport([_result("T002", CheckStatus.PASS, measurements=[Measurement("rms", 0.05)])])


def _isolated_warning_window():
    return _FakeReport([_result("T009", CheckStatus.WARNING)])


def test_assess_many_all_pass_session_is_none():
    session = [_healthy_window() for _ in range(40)]
    outcome = assess_many(session)
    assert outcome.primary_cause is RootCause.NONE


def test_assess_many_occasional_isolated_warning_is_not_sensor_link():
    # A single WARNING window out of a 40-window session -- background piezo
    # noise, not a real link fault. mean_score = 1.0/40 = 0.025, well under
    # the default cutoff -- must NOT resolve SENSOR_LINK (this is exactly the
    # case that saturated confidence to 1.00 before the fix).
    session = [_healthy_window() for _ in range(39)] + [_isolated_warning_window()]
    outcome = assess_many(session)
    assert outcome.primary_cause is not RootCause.SENSOR_LINK
    assert outcome.primary_cause is RootCause.UNKNOWN


def test_assess_many_persistent_fault_pattern_is_sensor_link():
    # A T009 WARNING recurring across most of the session -- a persistent
    # pattern, not a one-off blip. mean_score = 30*1.0/40 = 0.75, clearly over
    # the default cutoff (0.55 -- see DEFAULT_SESSION_CUTOFF's docstring for
    # why it's this high: it's fit against real sessions, whose first ~4
    # windows carry a startup-buffer artifact that inflates every session's
    # floor, clean or faulty; these synthetic FakeReport sessions don't
    # reproduce that artifact, so the bad-window fraction has to clear the
    # same bar on its own).
    session = [_healthy_window() for _ in range(10)] + [
        _isolated_warning_window() for _ in range(30)
    ]
    outcome = assess_many(session)
    assert outcome.primary_cause is RootCause.SENSOR_LINK


def test_assess_many_confidence_scales_with_persistence_within_fixed_session():
    # Same session length (10 windows) in both cases -- only the FRACTION of
    # bad windows differs. A higher recurrence rate must resolve with higher
    # confidence than a low one, and the low-rate case should not even clear
    # the session floor. (Replaces the old
    # test_assess_many_recurring_pattern_accumulates_confidence, which
    # compared sessions of DIFFERENT total length holding the per-window
    # pattern fixed -- i.e. it was asserting the raw-sum saturation bug on
    # purpose. Rate within a fixed-length session is the correct axis.)
    mostly_healthy = [_healthy_window() for _ in range(9)] + [_isolated_warning_window()]
    mostly_faulty = [_healthy_window() for _ in range(3)] + [
        _isolated_warning_window() for _ in range(7)
    ]
    low_rate = assess_many(mostly_healthy)
    high_rate = assess_many(mostly_faulty)
    assert low_rate.primary_cause is not RootCause.SENSOR_LINK
    assert high_rate.primary_cause is RootCause.SENSOR_LINK
    assert high_rate.confidence > low_rate.confidence


# ---------------------------------------------------------------------------
# assess_results / assess (single window): NOT subject to the session-sum
# bug -- a single window's score against MAX_SCORE=10 is already dimensionally
# correct (MAX_SCORE is defined as exactly the highest score one window can
# produce). Confirmed unchanged: any positive single-window score still
# resolves SENSOR_LINK, same as before recalibration.
# ---------------------------------------------------------------------------


def test_assess_results_single_window_any_positive_score_is_still_sensor_link():
    outcome = assess_results([_result("T009", CheckStatus.WARNING)])
    assert outcome.primary_cause is RootCause.SENSOR_LINK
    assert outcome.confidence == min(1.0, 1.0 / MAX_SCORE)


# ---------------------------------------------------------------------------
# RootCauseSessionConfig: persisted cutoff, mirrors app/decision/threshold.py
# ---------------------------------------------------------------------------

from app.health.rootcause import (  # noqa: E402
    RootCauseSessionConfig,
    default_session_config,
)


def test_root_cause_session_config_round_trips_json():
    cfg = RootCauseSessionConfig(cutoff=0.42)
    restored = RootCauseSessionConfig.from_json(cfg.to_json())
    assert restored.cutoff == 0.42


def test_default_session_config_falls_back_when_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    cfg = default_session_config(str(missing))
    assert isinstance(cfg, RootCauseSessionConfig)
    assert cfg.cutoff == RootCauseSessionConfig().cutoff


def test_default_session_config_reads_shipped_repo_file():
    cfg = default_session_config()
    assert isinstance(cfg, RootCauseSessionConfig)
    assert cfg.cutoff > 0.0


def test_default_session_config_reads_custom_file(tmp_path):
    custom = tmp_path / "custom.json"
    custom.write_text('{"method": "mean_score", "cutoff": 0.9}', encoding="utf-8")
    cfg = default_session_config(str(custom))
    assert cfg.cutoff == 0.9


# ---------------------------------------------------------------------------
# Corpus replay: the acceptance check from the design spec. Skipped
# gracefully (not failed) if soundfile or the WAV corpora aren't available in
# this environment/checkout, mirroring how tests/test_calibrate_cli.py and
# tests/decision/test_manifest.py treat WAV-backed fixtures as
# environment-dependent rather than mandatory.
# ---------------------------------------------------------------------------

import glob  # noqa: E402
import math  # noqa: E402
import os as _os  # noqa: E402

import pytest  # noqa: E402

soundfile = pytest.importorskip("soundfile", reason="corpus replay needs soundfile to read WAV files")

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_FP_F_DIR = _os.path.join(_REPO_ROOT, "test_data", "audio_signal_health", "fp", "F")
_TN_F_DIR = _os.path.join(_REPO_ROOT, "test_data", "F")

_CORPUS_SR = 44100
_CORPUS_HOP_SEC = 0.5
_CORPUS_WINDOW_SEC = 2.5


def _load_wav(path):
    data, sr = soundfile.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    assert sr == _CORPUS_SR, f"unexpected sample rate {sr} for {path}"
    return data


def _session_audio_windows(signal, sample_rate=_CORPUS_SR):
    """Reproduces main.py's live sliding-window scheme exactly (handle_audio_chunk,
    main.py:890-901) -- NOT a bare non-overlapping slice of the raw file. A
    persistent buffer is zero-initialized like a freshly started session, then
    updated every 0.5s hop via np.roll (tail hop zero-padded rather than
    dropped), so the first ~4 windows of every real session are built from a
    buffer that's still mostly/partly exact zero. This is also exactly what
    offline_score.py's score_wav_file() does -- its docstring calls it out as
    the documented canonical replication of main.py's windowing, adapted here
    to yield AudioWindow objects for the health pipeline instead of running
    inference. Using a naive non-overlapping slice (as an earlier version of
    this test did, via app.health.calibration.iter_windows) understates the
    session length (36 windows instead of 40 for a 20s file) and skips the
    zero-buffer warmup entirely -- which turned out to matter (see
    DEFAULT_SESSION_CUTOFF's docstring in rootcause.py)."""
    from app.health.models import AudioWindow

    block_size = int(sample_rate * _CORPUS_HOP_SEC)
    buffer_len = int(sample_rate * _CORPUS_WINDOW_SEC)
    total_samples = len(signal)
    n_hops = math.ceil(total_samples / block_size) if total_samples > 0 else 0

    buffer = np.zeros(buffer_len, dtype=np.float32)
    for hop in range(n_hops):
        start = hop * block_size
        end = min(start + block_size, total_samples)
        chunk = signal[start:end]
        if len(chunk) < block_size:
            chunk = np.pad(chunk, (0, block_size - len(chunk)))
        buffer = np.roll(buffer, -block_size)
        buffer[-block_size:] = chunk
        yield AudioWindow(samples=buffer.copy(), sample_rate=sample_rate)


def _session_outcome(path):
    from app.health.config import pipeline_for_profile

    pipeline = pipeline_for_profile("development")  # fresh session, no calibration profile
    signal = _load_wav(path)
    reports = [pipeline.analyze(w) for w in _session_audio_windows(signal)]
    return assess_many(reports)


def _corpus_wavs(directory):
    return sorted(glob.glob(_os.path.join(directory, "*.wav")))


@pytest.mark.skipif(
    not _os.path.isdir(_FP_F_DIR), reason="fp/F fault corpus not present in this checkout"
)
def test_corpus_replay_fp_f_mostly_resolves_sensor_link():
    paths = _corpus_wavs(_FP_F_DIR)
    if not paths:
        pytest.skip("no WAV files found under test_data/audio_signal_health/fp/F")
    outcomes = [_session_outcome(p) for p in paths]
    sensor_link_count = sum(1 for o in outcomes if o.primary_cause is RootCause.SENSOR_LINK)
    # "Mostly" per the spec's acceptance criteria -- this is a real, confirmed
    # fault corpus; don't regress detection of it. n=8, informal per spec.
    assert sensor_link_count >= (len(outcomes) + 1) // 2, (
        f"only {sensor_link_count}/{len(outcomes)} fp/F fault recordings resolved SENSOR_LINK: "
        f"{[(_os.path.basename(p), o.primary_cause.value) for p, o in zip(paths, outcomes)]}"
    )


@pytest.mark.skipif(
    not _os.path.isdir(_TN_F_DIR), reason="test_data/F TN reference corpus not present in this checkout"
)
def test_corpus_replay_tn_f_mostly_not_sensor_link():
    paths = _corpus_wavs(_TN_F_DIR)
    if not paths:
        pytest.skip("no WAV files found under test_data/F")
    outcomes = [_session_outcome(p) for p in paths]
    not_sensor_link_count = sum(1 for o in outcomes if o.primary_cause is not RootCause.SENSOR_LINK)
    assert not_sensor_link_count >= (len(outcomes) + 1) // 2, (
        f"only {not_sensor_link_count}/{len(outcomes)} test_data/F clean recordings resolved "
        f"NOT SENSOR_LINK: {[(_os.path.basename(p), o.primary_cause.value) for p, o in zip(paths, outcomes)]}"
    )
