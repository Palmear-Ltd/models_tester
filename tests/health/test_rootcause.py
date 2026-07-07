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
    reports = [_FakeReport([_result("T008", CheckStatus.WARNING)]) for _ in range(3)]
    outcome = assess_many(reports)
    assert outcome.primary_cause is RootCause.SENSOR_LINK
    assert outcome.confidence == min(1.0, 6.0 / MAX_SCORE)  # 3 reports x T008 WARNING (+2 each)


def test_assess_many_recurring_pattern_accumulates_confidence():
    # A T009 WARNING recurring across several windows should accumulate a
    # higher score (and therefore confidence) than the same signal appearing
    # in just one window -- confirms assess_many sums across the whole
    # capture rather than only looking at the last/a single window.
    one_window = assess_many([_FakeReport([_result("T009", CheckStatus.WARNING)])])
    three_windows = assess_many(
        [_FakeReport([_result("T009", CheckStatus.WARNING)]) for _ in range(3)]
    )
    assert one_window.primary_cause is RootCause.SENSOR_LINK
    assert three_windows.primary_cause is RootCause.SENSOR_LINK
    assert three_windows.confidence > one_window.confidence


def test_assess_many_no_windows_considered_is_unknown():
    outcome = assess_many([])
    assert outcome.primary_cause is RootCause.UNKNOWN
    assert outcome.contributing_check_ids == []
