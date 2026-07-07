"""Tests for app/health/rootcause.py — root-cause attribution (plan sub-phase 8c).

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
    DEFAULT_TIE_MARGIN,
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


def _t002(status, rms):
    return _result("T002", status, measurements=[Measurement("rms", rms)])


# ---------------------------------------------------------------------------
# Basic bucket attribution
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


def test_t001_fail_alone_is_cable():
    results = [_result("T001", CheckStatus.FAIL)]
    outcome = assess_results(results)
    assert outcome.primary_cause is RootCause.CABLE
    assert "T001" in outcome.contributing_check_ids


def test_s002_warning_alone_is_microphone():
    # F005 can no longer serve as this example: it's permanently measurement-only
    # (demoted after real-corpus validation showed it fires backwards) and never
    # produces a non-PASS SignalCheckResult, so S002 (spectral centroid drift)
    # stands in as the "MICROPHONE alone" case instead.
    results = [_result("S002", CheckStatus.WARNING)]
    outcome = assess_results(results)
    assert outcome.primary_cause is RootCause.MICROPHONE
    assert "S002" in outcome.contributing_check_ids


def test_f004_warning_alone_is_environment():
    results = [_result("F004", CheckStatus.WARNING)]
    outcome = assess_results(results)
    assert outcome.primary_cause is RootCause.ENVIRONMENT
    assert "F004" in outcome.contributing_check_ids


def test_t009_fail_alone_is_multiple_by_construction():
    # T009 splits evenly across CABLE (+2) and MICROPHONE (+2): a lone T009 FAIL
    # is an exact tie by construction, and is the natural (not artificially
    # engineered) example of MULTIPLE described in the task brief.
    results = [_result("T009", CheckStatus.FAIL)]
    outcome = assess_results(results)
    assert outcome.primary_cause is RootCause.MULTIPLE
    explanation = outcome.explanation.lower()
    assert "cable" in explanation
    assert "microphone" in explanation


def test_t008_fail_plus_t009_fail_resolves_to_cable_not_multiple():
    # T008 FAIL: CABLE +3. T009 FAIL: CABLE +2, MICROPHONE +2.
    # Total CABLE=5, MICROPHONE=2 -> runner-up/top gap = (5-2)/5 = 0.6, which is
    # outside the default 0.34 tie margin, so this resolves to CABLE.
    t008_alone = assess_results([_result("T008", CheckStatus.FAIL)])
    combined = assess_results(
        [_result("T008", CheckStatus.FAIL), _result("T009", CheckStatus.FAIL)]
    )
    assert t008_alone.primary_cause is RootCause.CABLE
    assert combined.primary_cause is RootCause.CABLE
    # Additivity: the CABLE score (and therefore confidence) with T009 also
    # firing must be strictly higher than T008 alone.
    assert combined.confidence > t008_alone.confidence
    cable_score_alone = next(s for c, s, _ in t008_alone.ranked_causes if c is RootCause.CABLE)
    cable_score_combined = next(s for c, s, _ in combined.ranked_causes if c is RootCause.CABLE)
    assert cable_score_combined > cable_score_alone


def test_t002_fail_direction_dependent_cable_vs_environment():
    low = assess_results([_t002(CheckStatus.FAIL, rms=1e-5)])
    high = assess_results([_t002(CheckStatus.FAIL, rms=0.95)])
    assert low.primary_cause is RootCause.CABLE
    assert high.primary_cause is RootCause.ENVIRONMENT


def test_t004_fail_joint_rule_microphone_vs_environment():
    mic = assess_results(
        [
            _result("T004", CheckStatus.FAIL),
            _t002(CheckStatus.PASS, rms=0.05),
        ]
    )
    env = assess_results(
        [
            _result("T004", CheckStatus.FAIL),
            _t002(CheckStatus.FAIL, rms=0.95),
        ]
    )
    assert mic.primary_cause is RootCause.MICROPHONE
    assert env.primary_cause is RootCause.ENVIRONMENT


def test_unmapped_non_pass_result_yields_unknown():
    # T007 is explicitly excluded from the scoring table.
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
    assert augmented.ranked_causes == plain.ranked_causes
    assert augmented.contributing_check_ids == plain.contributing_check_ids
    # Only the tester-facing explanation may grow a qualifier sentence.
    assert augmented.explanation != plain.explanation
    assert len(augmented.explanation) > len(plain.explanation)


def test_calibration_and_anomaly_absent_by_default_is_unaffected():
    results = [_result("T001", CheckStatus.FAIL)]
    outcome = assess_results(results, calibration_evaluation=None, anomaly_result=None)
    assert outcome.primary_cause is RootCause.CABLE


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
    # Silence trips FlatlineCheck (T001) and SignalEnergyCheck (T002, low-rms
    # direction) at minimum -> CABLE is the expected dominant cause.
    assert outcome.primary_cause is RootCause.CABLE


# ---------------------------------------------------------------------------
# assess_many: a persistent pattern outweighs one noisy window
# ---------------------------------------------------------------------------


class _FakeReport:
    def __init__(self, check_results):
        self.check_results = check_results
        self.calibration_evaluation = None
        self.anomaly_result = None


def test_assess_many_persistent_pattern_beats_single_noisy_window():
    # Deviation from a literal reading of "N mostly-PASS reports plus 1 report
    # with a strong non-PASS signal": if the other N reports are truly all-PASS
    # they contribute zero to every bucket, so the sum trivially collapses to
    # whatever the one non-PASS report's own scoring already is -- that doesn't
    # exercise "a persistent pattern outweighs a single noisy window" at all, it
    # just reproduces the single window's own verdict. Documented as a brief
    # deviation in the task report (the brief explicitly sanctions this: "reason
    # explicitly ... flag as a concern rather than guessing silently").
    #
    # Instead: 10 windows each show a small recurring MICROPHONE-only signal
    # (S002 WARNING, MICROPHONE +1 each -> sums to 10), and exactly one window
    # shows a strong, one-off CABLE signal (T001 FAIL, CABLE +4). The recurring
    # pattern's cumulative total (10) beats the single spike (4), so the
    # combined verdict should still be MICROPHONE, not flipped by the one loud
    # window.
    persistent = [_FakeReport([_result("S002", CheckStatus.WARNING)]) for _ in range(10)]
    noisy_once = _FakeReport([_result("T001", CheckStatus.FAIL)])

    outcome = assess_many(persistent + [noisy_once])
    assert outcome.primary_cause is RootCause.MICROPHONE


def test_assess_many_ignores_reports_with_no_executed_checks():
    not_executed = _FakeReport([_result("T001", CheckStatus.FAIL, executed=False)])
    healthy = _FakeReport([_result("T002", CheckStatus.PASS, measurements=[Measurement("rms", 0.05)])])

    outcome = assess_many([not_executed, healthy])
    assert outcome.primary_cause is RootCause.NONE


def test_assess_many_sums_scores_across_reports():
    reports = [_FakeReport([_result("T008", CheckStatus.WARNING)]) for _ in range(3)]
    outcome = assess_many(reports)
    cable_score = next(s for c, s, _ in outcome.ranked_causes if c is RootCause.CABLE)
    assert cable_score == 6.0  # 3 reports x T008 WARNING (+2 each)


def test_default_tie_margin_constant():
    assert DEFAULT_TIE_MARGIN == 0.34
