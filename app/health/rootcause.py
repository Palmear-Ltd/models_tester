"""Root-cause attribution (plan sub-phase 8c).

Scores an already-produced set of ``SignalCheckResult``s (or a whole
``HealthReport``, or a multi-window capture of reports) against a static,
transparent weight table to guess *which hardware/environment component* is
most likely responsible for a signal-health problem: the CABLE, the
MICROPHONE, or the recording ENVIRONMENT. This is deterministic, rule-based
scoring -- no ML, no learned weights -- so the weight table below is a plain
module-level dict, not a hidden/computed structure.

Pure stdlib. This module does no signal math itself (it scores results other
checks already computed), so unlike most of ``app/health/`` it needs no NumPy
either. No tkinter/UI imports.

Design notes carried over from real-corpus validation (see the task brief,
``.superpowers/sdd/cm-task-3-brief.md``, for the full story):

- F005 (HarmonicResonanceCheck) is permanently measurement-only (it fired
  backwards on labeled data) and never produces a non-PASS result, so it has
  no row in the weight table below and contributes nothing to scoring.
- T009 (ClickTransientCheck) splits its weight evenly across CABLE and
  MICROPHONE: clicking/crackling is acoustically ambiguous between a loose
  cable and a damaged capsule/solder joint. Firing alone it produces an exact
  tie (-> MULTIPLE, correctly hedging); firing alongside a CABLE-specific
  signal (T008, T001) tips the balance toward CABLE via simple additivity.

Additive/optional pattern: ``calibration_evaluation`` and ``anomaly_result``
are optional keyword args (default ``None``) and are never required to
produce a result. When present they may only append a qualifying sentence to
``explanation`` -- they never change ``primary_cause``, ``confidence``, or
``ranked_causes``. This mirrors the established idiom in ``anomaly.py`` /
``calibration_eval.py``: optional, additive, never state-changing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.health.models import CheckStatus, SignalCheckResult


class RootCause(Enum):
    NONE = "NONE"
    CABLE = "CABLE"
    MICROPHONE = "MICROPHONE"
    ENVIRONMENT = "ENVIRONMENT"
    MULTIPLE = "MULTIPLE"
    UNKNOWN = "UNKNOWN"


@dataclass
class RootCauseAssessment:
    primary_cause: RootCause
    confidence: float  # this module's own 0..1 score, NOT report.confidence
    explanation: str  # one plain-language sentence/paragraph, tester-facing
    ranked_causes: list  # list[tuple[RootCause, float, str]], up to 3, mirrors anomaly.py's `contributors` shape
    contributing_check_ids: list = field(default_factory=list)


_CAUSE_BUCKETS = (RootCause.CABLE, RootCause.MICROPHONE, RootCause.ENVIRONMENT)

_CAUSE_LABELS = {
    RootCause.CABLE: "a cable problem",
    RootCause.MICROPHONE: "a microphone problem",
    RootCause.ENVIRONMENT: "an environmental factor (not a hardware fault)",
}

# Heuristic normalizer only -- NOT a probability. Chosen as roughly the sum of
# every CABLE-bucket weight that could plausibly fire together in one window:
# T001 FAIL (4) + T008 FAIL (3) + T009 FAIL (2) + S004 WARNING (1) +
# T002 FAIL low-rms (1.5) = 11.5, rounded up to 12 for headroom. CABLE is the
# heaviest bucket in the table, so this keeps the lighter buckets (MICROPHONE,
# ENVIRONMENT) from saturating confidence to 1.0 too easily.
MAX_SCORE = 12.0

# "Runner-up is considered tied with the top bucket if it is within this
# fraction of the top bucket's score" -- i.e. tie iff
# (top - runner_up) / top <= DEFAULT_TIE_MARGIN. At the default 0.34, a lone
# T009 FAIL (CABLE=2, MICROPHONE=2, gap=0 <= 0.34) ties into MULTIPLE, while
# T008 FAIL + T009 FAIL (CABLE=5, MICROPHONE=2, gap=0.6 > 0.34) resolves
# cleanly to CABLE -- see the task brief's worked example.
DEFAULT_TIE_MARGIN = 0.34

# Static per-(check_id, status) weight table. Each entry maps to a tuple of
# (bucket, weight, short reason) -- a tuple because T009 and S003 split their
# weight across two buckets (see module docstring).
#
# S003 (rising noise floor) is genuinely ambiguous between "the microphone is
# degrading" and "the environment got noisier" -- both buckets get half credit
# (0.5 + 0.5) rather than picking one arbitrarily.
#
# Explicitly EXCLUDED from this table (do not add rows for these): T003, T005,
# T007, F001, F002, F003, F005. F005 is measurement-only and never produces a
# non-PASS result (see module docstring). The others are measurement-only or
# explicitly "complementary, not sole indicators" per their own docstrings;
# including them would add noise to a table meant to stay transparent and
# would double-count signal already captured by the stronger checks below.
_WEIGHT_TABLE: dict = {
    ("T001", CheckStatus.FAIL): (
        (RootCause.CABLE, 4.0, "there was a complete loss of signal (flatline)"),
    ),
    ("T008", CheckStatus.WARNING): (
        (RootCause.CABLE, 2.0, "a brief dropout occurred mid-recording"),
    ),
    ("T008", CheckStatus.FAIL): (
        (RootCause.CABLE, 3.0, "frequent dropouts occurred during the recording"),
    ),
    ("T009", CheckStatus.WARNING): (
        (RootCause.CABLE, 1.0, "occasional clicking was detected, which can happen with a loose connection"),
        (RootCause.MICROPHONE, 1.0, "occasional clicking was detected, which can happen with a damaged microphone"),
    ),
    ("T009", CheckStatus.FAIL): (
        (RootCause.CABLE, 2.0, "frequent clicking/crackling was detected, which can happen with a loose or damaged connection"),
        (RootCause.MICROPHONE, 2.0, "frequent clicking/crackling was detected, which can happen with a damaged microphone"),
    ),
    ("S004", CheckStatus.WARNING): (
        (RootCause.CABLE, 1.0, "the same dropout pattern recurred across several consecutive recordings, not just a one-off glitch"),
    ),
    ("S002", CheckStatus.WARNING): (
        (RootCause.MICROPHONE, 1.0, "the recorded frequency content has been drifting over time"),
    ),
    ("S003", CheckStatus.WARNING): (
        (RootCause.MICROPHONE, 0.5, "the background noise floor has been rising over time, which can indicate microphone degradation"),
        (RootCause.ENVIRONMENT, 0.5, "the background noise floor has been rising over time, which can also reflect a noisier environment"),
    ),
    ("F004", CheckStatus.WARNING): (
        (RootCause.ENVIRONMENT, 3.0, "mains electrical hum was detected"),
    ),
    ("T006", CheckStatus.WARNING): (
        (RootCause.ENVIRONMENT, 1.5, "a mild constant DC bias was detected"),
    ),
    ("T006", CheckStatus.FAIL): (
        (RootCause.ENVIRONMENT, 2.5, "a strong constant DC bias was detected"),
    ),
}

# T002 (SignalEnergyCheck) direction-dependent rule: a FAIL can mean "too
# quiet" (CABLE) or "too loud" (ENVIRONMENT), but T002's diagnostic message
# text is identical for both directions ("... outside acceptable range"), so
# direction cannot be told apart from diagnostic_messages alone. Instead we
# read the "rms" Measurement directly (see SignalEnergyCheck in
# checks/time_domain.py). The split point below sits deep inside the gap
# between the default low-RMS fault bound (1e-4) and high-RMS fault bound
# (0.9), so any FAIL is unambiguously on one side of it by construction,
# regardless of the exact threshold overrides the running instance uses.
_T002_DIRECTION_SPLIT = 0.1
_T002_LOW_WEIGHT = 1.5
_T002_HIGH_WEIGHT = 1.5
_T002_LOW_REASON = "the signal was unusually quiet"
_T002_HIGH_REASON = "the signal was unusually loud/saturated"

# T004 (ClippingCheck) x T002 (SignalEnergyCheck) joint rule: clipping without
# a loud input suggests an internal fault (MICROPHONE); clipping together with
# a confirmed loud input suggests the environment itself is too loud
# (ENVIRONMENT), not a hardware fault.
_T004_JOINT_WEIGHT = 2.0
_T004_MIC_REASON = "clipping occurred without a correspondingly loud input signal"
_T004_ENV_REASON = "clipping occurred alongside an unusually loud input signal"


def _t002_direction(result: SignalCheckResult):
    """Classify a T002 FAIL as "low" (too quiet) or "high" (too loud) via its
    "rms" Measurement. Returns None if no rms measurement is present (no
    contribution rather than a guess)."""
    rms = next((m.value for m in result.measurements if m.name == "rms"), None)
    if rms is None:
        return None
    return "low" if rms < _T002_DIRECTION_SPLIT else "high"


def _score_buckets(executed: list):
    """Score one window's worth of executed check results.

    Returns (bucket_scores, bucket_contributors, has_non_pass) where
    bucket_contributors[bucket] is a list of (check_id, weight, reason).
    """
    by_id: dict = {}
    for r in executed:
        by_id.setdefault(r.check_id, r)

    scores = {c: 0.0 for c in _CAUSE_BUCKETS}
    contributors: dict = {c: [] for c in _CAUSE_BUCKETS}
    has_non_pass = False

    for r in executed:
        if r.status is CheckStatus.PASS:
            continue
        has_non_pass = True
        for bucket, weight, reason in _WEIGHT_TABLE.get((r.check_id, r.status), ()):
            scores[bucket] += weight
            contributors[bucket].append((r.check_id, weight, reason))

    t002 = by_id.get("T002")
    if t002 is not None and t002.status is CheckStatus.FAIL:
        direction = _t002_direction(t002)
        if direction == "low":
            scores[RootCause.CABLE] += _T002_LOW_WEIGHT
            contributors[RootCause.CABLE].append(("T002", _T002_LOW_WEIGHT, _T002_LOW_REASON))
        elif direction == "high":
            scores[RootCause.ENVIRONMENT] += _T002_HIGH_WEIGHT
            contributors[RootCause.ENVIRONMENT].append(("T002", _T002_HIGH_WEIGHT, _T002_HIGH_REASON))

    t004 = by_id.get("T004")
    if t004 is not None and t004.status is CheckStatus.FAIL:
        t002_fail_high = (
            t002 is not None
            and t002.status is CheckStatus.FAIL
            and _t002_direction(t002) == "high"
        )
        if t002_fail_high:
            scores[RootCause.ENVIRONMENT] += _T004_JOINT_WEIGHT
            contributors[RootCause.ENVIRONMENT].append(("T004", _T004_JOINT_WEIGHT, _T004_ENV_REASON))
        else:
            scores[RootCause.MICROPHONE] += _T004_JOINT_WEIGHT
            contributors[RootCause.MICROPHONE].append(("T004", _T004_JOINT_WEIGHT, _T004_MIC_REASON))

    return scores, contributors, has_non_pass


def _rank_buckets(scores: dict, contributors: dict) -> list:
    ranked = []
    for cause in _CAUSE_BUCKETS:
        bucket_contributors = contributors[cause]
        if bucket_contributors:
            reason = max(bucket_contributors, key=lambda c: c[1])[2]
        else:
            reason = "no checks currently implicate this cause"
        ranked.append((cause, scores[cause], reason))
    ranked.sort(key=lambda t: t[1], reverse=True)
    return ranked[:3]


def _unique_reasons(contributors: list) -> list:
    seen: list = []
    for _, _, reason in contributors:
        if reason not in seen:
            seen.append(reason)
    return seen


def _single_cause_explanation(cause: RootCause, contributors: list) -> str:
    label = _CAUSE_LABELS[cause]
    reasons = _unique_reasons(contributors)
    body = "; ".join(reasons) if reasons else "multiple related checks flagged issues"
    return f"Likely {label}: {body}."


def _multiple_explanation(cause_a: RootCause, cause_b: RootCause) -> str:
    return (
        f"Could be {_CAUSE_LABELS[cause_a]} or {_CAUSE_LABELS[cause_b]} -- the recording "
        "does not clearly distinguish between them yet. Keep monitoring; if this pattern "
        "repeats consistently across more windows, that will help narrow down which one it is."
    )


def _finalize(
    scores: dict,
    contributors: dict,
    has_non_pass: bool,
    tie_margin: float,
    *,
    calibration_evaluation=None,
    anomaly_result=None,
) -> RootCauseAssessment:
    ranked = _rank_buckets(scores, contributors)

    if not has_non_pass:
        # NONE reads as "healthy" -- a confident verdict, not a low-confidence
        # guess -- so confidence is 1.0, not 0.0.
        primary_cause = RootCause.NONE
        confidence = 1.0
        contributing_check_ids: list = []
        explanation = "No signal health issues were detected in this recording; it looks healthy."
    elif sum(scores.values()) <= 0.0:
        # Some check(s) fired, but none of them map to a cause bucket (e.g.
        # only T007, which is excluded from scoring) -- an honest hedge.
        primary_cause = RootCause.UNKNOWN
        confidence = 0.0
        contributing_check_ids = []
        explanation = (
            "Some checks flagged an issue, but it does not clearly point to a cable, "
            "microphone, or environmental cause. Keep monitoring for a clearer pattern."
        )
    else:
        top_cause, top_score, _ = ranked[0]
        second_cause, second_score, _ = ranked[1]
        gap_ratio = (top_score - second_score) / top_score if top_score > 0 else 0.0
        if gap_ratio <= tie_margin:
            primary_cause = RootCause.MULTIPLE
            confidence = min(1.0, top_score / MAX_SCORE)
            contributing_check_ids = sorted(
                {cid for cid, _, _ in contributors[top_cause] + contributors[second_cause]}
            )
            explanation = _multiple_explanation(top_cause, second_cause)
        else:
            primary_cause = top_cause
            confidence = min(1.0, top_score / MAX_SCORE)
            contributing_check_ids = sorted({cid for cid, _, _ in contributors[top_cause]})
            explanation = _single_cause_explanation(top_cause, contributors[top_cause])

    if calibration_evaluation is not None and getattr(calibration_evaluation, "deviations", None):
        explanation += " This also deviates from the device's calibrated baseline."
    if anomaly_result is not None and getattr(anomaly_result, "is_anomalous", False):
        explanation += " This recording's overall statistics are also anomalous compared to the calibration baseline."

    return RootCauseAssessment(
        primary_cause=primary_cause,
        confidence=confidence,
        explanation=explanation,
        ranked_causes=ranked,
        contributing_check_ids=contributing_check_ids,
    )


def assess(report) -> RootCauseAssessment:
    """Thin wrapper around assess_results for a single HealthReport."""
    return assess_results(
        report.check_results,
        calibration_evaluation=report.calibration_evaluation,
        anomaly_result=report.anomaly_result,
    )


def assess_results(
    results,
    *,
    calibration_evaluation=None,
    anomaly_result=None,
    tie_margin: float = DEFAULT_TIE_MARGIN,
) -> RootCauseAssessment:
    """Core scoring logic for one window's worth of SignalCheckResults."""
    executed = [r for r in results if r.executed]
    scores, contributors, has_non_pass = _score_buckets(executed)
    return _finalize(
        scores,
        contributors,
        has_non_pass,
        tie_margin,
        calibration_evaluation=calibration_evaluation,
        anomaly_result=anomaly_result,
    )


def assess_many(reports, *, tie_margin: float = DEFAULT_TIE_MARGIN) -> RootCauseAssessment:
    """Assess a multi-window capture (Validate Acquisition's ~40 windows).

    Sums the per-bucket scores across ALL reports (not just the last window)
    before ranking, so one noisy window does not dominate a ~20s capture --
    a persistent pattern recurring across many windows accumulates a much
    larger cumulative score than a single one-off spike. Only reports whose
    checks executed are considered; the rest are skipped entirely.
    """
    combined_scores = {c: 0.0 for c in _CAUSE_BUCKETS}
    combined_contributors: dict = {c: [] for c in _CAUSE_BUCKETS}
    any_considered = False
    any_has_non_pass = False
    last_calibration_evaluation = None
    last_anomaly_result = None

    for report in reports:
        results = list(getattr(report, "check_results", []) or [])
        executed = [r for r in results if r.executed]
        if not executed:
            continue
        any_considered = True
        scores, contributors, has_non_pass = _score_buckets(executed)
        if has_non_pass:
            any_has_non_pass = True
        for cause in _CAUSE_BUCKETS:
            combined_scores[cause] += scores[cause]
            combined_contributors[cause].extend(contributors[cause])
        cal = getattr(report, "calibration_evaluation", None)
        if cal is not None:
            last_calibration_evaluation = cal
        anomaly = getattr(report, "anomaly_result", None)
        if anomaly is not None:
            last_anomaly_result = anomaly

    if not any_considered:
        return RootCauseAssessment(
            primary_cause=RootCause.UNKNOWN,
            confidence=0.0,
            explanation="No windows had executed checks to assess.",
            ranked_causes=[],
            contributing_check_ids=[],
        )

    return _finalize(
        combined_scores,
        combined_contributors,
        any_has_non_pass,
        tie_margin,
        calibration_evaluation=last_calibration_evaluation,
        anomaly_result=last_anomaly_result,
    )
