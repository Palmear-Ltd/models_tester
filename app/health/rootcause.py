"""Sensor-link fault attribution.

Scores an already-produced set of ``SignalCheckResult``s (or a whole
``HealthReport``, or a multi-window capture of reports) against a static,
transparent weight table to decide whether the problem looks like a fault in
the physical link from the piezo sensor through its cable to the audio jack
-- as opposed to some other signal-health issue this subsystem also tracks
(loud environment, electrical hum, DC bias, general microphone aging, etc.,
which are reported separately by their own checks and are not this module's
concern). This is deterministic, rule-based scoring -- no ML, no learned
weights -- so the weight table below is a plain module-level dict, not a
hidden/computed structure.

Pure stdlib. This module does no signal math itself (it scores results other
checks already computed), so unlike most of ``app/health/`` it needs no NumPy
either. No tkinter/UI imports.

Scope, deliberately narrow (owner decision, 2026-07-07): the tester's actual
hardware is a piezo needle sensor wired through a cable into a jack -- there
is no microphone diaphragm/capsule to distort, and field failures are
overwhelmingly mechanical (rough handling twists/damages the cable, or the
sensor-to-cable joint works loose), which shows up acoustically as either a
total loss of signal or intermittent contact (clicking, brief dropouts) --
never as harmonic distortion. A prior version of this module tried to guess
CABLE vs. MICROPHONE vs. ENVIRONMENT separately; that distinction doesn't
hold up (clicking sounds the same whether it's the cable or an internal
joint) and only added confusing, low-value output. This version reports one
thing: is there a sensor-link problem, yes or no, with a plain explanation.

Additive/optional pattern: ``calibration_evaluation`` and ``anomaly_result``
are optional keyword args (default ``None``) and are never required to
produce a result. When present they may only append a qualifying sentence to
``explanation`` -- they never change ``primary_cause`` or ``confidence``.
This mirrors the established idiom in ``anomaly.py`` / ``calibration_eval.py``:
optional, additive, never state-changing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.health.models import CheckStatus


class RootCause(Enum):
    NONE = "NONE"
    SENSOR_LINK = "SENSOR_LINK"
    UNKNOWN = "UNKNOWN"


@dataclass
class RootCauseAssessment:
    primary_cause: RootCause
    confidence: float  # this module's own 0..1 score, NOT report.confidence
    explanation: str  # one plain-language sentence, tester-facing
    contributing_check_ids: list = field(default_factory=list)


# Heuristic normalizer only -- NOT a probability. Equal to the sum of every
# weight below, i.e. the highest score a single window can produce if every
# contributing check fires at once: T001 FAIL (4) + T008 FAIL (3) +
# T009 FAIL (2) + S004 WARNING (1) = 10.
MAX_SCORE = 10.0

# Static per-(check_id, status) weight table: only checks that indicate a
# broken/intermittent sensor-to-jack link. Each entry maps to (weight, short
# reason).
#
# Explicitly EXCLUDED from this table: every other check in the registry
# (T002-T007, F001-F004, S001-S003). Those report on things unrelated to a
# physical sensor-link fault -- loudness, clipping, DC bias, mains hum,
# spectral drift, general microphone aging. Including them here would
# resurface the confusing cable/mic/environment guessing this module
# deliberately dropped.
_WEIGHT_TABLE: dict = {
    ("T001", CheckStatus.FAIL): (4.0, "there was a complete loss of signal (flatline)"),
    ("T008", CheckStatus.WARNING): (2.0, "a brief dropout occurred mid-recording"),
    ("T008", CheckStatus.FAIL): (3.0, "frequent dropouts occurred during the recording"),
    ("T009", CheckStatus.WARNING): (1.0, "occasional clicking was detected, consistent with an intermittent connection"),
    ("T009", CheckStatus.FAIL): (2.0, "frequent clicking/crackling was detected, consistent with an intermittent or damaged connection"),
    ("S004", CheckStatus.WARNING): (1.0, "the same dropout pattern recurred across several consecutive recordings, not just a one-off glitch"),
}


# ---------------------------------------------------------------------------
# Session-level decision config (Bug 2 fix, spec 2026-07-16): `assess_many`
# used to fire SENSOR_LINK on ANY positive combined score summed across a
# ~40-window session, with confidence saturating via MAX_SCORE (a PER-WINDOW
# cap) applied to that sum. The fix scores a session by its MEAN per-window
# score (combined_score / windows considered) -- dimensionally comparable to
# a single window's score again -- and requires it to clear a persisted,
# data-driven cutoff before declaring SENSOR_LINK, instead of merely being
# positive. Mirrors app/decision/threshold.py's ThresholdConfig/
# default_config() idiom: a frozen JSON file is the source of truth if
# present, with a hardcoded fallback if it's missing.
#
# The cutoff below was fit informally (n=12: the 4 local TN clean reference
# recordings in test_data/F + the 8 confirmed-fault recordings in
# test_data/audio_signal_health/fp/F, replayed through the *new* per-check
# thresholds) -- good enough to stop "saturates on everything," not a
# statistically rigorous cutoff. See docs/superpowers/specs/
# 2026-07-16-rootcause-threshold-recalibration-design.md for the replay
# numbers. Retune by editing rootcause_session_config.json, no code change
# needed.
#
# IMPORTANT: fit against the REAL session windowing (a persistent 2.5s
# buffer, zero-initialized like a freshly started session, rolled+updated
# every 0.5s hop -- see main.py:handle_audio_chunk / offline_score.py's
# score_wav_file(), the documented canonical replication), NOT a bare
# non-overlapping slice of the raw file. Every real session's first ~4
# windows are built from a buffer that is mostly/partly exact zero (the
# startup buffer hasn't filled with real audio yet), and the corrected
# replay showed this isn't just cosmetic: the hard zero->signal edge inside
# that partially-zero window makes ClickTransientCheck's MAD-based sigma
# (computed from the whole window's sample-diffs, dominated by exact-zero
# diffs in the padded region) collapse to a tiny value, so the ordinary,
# non-anomalous audio in the *real* portion of that same window trips the
# click threshold thousands of times over -- and the zero-padded run itself
# also reads as a genuine dropout to T008/S004. This inflates the mean score
# of EVERY session (clean or faulty) by a roughly constant amount in its
# first few windows, raising the floor this cutoff has to clear. It was
# accounted for empirically here (the cutoff is fit against sessions that
# include this artifact), but the artifact itself is not fixed by this
# change -- if a future recalibration needs a lower cutoff (e.g. once a
# finer-grained calibration profile exists) and can't get clean separation,
# this warmup-window distortion is the first thing to fix, e.g. by warming
# the buffer with the first real audio instead of zeros, or having
# ClickTransientCheck/DropoutSegmentCheck exclude a leading exact-zero run
# from their statistics.
DEFAULT_SESSION_CUTOFF = 0.55

DEFAULT_SESSION_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "rootcause_session_config.json"
)


@dataclass(frozen=True)
class RootCauseSessionConfig:
    cutoff: float = DEFAULT_SESSION_CUTOFF

    def to_json(self) -> str:
        return json.dumps({"method": "mean_score", "cutoff": self.cutoff})

    @staticmethod
    def from_json(text: str) -> "RootCauseSessionConfig":
        data = json.loads(text)
        return RootCauseSessionConfig(cutoff=float(data["cutoff"]))


def default_session_config(path: Optional[str] = None) -> RootCauseSessionConfig:
    """Loads a persisted session-cutoff JSON if present, otherwise falls back to
    the shipped default. Never raises -- a missing/unreadable/malformed file
    just means "use the shipped default," same as
    app/decision/threshold.py's default_config()."""
    resolved = path if path is not None else DEFAULT_SESSION_CONFIG_PATH
    if resolved is not None and os.path.exists(resolved):
        try:
            with open(resolved, "r", encoding="utf-8") as f:
                return RootCauseSessionConfig.from_json(f.read())
        except (OSError, ValueError, KeyError):
            pass
    return RootCauseSessionConfig()


def _score(executed: list):
    """Score one window's worth of executed check results.

    Returns (score, contributors, has_non_pass) where contributors is a list
    of (check_id, weight, reason) for whichever entries fired.
    """
    score = 0.0
    contributors: list = []
    has_non_pass = False

    for r in executed:
        if r.status is CheckStatus.PASS:
            continue
        has_non_pass = True
        entry = _WEIGHT_TABLE.get((r.check_id, r.status))
        if entry is not None:
            weight, reason = entry
            score += weight
            contributors.append((r.check_id, weight, reason))

    return score, contributors, has_non_pass


def _unique_reasons(contributors: list) -> list:
    seen: list = []
    for _, _, reason in contributors:
        if reason not in seen:
            seen.append(reason)
    return seen


def _finalize(
    score: float,
    contributors: list,
    has_non_pass: bool,
    *,
    threshold: float = 0.0,
    calibration_evaluation=None,
    anomaly_result=None,
) -> RootCauseAssessment:
    """`threshold` is the bar `score` must clear (strictly) to declare
    SENSOR_LINK. Single-window callers (`assess_results`/`assess`) leave it at
    the default 0.0 -- any positive single-window score is meaningful on its
    own, MAX_SCORE is defined as exactly that window's own ceiling. Session
    callers (`assess_many`) pass a persisted, data-driven cutoff instead,
    since summed/averaged evidence across many windows needs a floor above
    the noise a genuinely healthy session still produces (see
    RootCauseSessionConfig)."""
    if not has_non_pass:
        # NONE reads as "healthy" -- a confident verdict, not a low-confidence
        # guess -- so confidence is 1.0, not 0.0.
        primary_cause = RootCause.NONE
        confidence = 1.0
        contributing_check_ids: list = []
        explanation = "No signal health issues were detected in this recording; it looks healthy."
    elif score <= threshold:
        # Some check(s) fired, but not enough to indicate a sensor-link fault
        # (either nothing in the weight table fired at all, or -- for a
        # session -- the fired checks didn't clear the session floor) -- an
        # honest hedge.
        primary_cause = RootCause.UNKNOWN
        confidence = 0.0
        contributing_check_ids = []
        explanation = (
            "Some checks flagged an issue, but it does not look like a sensor-link problem. "
            "Keep monitoring for a clearer pattern."
        )
    else:
        primary_cause = RootCause.SENSOR_LINK
        confidence = min(1.0, score / MAX_SCORE)
        contributing_check_ids = sorted({cid for cid, _, _ in contributors})
        reasons = _unique_reasons(contributors)
        body = "; ".join(reasons) if reasons else "multiple related checks flagged issues"
        explanation = f"Likely a sensor/cable link problem: {body}."

    if calibration_evaluation is not None and getattr(calibration_evaluation, "deviations", None):
        explanation += " This also deviates from the device's calibrated baseline."
    if anomaly_result is not None and getattr(anomaly_result, "is_anomalous", False):
        explanation += " This recording's overall statistics are also anomalous compared to the calibration baseline."

    return RootCauseAssessment(
        primary_cause=primary_cause,
        confidence=confidence,
        explanation=explanation,
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
) -> RootCauseAssessment:
    """Core scoring logic for one window's worth of SignalCheckResults."""
    executed = [r for r in results if r.executed]
    score, contributors, has_non_pass = _score(executed)
    return _finalize(
        score,
        contributors,
        has_non_pass,
        calibration_evaluation=calibration_evaluation,
        anomaly_result=anomaly_result,
    )


def assess_many(reports, *, session_config: Optional[RootCauseSessionConfig] = None) -> RootCauseAssessment:
    """Assess a multi-window capture (Validate Acquisition's ~40 windows).

    Sums the score across ALL reports (not just the last window), then
    normalizes by the number of windows considered into a session MEAN score
    -- so one noisy window does not dominate a ~20s capture, but the result
    stays on the same 0..MAX_SCORE scale as a single window's score instead
    of growing without bound as the session gets longer. A persistent
    pattern recurring across many windows still produces a higher mean than
    a single one-off spike (more windows clear the per-window threshold), it
    just no longer saturates confidence at 1.0 on every session regardless of
    true fault rate. The mean must clear a persisted, data-driven session
    cutoff (`session_config`, defaults to `default_session_config()`) to
    declare SENSOR_LINK -- replacing the old "any positive sum" rule. Only
    reports whose checks executed are considered; the rest are skipped
    entirely.
    """
    combined_score = 0.0
    combined_contributors: list = []
    windows_considered = 0
    any_has_non_pass = False
    last_calibration_evaluation = None
    last_anomaly_result = None

    for report in reports:
        results = list(getattr(report, "check_results", []) or [])
        executed = [r for r in results if r.executed]
        if not executed:
            continue
        windows_considered += 1
        score, contributors, has_non_pass = _score(executed)
        if has_non_pass:
            any_has_non_pass = True
        combined_score += score
        combined_contributors.extend(contributors)
        cal = getattr(report, "calibration_evaluation", None)
        if cal is not None:
            last_calibration_evaluation = cal
        anomaly = getattr(report, "anomaly_result", None)
        if anomaly is not None:
            last_anomaly_result = anomaly

    if windows_considered == 0:
        return RootCauseAssessment(
            primary_cause=RootCause.UNKNOWN,
            confidence=0.0,
            explanation="No windows had executed checks to assess.",
            contributing_check_ids=[],
        )

    mean_score = combined_score / windows_considered
    cutoff = (session_config if session_config is not None else default_session_config()).cutoff

    return _finalize(
        mean_score,
        combined_contributors,
        any_has_non_pass,
        threshold=cutoff,
        calibration_evaluation=last_calibration_evaluation,
        anomaly_result=last_anomaly_result,
    )
