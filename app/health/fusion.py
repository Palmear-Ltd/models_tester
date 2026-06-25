"""Decision Fusion — combines Signal Check Results into a final health state.

Implements the spec's Chapter 10 decision rules:
- any CRITICAL-category check failing, or two or more CRITICAL/PRIMARY checks
  failing, -> FAULT
- any other failure or warning -> WARNING
- all executed checks passing -> OK
- no executed checks -> UNKNOWN

Confidence is a Phase 1 heuristic: the fraction of executed checks that agree
with the chosen state (passes for OK, non-passes for WARNING/FAULT). Note this
is a coverage proxy, NOT a severity measure — a single CRITICAL failure yields a
low number even though it is a high-certainty FAULT. Phase 6 refines this.
"""
from __future__ import annotations

from app.health.models import CheckCategory, CheckStatus, HealthState, SignalCheckResult


_SEVERITY_RANK = {
    HealthState.UNKNOWN: -1,
    HealthState.OK: 0,
    HealthState.WARNING: 1,
    HealthState.FAULT: 2,
}


def _calibration_state(evaluation) -> HealthState:
    """Map a CalibrationEvaluation's deviation tallies to a health state."""
    if evaluation.fault_count >= 2:
        return HealthState.FAULT
    if evaluation.fault_count == 1 or evaluation.warn_count > 0:
        return HealthState.WARNING
    return HealthState.OK


def decide(results, calibration_evaluation=None) -> tuple[HealthState, float, str]:
    """Return (final_state, confidence, diagnostic_summary).

    The check-based decision is computed first; if a calibration evaluation is
    supplied, the final state is the more severe of the two (calibration never
    downgrades a check verdict). Confidence reflects check agreement only — a
    calibration-driven escalation does not alter it (refined in Phase 6).
    """
    # `executed` is stamped True by the SignalCheckManager once a check has run
    # (a NOT_EXECUTED result is left executed=False), so this filter == "ran".
    executed = [r for r in results if r.executed]
    if not executed:
        state, confidence, summary = HealthState.UNKNOWN, 0.0, "No signal health checks executed."
    else:
        fails = [r for r in executed if r.status is CheckStatus.FAIL]
        warns = [r for r in executed if r.status is CheckStatus.WARNING]
        critical_fails = [r for r in fails if r.category is CheckCategory.CRITICAL]
        major_fails = [
            r
            for r in fails
            if r.category in (CheckCategory.CRITICAL, CheckCategory.PRIMARY)
        ]

        if critical_fails or len(major_fails) >= 2:
            state = HealthState.FAULT
            culprits = major_fails  # superset of critical_fails; no contributors dropped
        elif fails or warns:
            state = HealthState.WARNING
            culprits = fails + warns
        else:
            state = HealthState.OK
            culprits = []

        if state is HealthState.OK:
            confidence = 1.0
            summary = f"OK: all {len(executed)} checks passed."
        else:
            confidence = (len(fails) + len(warns)) / len(executed)
            messages = []
            for r in culprits:
                if r.diagnostic_messages:
                    messages.extend(r.diagnostic_messages)
                else:
                    messages.append(f"{r.check_id} {r.status.value}")
            summary = f"{state.value}: " + "; ".join(messages)

    # Fold in calibration evaluation: the more severe verdict wins.
    if calibration_evaluation is not None:
        cal_state = _calibration_state(calibration_evaluation)
        if _SEVERITY_RANK[cal_state] > _SEVERITY_RANK[state]:
            state = cal_state
        if calibration_evaluation.summary:
            summary = (
                f"{summary} | {calibration_evaluation.summary}"
                if summary
                else calibration_evaluation.summary
            )

    return state, confidence, summary
