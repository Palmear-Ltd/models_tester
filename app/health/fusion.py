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


def decide(results: list[SignalCheckResult]) -> tuple[HealthState, float, str]:
    """Return (final_state, confidence, diagnostic_summary)."""
    # `executed` is stamped True by the SignalCheckManager once a check has run
    # (a NOT_EXECUTED result is left executed=False), so this filter == "ran".
    executed = [r for r in results if r.executed]
    if not executed:
        return HealthState.UNKNOWN, 0.0, "No signal health checks executed."

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
        # major_fails is a superset of critical_fails, so it already names every
        # CRITICAL/PRIMARY failure that drove the FAULT (no contributors dropped).
        culprits = major_fails
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

    return state, confidence, summary
