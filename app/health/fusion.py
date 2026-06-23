"""Decision Fusion — combines Signal Check Results into a final health state.

Phase 0 is minimal: with no registered checks (and therefore no results), the
state is UNKNOWN. Rule-based fusion over real results arrives in Phase 1.
"""
from __future__ import annotations

from app.health.models import HealthState, SignalCheckResult


def decide(results: list[SignalCheckResult]) -> tuple[HealthState, float, str]:
    """Return (final_state, confidence, diagnostic_summary)."""
    if not results:
        return HealthState.UNKNOWN, 0.0, "No signal health checks executed."
    # Phase 0: checks exist but no decision rules yet; report UNKNOWN until Phase 1.
    return HealthState.UNKNOWN, 0.0, "Signal health checks ran; fusion pending (Phase 1)."
