"""Runtime Monitoring (spec Ch. 8): debounce the per-window health verdict into a
stable runtime state and emit transition events.

A new state is adopted only after the raw per-window state has persisted for that
state's threshold (recovery windows for OK, warn/fault persistence for the rest),
so transient single-window blips do not flap the reported state. Pure stdlib;
the report is duck-typed (only ``final_state`` is read).
"""
from __future__ import annotations

from dataclasses import dataclass

from app.health.models import HealthState


@dataclass
class MonitoringEvent:
    event_type: str  # fault_detected | warning_detected | fault_cleared | health_restored | signal_lost
    state: HealthState
    message: str


class RuntimeMonitor:
    """Debounces raw per-window verdicts into a stable runtime state."""

    def __init__(self, warn_persistence: int = 3, fault_persistence: int = 3, recovery: int = 5):
        self.warn_persistence = warn_persistence
        self.fault_persistence = fault_persistence
        self.recovery = recovery
        self.runtime_state = HealthState.OK
        self._streak_state = HealthState.OK
        self._streak_count = 0

    def _threshold(self, state: HealthState) -> int:
        return {
            HealthState.OK: self.recovery,
            HealthState.WARNING: self.warn_persistence,
            HealthState.FAULT: self.fault_persistence,
            HealthState.UNKNOWN: 1,
        }[state]

    def update(self, report) -> list[MonitoringEvent]:
        raw = report.final_state
        if raw is self._streak_state:
            self._streak_count += 1
        else:
            self._streak_state = raw
            self._streak_count = 1

        if raw is not self.runtime_state and self._streak_count >= self._threshold(raw):
            previous = self.runtime_state
            self.runtime_state = raw
            return [self._event(previous, raw)]
        return []

    def _event(self, previous: HealthState, new: HealthState) -> MonitoringEvent:
        if new is HealthState.FAULT:
            event_type, message = "fault_detected", "Fault detected"
        elif new is HealthState.WARNING:
            event_type, message = "warning_detected", "Warning detected"
        elif new is HealthState.OK:
            if previous is HealthState.FAULT:
                event_type, message = "fault_cleared", "Fault cleared"
            else:
                event_type, message = "health_restored", "Health restored"
        else:  # UNKNOWN
            event_type, message = "signal_lost", "Signal lost"
        return MonitoringEvent(event_type=event_type, state=new, message=message)
