from app.health.models import HealthState
from app.health.monitoring import MonitoringEvent, RuntimeMonitor


class _Rep:
    """Minimal HealthReport stand-in (RuntimeMonitor only reads final_state)."""

    def __init__(self, state, summary=""):
        self.final_state = state
        self.diagnostic_summary = summary


def _feed(monitor, states):
    events = []
    for s in states:
        events.extend(monitor.update(_Rep(s)))
    return events


def test_starts_ok():
    assert RuntimeMonitor().runtime_state is HealthState.OK


def test_single_fault_does_not_flap():
    m = RuntimeMonitor(fault_persistence=3)
    _feed(m, [HealthState.FAULT])
    assert m.runtime_state is HealthState.OK


def test_sustained_fault_escalates_and_emits_event():
    m = RuntimeMonitor(fault_persistence=3)
    events = _feed(m, [HealthState.FAULT, HealthState.FAULT, HealthState.FAULT])
    assert m.runtime_state is HealthState.FAULT
    assert any(e.event_type == "fault_detected" for e in events)


def test_transient_warning_does_not_escalate():
    m = RuntimeMonitor(warn_persistence=3)
    _feed(m, [HealthState.OK, HealthState.WARNING, HealthState.OK, HealthState.WARNING])
    assert m.runtime_state is HealthState.OK


def test_sustained_warning_escalates():
    m = RuntimeMonitor(warn_persistence=2)
    events = _feed(m, [HealthState.WARNING, HealthState.WARNING])
    assert m.runtime_state is HealthState.WARNING
    assert any(e.event_type == "warning_detected" for e in events)


def test_recovery_requires_sustained_ok():
    m = RuntimeMonitor(fault_persistence=2, recovery=3)
    _feed(m, [HealthState.FAULT, HealthState.FAULT])  # establish FAULT
    assert m.runtime_state is HealthState.FAULT
    _feed(m, [HealthState.OK, HealthState.OK])  # 2 OK < recovery 3
    assert m.runtime_state is HealthState.FAULT
    events = _feed(m, [HealthState.OK])  # 3rd OK
    assert m.runtime_state is HealthState.OK
    assert any(e.event_type == "fault_cleared" for e in events)


def test_event_carries_state_and_message():
    m = RuntimeMonitor(warn_persistence=1)
    (event,) = _feed(m, [HealthState.WARNING])
    assert isinstance(event, MonitoringEvent)
    assert event.state is HealthState.WARNING
    assert event.message
