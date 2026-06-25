from app.health.models import CheckStatus, HealthState
from app.health.startup import (
    SignalAggregate,
    SystemValidation,
    aggregate_signal,
    validate_system,
)


class _Check:
    def __init__(self, check_id, status):
        self.check_id = check_id
        self.status = status


class _Rep:
    def __init__(self, state, checks=()):
        self.final_state = state
        self.check_results = list(checks)


def test_system_ok_when_config_correct():
    v = validate_system(sample_rate=44100, input_ready=True, calibration_loaded=True)
    assert v.passed is True
    assert v.errors == []
    assert v.warnings == []


def test_system_errors_on_wrong_sample_rate_and_no_input():
    v = validate_system(sample_rate=22050, input_ready=False, calibration_loaded=True)
    assert v.passed is False
    assert len(v.errors) == 2


def test_missing_calibration_is_a_warning_not_an_error():
    v = validate_system(sample_rate=44100, input_ready=True, calibration_loaded=False)
    assert v.passed is True
    assert v.errors == []
    assert any("calibration" in w.lower() for w in v.warnings)


def test_aggregate_signal_counts_states_and_check_failures():
    reports = [
        _Rep(HealthState.OK, [_Check("T001", CheckStatus.PASS)]),
        _Rep(HealthState.WARNING, [_Check("T002", CheckStatus.WARNING)]),
        _Rep(HealthState.FAULT, [_Check("T001", CheckStatus.FAIL), _Check("T002", CheckStatus.WARNING)]),
        _Rep(HealthState.UNKNOWN, []),  # counted as fault for safety
    ]
    agg = aggregate_signal(reports)
    assert isinstance(agg, SignalAggregate)
    assert agg.total == 4
    assert agg.ok == 1
    assert agg.warning == 1
    assert agg.fault == 2  # FAULT + UNKNOWN
    assert agg.check_failures == {"T001": 1, "T002": 2}  # non-PASS occurrences


def test_aggregate_empty():
    agg = aggregate_signal([])
    assert agg.total == 0 and agg.ok == 0 and agg.check_failures == {}


from app.health.startup import (  # noqa: E402
    StartupDecision,
    StartupResult,
    evaluate_startup,
    run_validation,
)


def _agg(total, ok, warning, fault):
    return SignalAggregate(total=total, ok=ok, warning=warning, fault=fault, check_failures={})


def _ok_system():
    return SystemValidation(passed=True, errors=[], warnings=[])


def test_pass_when_all_ok():
    r = evaluate_startup(_ok_system(), _agg(40, 40, 0, 0))
    assert r.decision is StartupDecision.PASS
    assert isinstance(r, StartupResult)


def test_warning_when_some_non_ok_below_fault_fraction():
    r = evaluate_startup(_ok_system(), _agg(40, 36, 3, 1))  # 1/40 fault < 0.25
    assert r.decision is StartupDecision.WARNING


def test_fail_when_fault_fraction_exceeded():
    r = evaluate_startup(_ok_system(), _agg(40, 20, 5, 15))  # 15/40 >= 0.25
    assert r.decision is StartupDecision.FAIL


def test_fail_when_system_errors():
    bad = SystemValidation(passed=False, errors=["No input source ready"], warnings=[])
    r = evaluate_startup(bad, _agg(40, 40, 0, 0))
    assert r.decision is StartupDecision.FAIL


def test_fail_when_no_windows():
    r = evaluate_startup(_ok_system(), _agg(0, 0, 0, 0))
    assert r.decision is StartupDecision.FAIL


def test_run_validation_end_to_end():
    reports = [_Rep(HealthState.OK, [_Check("T001", CheckStatus.PASS)]) for _ in range(40)]
    r = run_validation(reports, sample_rate=44100, input_ready=True, calibration_loaded=True)
    assert r.decision is StartupDecision.PASS
    assert r.signal.total == 40
    assert r.summary
