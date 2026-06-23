from app.health.fusion import decide
from app.health.models import CheckCategory, CheckStatus, HealthState, SignalCheckResult


def _res(cid, status, category, executed=True, diag=None):
    return SignalCheckResult(
        check_id=cid,
        check_name=cid,
        status=status,
        category=category,
        executed=executed,
        diagnostic_messages=list(diag) if diag else [],
    )


def test_empty_results_is_unknown():
    state, confidence, summary = decide([])
    assert state is HealthState.UNKNOWN
    assert confidence == 0.0
    assert summary == "No signal health checks executed."


def test_fault_summary_includes_all_major_failures():
    # A CRITICAL fail co-occurring with a PRIMARY fail: both must appear.
    rs = [
        _res("T001", CheckStatus.FAIL, CheckCategory.CRITICAL, diag=["flatline dead"]),
        _res("T006", CheckStatus.FAIL, CheckCategory.PRIMARY, diag=["dc offset high"]),
    ]
    state, confidence, summary = decide(rs)
    assert state is HealthState.FAULT
    assert "flatline dead" in summary
    assert "dc offset high" in summary


def test_fault_confidence_reflects_failing_fraction():
    rs = [
        _res("T001", CheckStatus.FAIL, CheckCategory.CRITICAL),
        _res("T003", CheckStatus.PASS, CheckCategory.PRIMARY),
    ]
    state, confidence, _ = decide(rs)
    assert state is HealthState.FAULT
    assert confidence == 0.5


def test_unexecuted_checks_are_excluded():
    # One executed PASS + one unexecuted: OK with confidence over executed only.
    rs = [
        _res("T001", CheckStatus.PASS, CheckCategory.CRITICAL),
        _res("T002", CheckStatus.NOT_EXECUTED, CheckCategory.CRITICAL, executed=False),
    ]
    state, confidence, _ = decide(rs)
    assert state is HealthState.OK
    assert confidence == 1.0


def test_only_unexecuted_is_unknown():
    r = _res("T001", CheckStatus.NOT_EXECUTED, CheckCategory.CRITICAL, executed=False)
    assert decide([r])[0] is HealthState.UNKNOWN


def test_all_pass_is_ok_full_confidence():
    rs = [
        _res("T001", CheckStatus.PASS, CheckCategory.CRITICAL),
        _res("T003", CheckStatus.PASS, CheckCategory.PRIMARY),
    ]
    state, confidence, summary = decide(rs)
    assert state is HealthState.OK
    assert confidence == 1.0


def test_critical_fail_is_fault():
    rs = [
        _res("T001", CheckStatus.FAIL, CheckCategory.CRITICAL, diag=["flatline"]),
        _res("T003", CheckStatus.PASS, CheckCategory.PRIMARY),
    ]
    assert decide(rs)[0] is HealthState.FAULT


def test_two_major_fails_is_fault():
    rs = [
        _res("T003", CheckStatus.FAIL, CheckCategory.PRIMARY),
        _res("T006", CheckStatus.FAIL, CheckCategory.PRIMARY),
    ]
    assert decide(rs)[0] is HealthState.FAULT


def test_single_primary_fail_is_warning():
    rs = [
        _res("T003", CheckStatus.FAIL, CheckCategory.PRIMARY),
        _res("T001", CheckStatus.PASS, CheckCategory.CRITICAL),
    ]
    assert decide(rs)[0] is HealthState.WARNING


def test_supporting_warning_is_warning():
    rs = [
        _res("T005", CheckStatus.WARNING, CheckCategory.SUPPORTING),
        _res("T001", CheckStatus.PASS, CheckCategory.CRITICAL),
    ]
    assert decide(rs)[0] is HealthState.WARNING


def test_summary_includes_diagnostics():
    rs = [_res("T001", CheckStatus.FAIL, CheckCategory.CRITICAL, diag=["Flatline: dead"])]
    state, confidence, summary = decide(rs)
    assert state is HealthState.FAULT
    assert "Flatline" in summary
