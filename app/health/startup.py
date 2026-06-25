"""Startup Validation (spec Ch. 7): verify system config and aggregate a capture's
HealthReports into a single PASS / WARNING / FAIL decision.

Pure stdlib. Reports are duck-typed: each needs ``final_state`` (a HealthState) and
``check_results`` (items with ``check_id`` and ``status``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.health.models import CheckStatus, HealthState


@dataclass
class SystemValidation:
    passed: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def validate_system(
    *, sample_rate, input_ready, calibration_loaded, expected_sample_rate=44100
) -> SystemValidation:
    errors = []
    warnings = []
    if sample_rate != expected_sample_rate:
        errors.append(f"Sample rate {sample_rate} != expected {expected_sample_rate}")
    if not input_ready:
        errors.append("No input source ready")
    if not calibration_loaded:
        warnings.append("No calibration profile loaded")
    return SystemValidation(passed=not errors, errors=errors, warnings=warnings)


@dataclass
class SignalAggregate:
    total: int
    ok: int
    warning: int
    fault: int
    check_failures: dict = field(default_factory=dict)  # check_id -> non-PASS window count


def aggregate_signal(reports) -> SignalAggregate:
    ok = warning = fault = 0
    check_failures: dict = {}
    for report in reports:
        state = report.final_state
        if state is HealthState.OK:
            ok += 1
        elif state is HealthState.WARNING:
            warning += 1
        else:  # FAULT or UNKNOWN — count as fault for safety
            fault += 1
        for result in report.check_results:
            if result.status is not CheckStatus.PASS:
                check_failures[result.check_id] = check_failures.get(result.check_id, 0) + 1
    return SignalAggregate(
        total=len(reports), ok=ok, warning=warning, fault=fault, check_failures=check_failures
    )


class StartupDecision(Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class StartupResult:
    decision: StartupDecision
    system: SystemValidation
    signal: SignalAggregate
    summary: str


def evaluate_startup(system, signal, *, fault_fraction=0.25) -> StartupResult:
    if not system.passed:
        decision = StartupDecision.FAIL
    elif signal.total == 0:
        decision = StartupDecision.FAIL
    elif signal.fault / signal.total >= fault_fraction:
        decision = StartupDecision.FAIL
    elif (signal.warning + signal.fault) > 0:
        decision = StartupDecision.WARNING
    else:
        decision = StartupDecision.PASS

    parts = [
        f"{signal.ok} OK / {signal.warning} WARNING / {signal.fault} FAULT "
        f"of {signal.total} windows"
    ]
    if system.errors:
        parts.append("system: " + "; ".join(system.errors))
    if system.warnings:
        parts.append("; ".join(system.warnings))
    if signal.check_failures:
        top = sorted(signal.check_failures.items(), key=lambda kv: -kv[1])[:3]
        parts.append("top: " + ", ".join(f"{cid}×{n}" for cid, n in top))
    summary = f"{decision.value}: " + " | ".join(parts)

    return StartupResult(decision=decision, system=system, signal=signal, summary=summary)


def run_validation(
    reports, *, sample_rate, input_ready, calibration_loaded, fault_fraction=0.25
) -> StartupResult:
    system = validate_system(
        sample_rate=sample_rate, input_ready=input_ready, calibration_loaded=calibration_loaded
    )
    signal = aggregate_signal(reports)
    return evaluate_startup(system, signal, fault_fraction=fault_fraction)
