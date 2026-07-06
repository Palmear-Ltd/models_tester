"""Serialize duck-typed health reports to JSON-ready dicts (stdlib only).

File writing lives in the app layer (main.py); this module stays pure and
portable so it is testable under the numpy-only venv.
"""
from __future__ import annotations


def startup_result_to_dict(result) -> dict:
    system = result.system
    signal = result.signal
    return {
        "decision": result.decision.value,
        "summary": result.summary,
        "system": {
            "passed": bool(system.passed),
            "errors": list(system.errors),
            "warnings": list(system.warnings),
        },
        "signal": {
            "total": signal.total,
            "ok": signal.ok,
            "warning": signal.warning,
            "fault": signal.fault,
            "check_failures": dict(signal.check_failures),
        },
    }


def anomaly_event_to_dict(anomaly, *, source, timestamp) -> dict:
    return {
        "timestamp": timestamp,
        "source": source,
        "distance": float(anomaly.distance),
        "threshold": float(anomaly.threshold),
        "is_anomalous": bool(anomaly.is_anomalous),
        "confidence": float(anomaly.confidence),
        "contributors": [[label, float(v)] for label, v in anomaly.contributors],
    }
