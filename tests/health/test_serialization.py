import json
from app.health.serialization import (
    startup_result_to_dict,
    anomaly_event_to_dict,
    root_cause_to_dict,
)
from app.health.rootcause import RootCause, RootCauseAssessment


class _Sys:
    passed = True
    errors = []
    warnings = ["No calibration profile loaded"]


class _Sig:
    total = 40
    ok = 38
    warning = 1
    fault = 1
    check_failures = {"T002": 2}


class _Decision:
    value = "WARNING"


class _Startup:
    decision = _Decision()
    system = _Sys()
    signal = _Sig()
    summary = "WARNING: 38 OK / 1 WARNING / 1 FAULT of 40 windows"


class _Anomaly:
    distance = 5.2
    threshold = 4.0
    is_anomalous = True
    contributors = [("T002.rms", 3.1), ("F001.spectral_centroid", 1.2)]
    confidence = 0.35


def test_startup_result_to_dict_is_json_serializable():
    d = startup_result_to_dict(_Startup())
    assert d["decision"] == "WARNING"
    assert d["signal"]["total"] == 40
    assert d["signal"]["check_failures"] == {"T002": 2}
    assert d["system"]["warnings"] == ["No calibration profile loaded"]
    assert d["summary"].startswith("WARNING")
    json.dumps(d)  # must not raise


def test_anomaly_event_to_dict_is_json_serializable():
    d = anomaly_event_to_dict(_Anomaly(), source="mic", timestamp="20260706_120000")
    assert d["distance"] == 5.2
    assert d["threshold"] == 4.0
    assert d["is_anomalous"] is True
    assert d["confidence"] == 0.35
    assert d["source"] == "mic"
    assert d["timestamp"] == "20260706_120000"
    assert d["contributors"] == [["T002.rms", 3.1], ["F001.spectral_centroid", 1.2]]
    json.dumps(d)  # must not raise


def test_root_cause_to_dict_is_json_serializable():
    assessment = RootCauseAssessment(
        primary_cause=RootCause.CABLE,
        confidence=0.42,
        explanation="Likely a cable problem: there was a complete loss of signal (flatline).",
        ranked_causes=[
            (RootCause.CABLE, 4.0, "there was a complete loss of signal (flatline)"),
            (RootCause.MICROPHONE, 0.0, "no checks currently implicate this cause"),
        ],
        contributing_check_ids=["T001"],
    )
    d = root_cause_to_dict(assessment)
    assert d["primary_cause"] == "CABLE"
    assert d["confidence"] == 0.42
    assert d["explanation"].startswith("Likely a cable problem")
    assert d["ranked_causes"] == [
        ["CABLE", 4.0, "there was a complete loss of signal (flatline)"],
        ["MICROPHONE", 0.0, "no checks currently implicate this cause"],
    ]
    assert d["contributing_check_ids"] == ["T001"]
    json.dumps(d)  # must not raise
