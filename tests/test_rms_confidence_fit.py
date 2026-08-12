"""evaluate_decision_rules.py's RMS verdict-confidence fitting stage
(app/decision/rms_confidence.py's data-fit half). Uses synthetic sessions/RMS traces
throughout -- no dependency on the real corpus, unlike the module-level defaults'
provenance comment.
"""
import csv
import json

import pytest

import evaluate_decision_rules as ev
from app.decision.baselines import ewma_peak
from app.decision.rms_confidence import RmsConfidenceConfig


CUTOFF = 0.5
SPAN = 5.0


def _session(path, label, base_score):
    scores = [base_score + 0.01 * (j % 3) for j in range(36)]
    return ev.SessionRecord(path=path, label=label, month_bucket="jan", scores=scores)


def _write_rms_manifest(tmp_path, rows):
    """rows: list of (path, label, rms_values)."""
    out = tmp_path / "rms_manifest.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "month_bucket", "cache_path", "rms_cache_path", "n_rms_windows", "rms_json"])
        for path, label, values in rows:
            writer.writerow([path, label, "jan", "", "", len(values), json.dumps(values)])
    return str(out)


def test_load_rms_manifest_computes_mean_and_peak(tmp_path):
    manifest_path = _write_rms_manifest(tmp_path, [("a.wav", "T", [0.1, 0.2, 0.3])])
    aggregates = ev.load_rms_manifest(manifest_path)
    assert aggregates["a.wav"]["mean_rms"] == pytest.approx(0.2)
    assert aggregates["a.wav"]["peak_rms"] == 0.3


def test_load_rms_manifest_skips_empty_rms_json(tmp_path):
    manifest_path = _write_rms_manifest(tmp_path, [("empty.wav", "F", [])])
    aggregates = ev.load_rms_manifest(manifest_path)
    assert "empty.wav" not in aggregates


def test_fit_logistic_1d_recovers_separating_sign():
    # Clearly separable: low x -> label 0, high x -> label 1.
    xs = [-3.0, -2.5, -2.0] * 5 + [2.0, 2.5, 3.0] * 5
    ys = [0, 0, 0] * 5 + [1, 1, 1] * 5
    fit = ev.fit_logistic_1d(xs, ys)
    assert fit is not None
    a, b = fit
    assert a > 0  # higher x -> higher P(label==1)


def test_fit_logistic_1d_returns_none_for_single_class():
    assert ev.fit_logistic_1d([1.0, 2.0, 3.0], [1, 1, 1]) is None


def test_rms_pool_examples_buckets_by_predicted_state_and_correctness():
    records = [
        _session("tp.wav", "T", 0.9),  # scores > cutoff -> predicted INFESTED, label T -> correct
        _session("fp.wav", "F", 0.9),  # predicted INFESTED, label F -> incorrect
        _session("tn.wav", "F", 0.1),  # predicted HEALTHY, label F -> correct
        _session("fn.wav", "T", 0.1),  # predicted HEALTHY, label T -> incorrect
    ]
    rms_by_path = {
        "tp.wav": {"mean_rms": 0.05, "peak_rms": 0.05},
        "fp.wav": {"mean_rms": 0.5, "peak_rms": 0.5},
        "tn.wav": {"mean_rms": 0.5, "peak_rms": 0.5},
        "fn.wav": {"mean_rms": 0.05, "peak_rms": 0.05},
    }
    inf_x, inf_y, heal_x, heal_y = ev._rms_pool_examples(records, rms_by_path, CUTOFF, SPAN, "mean_rms")

    # Sanity: predicted states came out as expected given the synthetic scores/cutoff.
    assert ewma_peak(records[0].scores, CUTOFF, span=SPAN).final_state == "INFESTED"
    assert ewma_peak(records[2].scores, CUTOFF, span=SPAN).final_state == "HEALTHY"

    assert len(inf_x) == 2 and set(inf_y) == {0, 1}
    assert len(heal_x) == 2 and set(heal_y) == {0, 1}


def test_fit_rms_confidence_end_to_end_synthetic(tmp_path):
    # Build a corpus where peak_rms cleanly separates correct/incorrect within each
    # predicted-state pool, so the fit + held-out AUC should come out strong.
    records = []
    rms_rows = []
    for i in range(30):
        # INFESTED pool: label T (correct) sessions run quiet, label F (incorrect/FP)
        # sessions run loud -- matches the design doc's real-world direction.
        infested_correct = i % 2 == 0
        label = "T" if infested_correct else "F"
        rms_value = 0.02 if infested_correct else 0.5
        path = f"inf_{i}.wav"
        records.append(_session(path, label, 0.9))  # all score high -> predicted INFESTED
        rms_rows.append((path, label, [rms_value] * 3))

    for i in range(30):
        # HEALTHY pool: label F (correct/TN) sessions run loud, label T (incorrect/FN)
        # sessions run quiet -- matches the design doc's real-world direction (FN
        # sessions are quieter than TN, insufficient signal to register the click
        # pattern).
        healthy_correct = i % 2 == 0
        label = "F" if healthy_correct else "T"
        rms_value = 0.5 if healthy_correct else 0.02
        path = f"heal_{i}.wav"
        records.append(_session(path, label, 0.1))  # all score low -> predicted HEALTHY
        rms_rows.append((path, label, [rms_value] * 3))

    train, test = ev.stratified_split(records, test_frac=0.3, seed=42)
    manifest_path = _write_rms_manifest(tmp_path, rms_rows)

    fitted = ev.fit_rms_confidence(records, train, test, CUTOFF, SPAN, manifest_path)

    assert fitted is not None
    assert fitted["feature"] in ("mean_rms", "peak_rms")
    assert fitted["a_infested"] < 0  # louder INFESTED session -> less confident (matches design sign convention)
    # HEALTHY-pool sign flips depending on which feature won, but should be > 0 either way
    assert fitted["a_healthy"] > 0
    lo, hi = fitted["tier_edges"]
    assert lo < hi
    assert fitted["n_infested_fit"] > 0
    assert fitted["n_healthy_fit"] > 0


def test_fit_rms_confidence_returns_none_when_no_rms_matches(tmp_path):
    records = [_session("a.wav", "T", 0.9), _session("b.wav", "F", 0.1)]
    train, test = ev.stratified_split(records, test_frac=0.5, seed=42)
    manifest_path = _write_rms_manifest(tmp_path, [("unrelated.wav", "T", [0.1, 0.2])])
    fitted = ev.fit_rms_confidence(records, train, test, CUTOFF, SPAN, manifest_path)
    assert fitted is None


def test_write_rms_confidence_config_round_trips(tmp_path):
    out = tmp_path / "rms_confidence.json"
    fitted = {
        "a_infested": -1.5, "b_infested": 0.5,
        "a_healthy": 2.0, "b_healthy": -0.5,
        "feature": "peak_rms", "tier_edges": (0.3, 0.7),
        "n_infested_fit": 100, "n_healthy_fit": 200,
    }
    config = ev.write_rms_confidence_config(str(out), fitted)

    assert isinstance(config, RmsConfidenceConfig)
    loaded = RmsConfidenceConfig.from_json(out.read_text())
    assert loaded.a_infested == -1.5
    assert loaded.feature == "peak_rms"
    assert loaded.tier_edges == (0.3, 0.7)
    assert loaded.n_infested_fit == 100
    assert loaded.n_healthy_fit == 200


def test_best_ewma_cutoff_and_span_is_shared_with_rms_confidence_stage():
    records = []
    for i in range(20):
        infested = i % 2 == 0
        records.append(_session(f"s{i}.wav", "T" if infested else "F", 0.8 if infested else 0.2))
    train, test = ev.stratified_split(records, test_frac=0.3, seed=42)
    candidates = ev.build_candidates(train)
    test_binary = [1 if r.label == "T" else 0 for r in test]
    results = [ev.evaluate_rule(c["name"], c["predict"], c["stat"], test, test_binary) for c in candidates]

    picked = ev._best_ewma_cutoff_and_span(candidates, results)
    assert picked is not None
    cutoff, span = picked
    assert isinstance(cutoff, float)
    assert span in ev.EWMA_SPAN_SWEEP
