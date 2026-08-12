import numpy as np

from app.decision.rms_confidence import (
    DEFAULT_A_HEALTHY,
    DEFAULT_A_INFESTED,
    DEFAULT_B_HEALTHY,
    DEFAULT_B_INFESTED,
    DEFAULT_FEATURE,
    DEFAULT_TIER_EDGES,
    RmsConfidenceConfig,
    default_config,
    estimate,
)


def test_default_config_falls_back_when_no_file(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.json")
    config = default_config(config_path=missing_path)
    assert config.a_infested == DEFAULT_A_INFESTED
    assert config.b_infested == DEFAULT_B_INFESTED
    assert config.a_healthy == DEFAULT_A_HEALTHY
    assert config.b_healthy == DEFAULT_B_HEALTHY
    assert config.feature == DEFAULT_FEATURE
    assert tuple(config.tier_edges) == DEFAULT_TIER_EDGES


def test_default_config_with_no_path_returns_shipped_default():
    config = default_config()
    assert config.a_infested == DEFAULT_A_INFESTED


def test_default_config_loads_fitted_file(tmp_path):
    path = tmp_path / "rms_confidence.json"
    fitted = RmsConfidenceConfig(
        a_infested=-2.0, b_infested=1.0, a_healthy=2.0, b_healthy=-1.0,
        feature="peak_rms", tier_edges=(0.3, 0.7), n_infested_fit=500, n_healthy_fit=400,
    )
    path.write_text(fitted.to_json())
    config = default_config(config_path=str(path))
    assert config.a_infested == -2.0
    assert config.feature == "peak_rms"
    assert tuple(config.tier_edges) == (0.3, 0.7)
    assert config.n_infested_fit == 500
    assert config.n_healthy_fit == 400


def test_json_roundtrip():
    cfg = RmsConfidenceConfig(
        a_infested=-1.5, b_infested=0.5, a_healthy=1.5, b_healthy=-0.5,
        feature="mean_rms", tier_edges=(0.35, 0.65), n_infested_fit=100, n_healthy_fit=50,
    )
    restored = RmsConfidenceConfig.from_json(cfg.to_json())
    assert restored == cfg


def test_from_json_defaults_missing_keys():
    restored = RmsConfidenceConfig.from_json("{}")
    assert restored.a_infested == DEFAULT_A_INFESTED
    assert restored.feature == DEFAULT_FEATURE
    assert restored.n_infested_fit is None


# --- monotonicity -----------------------------------------------------------------
# Sign convention (design doc): a_infested < 0 (louder session -> less confident the
# INFESTED call is right), a_healthy > 0 (quieter session -> less confident the
# HEALTHY call is right).

def test_infested_confidence_decreases_as_rms_increases():
    config = RmsConfidenceConfig()  # shipped defaults
    quiet_prob, _ = estimate("INFESTED", mean_rms=0.01, peak_rms=0.01, config=config)
    loud_prob, _ = estimate("INFESTED", mean_rms=1.0, peak_rms=1.0, config=config)
    assert quiet_prob > loud_prob


def test_healthy_confidence_increases_as_rms_increases():
    config = RmsConfidenceConfig()
    quiet_prob, _ = estimate("HEALTHY", mean_rms=0.01, peak_rms=0.01, config=config)
    loud_prob, _ = estimate("HEALTHY", mean_rms=1.0, peak_rms=1.0, config=config)
    assert loud_prob > quiet_prob


def test_infested_low_rms_yields_high_tier_with_steep_config():
    config = RmsConfidenceConfig(a_infested=-5.0, b_infested=2.0, tier_edges=(0.4, 0.6))
    prob, tier = estimate("INFESTED", mean_rms=0.01, peak_rms=0.01, config=config)
    assert tier == "high"
    assert prob > 0.6


def test_infested_high_rms_yields_low_tier_with_steep_config():
    config = RmsConfidenceConfig(a_infested=-5.0, b_infested=2.0, tier_edges=(0.4, 0.6))
    prob, tier = estimate("INFESTED", mean_rms=2.0, peak_rms=2.0, config=config)
    assert tier == "low"
    assert prob < 0.4


def test_healthy_high_rms_yields_high_tier_with_steep_config():
    config = RmsConfidenceConfig(a_healthy=5.0, b_healthy=-2.0, tier_edges=(0.4, 0.6))
    prob, tier = estimate("HEALTHY", mean_rms=2.0, peak_rms=2.0, config=config)
    assert tier == "high"
    assert prob > 0.6


def test_healthy_low_rms_yields_low_tier_with_steep_config():
    config = RmsConfidenceConfig(a_healthy=5.0, b_healthy=-2.0, tier_edges=(0.4, 0.6))
    prob, tier = estimate("HEALTHY", mean_rms=0.01, peak_rms=0.01, config=config)
    assert tier == "low"
    assert prob < 0.4


# --- feature selection -------------------------------------------------------------

def test_estimate_uses_peak_rms_when_configured():
    config = RmsConfidenceConfig(feature="peak_rms")
    prob_mean_only, _ = estimate("INFESTED", mean_rms=0.01, peak_rms=1.0, config=config)
    prob_peak_only, _ = estimate("INFESTED", mean_rms=1.0, peak_rms=0.01, config=config)
    # feature="peak_rms" should key off peak_rms, not mean_rms.
    assert prob_mean_only < prob_peak_only  # loud peak (first call) -> less confident


def test_estimate_uses_mean_rms_by_default():
    config = RmsConfidenceConfig(feature="mean_rms")
    prob_a, _ = estimate("INFESTED", mean_rms=0.01, peak_rms=1.0, config=config)
    prob_b, _ = estimate("INFESTED", mean_rms=1.0, peak_rms=0.01, config=config)
    assert prob_a > prob_b  # first call has the quiet mean -> more confident


# --- neutral fallback / never raises ------------------------------------------------

def test_unknown_state_returns_neutral():
    config = RmsConfidenceConfig()
    prob, tier = estimate("SUSPICIOUS", mean_rms=0.5, peak_rms=0.5, config=config)
    assert prob == 0.5
    assert tier == "medium"


def test_empty_state_returns_neutral():
    config = RmsConfidenceConfig()
    prob, tier = estimate("", mean_rms=0.5, peak_rms=0.5, config=config)
    assert prob == 0.5
    assert tier == "medium"


def test_zero_rms_does_not_raise_and_is_neutral():
    config = RmsConfidenceConfig()
    prob, tier = estimate("INFESTED", mean_rms=0.0, peak_rms=0.0, config=config)
    assert prob == 0.5
    assert tier == "medium"


def test_negative_rms_does_not_raise_and_is_neutral():
    config = RmsConfidenceConfig()
    prob, tier = estimate("HEALTHY", mean_rms=-1.0, peak_rms=-1.0, config=config)
    assert prob == 0.5
    assert tier == "medium"


def test_nan_rms_does_not_raise_and_is_neutral():
    config = RmsConfidenceConfig()
    prob, tier = estimate("INFESTED", mean_rms=float("nan"), peak_rms=float("nan"), config=config)
    assert prob == 0.5
    assert tier == "medium"


def test_extreme_config_does_not_overflow():
    config = RmsConfidenceConfig(a_infested=-1e6, b_infested=1e6)
    prob, tier = estimate("INFESTED", mean_rms=1e6, peak_rms=1e6, config=config)
    assert 0.0 <= prob <= 1.0
    assert tier in ("low", "medium", "high")


# --- tier edge boundaries ------------------------------------------------------------

def test_tier_edges_partition_probability_space():
    # a=0 collapses the logistic to a flat sigmoid(b): pick b so the probability lands
    # exactly at a known point relative to the configured tier edges.
    config = RmsConfidenceConfig(a_infested=0.0, b_infested=0.0, tier_edges=(0.4, 0.6))
    prob, tier = estimate("INFESTED", mean_rms=1.0, peak_rms=1.0, config=config)
    assert prob == 0.5
    assert tier == "medium"  # 0.4 <= 0.5 < 0.6


def test_probability_exactly_at_low_edge_is_medium_not_low():
    # tier boundaries are half-open on the low edge: prob == low_edge is "medium".
    config = RmsConfidenceConfig(a_healthy=0.0, b_healthy=0.0, tier_edges=(0.5, 0.6))
    prob, tier = estimate("HEALTHY", mean_rms=1.0, peak_rms=1.0, config=config)
    assert prob == 0.5
    assert tier == "medium"


# ---------------------------------------------------------------------------------
# Corpus-replay direction check (skip-gracefully pattern, mirrors
# tests/health/test_rootcause.py's corpus tests): on the local test_data/{T,F} sample,
# mean fitted confidence should be higher for TP than FP within the predicted-INFESTED
# pool, and higher for TN than FN within the predicted-HEALTHY pool. Thin (n=6 total in
# this checkout) -- an existence/direction check per the design doc's own acceptance
# criteria, not a statistical claim. Needs a scored manifest.csv + .score_cache (from
# offline_score.py) and soundfile to read the WAVs for RMS; both are dev-only, gitignored
# artifacts, so this skips entirely in a fresh checkout rather than failing.
# ---------------------------------------------------------------------------------
import csv  # noqa: E402
import json  # noqa: E402
import os as _os  # noqa: E402

import pytest  # noqa: E402

soundfile = pytest.importorskip("soundfile", reason="corpus replay needs soundfile to read WAV files")

from app.decision.threshold import EwmaPeakDecision, default_config as default_decision_config  # noqa: E402

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_MANIFEST_PATH = _os.path.join(_REPO_ROOT, "manifest.csv")
_CORPUS_SR = 44100
_CORPUS_HOP_SEC = 0.5
_CORPUS_WINDOW_SEC = 2.5


def _rms_windows(path):
    """Reproduces scripts/rms_scan.py's exact windowing (2.5s rolling buffer, 0.5s hop,
    held off until the buffer fills once with real audio) inline, mirroring how
    tests/health/test_rootcause.py reproduces main.py's windowing inline rather than
    importing a root-level script into the test suite."""
    data, sr = soundfile.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    assert sr == _CORPUS_SR, f"unexpected sample rate {sr} for {path}"

    block_size = int(_CORPUS_SR * _CORPUS_HOP_SEC)
    buffer_len = int(_CORPUS_SR * _CORPUS_WINDOW_SEC)
    n_hops = len(data) // block_size

    buffer = np.zeros(buffer_len, dtype=np.float32)
    samples_received = 0
    values = []
    for hop in range(n_hops):
        start = hop * block_size
        chunk = data[start:start + block_size]
        buffer = np.roll(buffer, -block_size)
        buffer[-block_size:] = chunk
        samples_received = min(buffer_len, samples_received + block_size)
        if samples_received < buffer_len:
            continue
        values.append(float(np.sqrt(np.mean(buffer.astype(np.float64) ** 2))))
    return values


def _load_manifest_rows_for(subdir):
    rows = []
    with open(_MANIFEST_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["path"].replace("\\", "/").startswith(f"test_data/{subdir}/"):
                rows.append(row)
    return rows


def _session_confidence(row):
    cache_path = _os.path.join(_REPO_ROOT, row["cache_path"])
    with open(cache_path, "r", encoding="utf-8") as f:
        scores = json.load(f)["scores"]
    if not scores:
        return None
    decision = EwmaPeakDecision(default_decision_config())
    for s in scores:
        decision.update(s)
    state = decision.state

    wav_path = _os.path.join(_REPO_ROOT, row["path"])
    rms_values = _rms_windows(wav_path)
    if not rms_values:
        return None
    mean_rms = sum(rms_values) / len(rms_values)
    peak_rms = max(rms_values)

    probability, _tier = estimate(state, mean_rms, peak_rms, default_config())
    return state, row["label"], probability


@pytest.mark.skipif(
    not _os.path.exists(_MANIFEST_PATH),
    reason="manifest.csv not present in this checkout (gitignored dev artifact from offline_score.py)",
)
def test_corpus_replay_test_data_confidence_direction():
    rows = _load_manifest_rows_for("T") + _load_manifest_rows_for("F")
    if not rows:
        pytest.skip("no test_data/{T,F} rows found in manifest.csv")

    outcomes = []
    for row in rows:
        cache_path = _os.path.join(_REPO_ROOT, row["cache_path"])
        wav_path = _os.path.join(_REPO_ROOT, row["path"])
        if not _os.path.exists(cache_path) or not _os.path.exists(wav_path):
            continue
        result = _session_confidence(row)
        if result is not None:
            outcomes.append(result)

    if not outcomes:
        pytest.skip("no test_data/{T,F} sessions had both a score cache and a readable WAV")

    infested_correct = [p for state, label, p in outcomes if state == "INFESTED" and label == "T"]
    infested_wrong = [p for state, label, p in outcomes if state == "INFESTED" and label == "F"]
    healthy_correct = [p for state, label, p in outcomes if state == "HEALTHY" and label == "F"]
    healthy_wrong = [p for state, label, p in outcomes if state == "HEALTHY" and label == "T"]

    # This corpus is thin (n=6 total in this checkout) -- per the design doc's own
    # acceptance criteria this is "informative only... no strong statistical claim,
    # direction check", so a bucket with a single example (pure noise, no averaging) is
    # not asserted on; only compared once each side has at least 2 examples.
    MIN_BUCKET_N = 2
    checked_any = False
    if len(infested_correct) >= MIN_BUCKET_N and len(infested_wrong) >= MIN_BUCKET_N:
        checked_any = True
        assert sum(infested_correct) / len(infested_correct) > sum(infested_wrong) / len(infested_wrong), (
            f"TP confidence not higher than FP confidence: TP={infested_correct} FP={infested_wrong}"
        )
    if len(healthy_correct) >= MIN_BUCKET_N and len(healthy_wrong) >= MIN_BUCKET_N:
        checked_any = True
        assert sum(healthy_correct) / len(healthy_correct) > sum(healthy_wrong) / len(healthy_wrong), (
            f"TN confidence not higher than FN confidence: TN={healthy_correct} FN={healthy_wrong}"
        )
    if not checked_any:
        pytest.skip(
            f"test_data/{{T,F}} too thin (need >={MIN_BUCKET_N} per class in a pool) this run: "
            f"{[(s, l) for s, l, _ in outcomes]}"
        )
