from app.decision.threshold import (
    DEFAULT_CUTOFF,
    EwmaPeakDecision,
    ThresholdConfig,
    default_config,
)


def test_default_config_falls_back_when_no_calibration_file(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.json")
    config = default_config(threshold_path=missing_path)
    assert config.cutoff == DEFAULT_CUTOFF


def test_default_config_loads_calibration_file(tmp_path):
    path = tmp_path / "decision_threshold.json"
    path.write_text(ThresholdConfig(cutoff=0.42, span=3.0).to_json())
    config = default_config(threshold_path=str(path))
    assert config.cutoff == 0.42
    assert config.span == 3.0


def test_threshold_config_json_roundtrip():
    cfg = ThresholdConfig(cutoff=0.6, span=5.0)
    restored = ThresholdConfig.from_json(cfg.to_json())
    assert restored.cutoff == cfg.cutoff
    assert restored.span == cfg.span


def test_ewma_peak_decision_high_scores_infested():
    decision = EwmaPeakDecision(ThresholdConfig(cutoff=0.5, span=5.0))
    for _ in range(20):
        decision.update(0.95)
    assert decision.state == "INFESTED"


def test_ewma_peak_decision_low_scores_healthy():
    decision = EwmaPeakDecision(ThresholdConfig(cutoff=0.5, span=5.0))
    for _ in range(20):
        decision.update(0.05)
    assert decision.state == "HEALTHY"


def test_ewma_peak_decision_never_abstains():
    decision = EwmaPeakDecision(ThresholdConfig(cutoff=0.5, span=5.0))
    decision.update(0.5)  # exactly at the boundary
    assert decision.state in ("HEALTHY", "INFESTED")


def test_ewma_peak_decision_tracks_peak_not_final_value():
    decision = EwmaPeakDecision(ThresholdConfig(cutoff=0.5, span=5.0))
    for s in [0.9, 0.9, 0.9, 0.1, 0.1, 0.1]:
        decision.update(s)
    # A late drop shouldn't erase evidence of an earlier spike.
    assert decision.state == "INFESTED"
    assert decision.peak > 0.5


def test_ewma_peak_decision_matches_batch_ewma_peak_score():
    from app.decision.baselines import ewma_peak_score

    scores = [0.2, 0.4, 0.9, 0.3, 0.1, 0.7, 0.2]
    decision = EwmaPeakDecision(ThresholdConfig(cutoff=0.5, span=5.0))
    for s in scores:
        decision.update(s)
    assert decision.peak == ewma_peak_score(scores, span=5.0)
