"""Regression guard for main.py's decision wiring, pinned to its exact shipped defaults.

main.py itself can't be imported headlessly in this environment (it needs tkinter, which
isn't installed here — GUI checks are the owner's job per CLAUDE.md). This instead pins
the non-GUI pieces main.py actually drives: the frozen calibration artifact next to the
default model, and the EwmaPeakDecision accumulator's behavior with those exact values.
"""
import json
import os

from app.decision.threshold import DEFAULT_CUTOFF, DEFAULT_SPAN, EwmaPeakDecision, default_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_CALIBRATION_PATH = os.path.join(REPO_ROOT, "models", "9_1_2", "decision_threshold.json")


def test_frozen_calibration_file_exists_next_to_default_model():
    assert os.path.exists(FROZEN_CALIBRATION_PATH)


def test_frozen_calibration_matches_shipped_defaults():
    # main.py's default_base_dir is models/9_1_2 (main.py:107) — the frozen file there
    # must match the in-code defaults, or default_config()'s fallback and its
    # file-backed path would silently diverge.
    with open(FROZEN_CALIBRATION_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["cutoff"] == DEFAULT_CUTOFF
    assert data["span"] == DEFAULT_SPAN


def test_default_config_from_default_model_dir_matches_frozen_file():
    config = default_config(threshold_path=FROZEN_CALIBRATION_PATH)
    assert config.cutoff == DEFAULT_CUTOFF
    assert config.span == DEFAULT_SPAN


def test_ewma_peak_decision_with_shipped_defaults_on_a_clear_infested_session():
    decision = EwmaPeakDecision(default_config())
    for _ in range(40):  # a real session is ~40 windows (20s at 0.5s hop)
        decision.update(0.95)
    assert decision.state == "INFESTED"


def test_ewma_peak_decision_with_shipped_defaults_on_a_clear_healthy_session():
    decision = EwmaPeakDecision(default_config())
    for _ in range(40):
        decision.update(0.05)
    assert decision.state == "HEALTHY"
