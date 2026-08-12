"""Regression guard for main.py's decision wiring, pinned to its exact shipped defaults.

main.py needs tkinter/tflite/sounddevice etc., which per CLAUDE.md aren't guaranteed to
be installed in every environment this test suite runs in ("the GUI ... needs the full
requirements.txt installed first -- so manual GUI checks are the owner's job; Claude
verifies headless"). Most of this file pins the non-GUI pieces main.py actually drives
(the frozen calibration artifacts next to the default model, and
EwmaPeakDecision/rms_confidence.estimate's behavior with those exact values) without
importing main.py at all. The calculate_diagnosis wiring tests at the bottom DO need
main.py itself (there's no non-GUI substitute for its diag_label text-building logic) --
those are gated behind pytest.importorskip so the suite still passes in a minimal
environment, and exercise the real method by calling it unbound against a lightweight
stand-in object rather than constructing a real Tk window.
"""
import json
import os

import pytest

from app.decision.threshold import DEFAULT_CUTOFF, DEFAULT_SPAN, EwmaPeakDecision, default_config
from app.decision.rms_confidence import (
    DEFAULT_A_HEALTHY,
    DEFAULT_A_INFESTED,
    DEFAULT_B_HEALTHY,
    DEFAULT_B_INFESTED,
    DEFAULT_FEATURE,
    DEFAULT_TIER_EDGES,
    RmsConfidenceConfig,
    default_config as default_rms_confidence_config,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN_CALIBRATION_PATH = os.path.join(REPO_ROOT, "models", "9_1_2", "decision_threshold.json")
FROZEN_RMS_CONFIDENCE_PATH = os.path.join(REPO_ROOT, "models", "9_1_2", "rms_confidence.json")


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


def test_frozen_rms_confidence_file_exists_next_to_default_model():
    assert os.path.exists(FROZEN_RMS_CONFIDENCE_PATH)


def test_frozen_rms_confidence_matches_shipped_defaults():
    # Same parity check as test_frozen_calibration_matches_shipped_defaults above, for
    # the RMS verdict-confidence curves: the frozen file next to the default model must
    # match the in-code DEFAULT_* constants, or default_config()'s fallback and its
    # file-backed path would silently diverge.
    with open(FROZEN_RMS_CONFIDENCE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["a_infested"] == DEFAULT_A_INFESTED
    assert data["b_infested"] == DEFAULT_B_INFESTED
    assert data["a_healthy"] == DEFAULT_A_HEALTHY
    assert data["b_healthy"] == DEFAULT_B_HEALTHY
    assert data["feature"] == DEFAULT_FEATURE
    assert tuple(data["tier_edges"]) == DEFAULT_TIER_EDGES


def test_default_rms_confidence_config_from_default_model_dir_matches_frozen_file():
    config = default_rms_confidence_config(config_path=FROZEN_RMS_CONFIDENCE_PATH)
    assert config.a_infested == DEFAULT_A_INFESTED
    assert config.feature == DEFAULT_FEATURE
    assert tuple(config.tier_edges) == DEFAULT_TIER_EDGES


# ---------------------------------------------------------------------------------
# calculate_diagnosis wiring: appends a verdict-confidence trust tier to diag_label's
# text without ever changing predicted_infested. Needs main.py itself -- skipped where
# the full GUI/tflite dependency stack isn't installed.
# ---------------------------------------------------------------------------------
main = pytest.importorskip("main", reason="main.py needs the full GUI/tflite dependency stack")


class _FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _FakeLabel:
    def __init__(self):
        self.text = None
        self.color = None

    def configure(self, text=None, foreground=None):
        if text is not None:
            self.text = text
        if foreground is not None:
            self.color = foreground


def _fake_app(energy_history, decision_scores, rms_confidence_config=None):
    """A minimal stand-in for ModelsTesterApp carrying only what calculate_diagnosis's
    sliding-window branch reads/writes -- no real Tk widgets, so this runs without a
    display."""
    fake = type("FakeApp", (), {})()
    fake.inference_mode_var = _FakeVar("sliding")
    fake.diag_label = _FakeLabel()
    fake.energy_history = list(energy_history)
    fake.rms_confidence_config = rms_confidence_config or RmsConfidenceConfig()
    accumulator = EwmaPeakDecision(default_config())
    for s in decision_scores:
        accumulator.update(s)
    fake.decision_accumulator = accumulator
    return fake


def test_calculate_diagnosis_appends_trust_tier_for_infested_session():
    fake = _fake_app(
        energy_history=[(0.0, 0.05), (0.5, 0.05), (1.0, 0.05)],
        decision_scores=[0.95] * 20,
    )
    predicted = main.ModelsTesterApp.calculate_diagnosis(fake)
    assert predicted is True
    assert fake.diag_label.text.startswith("INFESTED")
    assert "verdict confidence:" in fake.diag_label.text
    assert fake.verdict_confidence_tier in ("low", "medium", "high")
    assert fake.verdict_confidence_probability is not None


def test_calculate_diagnosis_appends_trust_tier_for_healthy_session():
    fake = _fake_app(
        energy_history=[(0.0, 0.5), (0.5, 0.5), (1.0, 0.5)],
        decision_scores=[0.05] * 20,
    )
    predicted = main.ModelsTesterApp.calculate_diagnosis(fake)
    assert predicted is False
    assert fake.diag_label.text.startswith("HEALTHY")
    assert "verdict confidence:" in fake.diag_label.text


def test_calculate_diagnosis_verdict_unaffected_by_energy_history():
    # Two sessions with identical decision scores (same state/peak) but wildly different
    # RMS traces must produce the SAME predicted_infested -- this feature is additive
    # only and must never influence the verdict itself.
    quiet = _fake_app(energy_history=[(0.0, 0.001)] * 5, decision_scores=[0.9] * 20)
    loud = _fake_app(energy_history=[(0.0, 2.0)] * 5, decision_scores=[0.9] * 20)
    predicted_quiet = main.ModelsTesterApp.calculate_diagnosis(quiet)
    predicted_loud = main.ModelsTesterApp.calculate_diagnosis(loud)
    assert predicted_quiet == predicted_loud
    assert quiet.decision_accumulator.state == loud.decision_accumulator.state
    # ...but the confidence tier is allowed to (and, with the shipped curve, does) differ.


def test_calculate_diagnosis_with_empty_energy_history_does_not_crash():
    fake = _fake_app(energy_history=[], decision_scores=[0.9] * 20)
    predicted = main.ModelsTesterApp.calculate_diagnosis(fake)
    assert predicted is True
    assert fake.diag_label.text.startswith("INFESTED")
    # No RMS trace to condition on -- no trust tier appended, no crash.
    assert "verdict confidence:" not in fake.diag_label.text
    assert fake.verdict_confidence_tier is None
    assert fake.verdict_confidence_probability is None


def test_calculate_diagnosis_infested_high_rms_yields_lower_confidence_than_low_rms():
    # Direction check matching the design doc's evidence: a loud session predicted
    # INFESTED should get a less confident (or equal) trust tier than a quiet one, using
    # the actual shipped curve (a_infested < 0).
    steep_config = RmsConfidenceConfig(a_infested=-2.0, b_infested=1.0, tier_edges=(0.4, 0.6))
    quiet = _fake_app(energy_history=[(0.0, 0.01)] * 5, decision_scores=[0.9] * 20, rms_confidence_config=steep_config)
    loud = _fake_app(energy_history=[(0.0, 2.0)] * 5, decision_scores=[0.9] * 20, rms_confidence_config=steep_config)
    main.ModelsTesterApp.calculate_diagnosis(quiet)
    main.ModelsTesterApp.calculate_diagnosis(loud)
    assert quiet.verdict_confidence_probability > loud.verdict_confidence_probability
