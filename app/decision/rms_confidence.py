"""Session-level verdict confidence: two small logistic curves mapping session RMS
energy to P(the EWMA-peak verdict is correct), conditioned on which way it went.

Design: docs/superpowers/specs/2026-08-09-rms-verdict-confidence-design.md
Plan:   docs/superpowers/plans/2026-08-09-rms-verdict-confidence.md

This is a second, independent trust signal alongside `app/decision/threshold.py`'s
EwmaPeakDecision -- it is purely additive and NEVER changes `state`/`predicted_infested`
(same hard convention `app/health/anomaly.py`'s AnomalyResult.confidence already follows:
"anomaly sets confidence only, never state"). Mirrors `app/decision/threshold.py`'s
ThresholdConfig / default_config() pattern exactly: a tiny data-fit config dataclass with
a JSON load-with-fallback idiom, no bigger model.

Corpus investigation (9_1_4/audio_data/wav/{T,F}, 2021-2026, n=4,775 after excluding the
deliberately-noised 2021 year) found raw/broadband session RMS is a weak *direct*
predictor of ground truth, but a much stronger predictor of whether the model's own
verdict on a session is likely *correct* -- conditioned on which way the model called it:
predicted-INFESTED sessions that run hot (high RMS) are disproportionately false
positives (consistent with the previously root-caused SENSOR_LINK contact-noise
pattern); predicted-HEALTHY sessions that run unusually quiet are disproportionately
false negatives (insufficient signal energy to register the click pattern). See the
design doc's Evidence section for the held-out AUCs this is based on.

    P(correct | INFESTED, rms) = sigmoid(a_infested * log(rms) + b_infested)   # a_infested < 0
    P(correct | HEALTHY, rms)  = sigmoid(a_healthy  * log(rms) + b_healthy)    # a_healthy  > 0

Fit via evaluate_decision_rules.py's --rms-manifest / --rms-confidence-out flags, using
the identical stratified_split(seed=42) train/test discipline already used to fit the
EWMA-peak cutoff, so this signal doesn't leak into whatever a future cutoff refit
evaluates against.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

# ---------------------------------------------------------------------------------
# Shipped defaults -- fit against models/9_1_2 from the full labeled corpus
# (test_data/{T,F} + 9_1_4/audio_data, 6,551 labeled sessions with a matching RMS scan,
# 2021-2026, 2021 NOT excluded here unlike the design doc's own investigatory numbers),
# 70/30 train/test split, seed=42 -- same manifest.csv / stratified_split machinery as
# DEFAULT_CUTOFF/DEFAULT_SPAN in app/decision/threshold.py, and explicitly conditioned on
# THOSE exact shipped cutoff/span values (0.5696557745704712 / 5.0) so the two files that
# ship together in models/9_1_2/ stay mutually consistent -- NOT on whatever cutoff a
# fresh evaluate_decision_rules.py run re-derives for itself (that sweeps EWMA span and,
# on today's grown corpus, now prefers a different span/cutoff than what's actually
# shipped; re-shipping decision_threshold.json is a separate, not-yet-made decision, out
# of scope here).
#
# feature=peak_rms won narrowly over mean_rms on held-out test AUC (0.655 combined vs.
# 0.609): INFESTED-pool (TP vs FP) test AUC 0.614, HEALTHY-pool (TN vs FN) test AUC
# 0.696, n_infested_fit=1472 (train), n_healthy_fit=3114 (train, of which 255 FN / 522
# FP -- both well above the thin-branch threshold used when choosing tier_edges, see
# evaluate_decision_rules.py:THIN_BRANCH_THRESHOLD). Real numbers, real fit -- but AUCs
# in the 0.6-0.7 range are moderate, not strong; treat the resulting tier as a rough,
# informally-validated trust signal (same spirit as this codebase's other "informal, not
# statistically rigorous" fits, e.g. rootcause.py), not a calibrated probability.
# Reproduce via:
#   .venv/bin/python evaluate_decision_rules.py --manifest manifest.csv \
#       --rms-manifest rms_manifest.csv --rms-confidence-out models/9_1_2/rms_confidence.json
# (after generating rms_manifest.csv with scripts/rms_scan.py) -- note the CLI's own
# --rms-confidence-out conditions on whatever cutoff/span *that* run just fit, not
# necessarily the currently-shipped one; the numbers below were produced by calling
# evaluate_decision_rules.fit_rms_confidence() directly with the shipped
# DEFAULT_CUTOFF/DEFAULT_SPAN to guarantee that consistency.
DEFAULT_A_INFESTED = -0.36386305699538923
DEFAULT_B_INFESTED = -1.421263330808008
DEFAULT_A_HEALTHY = 0.5183395813540846
DEFAULT_B_HEALTHY = 5.294206435919703
DEFAULT_FEATURE = "peak_rms"
DEFAULT_TIER_EDGES = (0.7989369755343451, 0.942593693487388)
DEFAULT_N_INFESTED_FIT: Optional[int] = 1472
DEFAULT_N_HEALTHY_FIT: Optional[int] = 3114

_VALID_FEATURES = ("mean_rms", "peak_rms")


@dataclass(frozen=True)
class RmsConfidenceConfig:
    a_infested: float = DEFAULT_A_INFESTED
    b_infested: float = DEFAULT_B_INFESTED
    a_healthy: float = DEFAULT_A_HEALTHY
    b_healthy: float = DEFAULT_B_HEALTHY
    feature: str = DEFAULT_FEATURE  # "mean_rms" or "peak_rms"
    tier_edges: Tuple[float, float] = DEFAULT_TIER_EDGES  # (low/medium edge, medium/high edge)
    # Fit-diagnostic counts (train-split pool sizes), carried through for transparency/
    # logging only -- not used by estimate(). None when unknown (e.g. a hand-built config
    # in a test).
    n_infested_fit: Optional[int] = DEFAULT_N_INFESTED_FIT
    n_healthy_fit: Optional[int] = DEFAULT_N_HEALTHY_FIT

    def to_json(self) -> str:
        return json.dumps(
            {
                "a_infested": self.a_infested,
                "b_infested": self.b_infested,
                "a_healthy": self.a_healthy,
                "b_healthy": self.b_healthy,
                "feature": self.feature,
                "tier_edges": list(self.tier_edges),
                "n_infested_fit": self.n_infested_fit,
                "n_healthy_fit": self.n_healthy_fit,
            }
        )

    @staticmethod
    def from_json(text: str) -> "RmsConfidenceConfig":
        data = json.loads(text)
        tier_edges = data.get("tier_edges", list(DEFAULT_TIER_EDGES))
        return RmsConfidenceConfig(
            a_infested=float(data.get("a_infested", DEFAULT_A_INFESTED)),
            b_infested=float(data.get("b_infested", DEFAULT_B_INFESTED)),
            a_healthy=float(data.get("a_healthy", DEFAULT_A_HEALTHY)),
            b_healthy=float(data.get("b_healthy", DEFAULT_B_HEALTHY)),
            feature=str(data.get("feature", DEFAULT_FEATURE)),
            tier_edges=(float(tier_edges[0]), float(tier_edges[1])),
            # Provenance-only metadata about *this* file's fit -- unlike the coefficients
            # above, a missing key here means "unknown," not "assume the shipped
            # default's fit size," since the a/b this JSON just supplied may not be the
            # shipped fit at all.
            n_infested_fit=data.get("n_infested_fit"),
            n_healthy_fit=data.get("n_healthy_fit"),
        )


def default_config(config_path: Optional[str] = None) -> RmsConfidenceConfig:
    """Loads a fitted rms_confidence.json if present, otherwise falls back to the
    shipped default. Identical fallback-on-missing-file idiom to
    app/decision/threshold.py:default_config -- a fitted curve is specific to the
    model/scaler pair it was fit against, so a different/uncalibrated model at runtime
    should fall back rather than silently use another model's fit."""
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return RmsConfidenceConfig.from_json(f.read())
    return RmsConfidenceConfig()


def _sigmoid(z: float) -> float:
    # Guard against OverflowError on math.exp for pathologically large |z| (e.g. a
    # corrupted/adversarial config, or rms so far outside the fitted range that
    # a*log(rms)+b blows up) -- clamp rather than crash the session-end path.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 700.0)))
    ez = math.exp(max(z, -700.0))
    return ez / (1.0 + ez)


def _tier_for(probability: float, tier_edges: Tuple[float, float]) -> str:
    low_edge, high_edge = tier_edges
    if probability < low_edge:
        return "low"
    if probability < high_edge:
        return "medium"
    return "high"


def estimate(
    state: str,
    mean_rms: float,
    peak_rms: float,
    config: RmsConfidenceConfig,
) -> Tuple[float, str]:
    """Returns (probability, tier) -- P(this session's verdict is correct) and a coarse
    low/medium/high trust tier, per the branch matching `state`.

    Never raises: an unknown `state` (anything other than "INFESTED"/"HEALTHY") or a
    non-finite/non-positive RMS value returns a neutral (0.5, "medium") rather than
    crashing the session-end path -- matching HealthState.UNKNOWN's spirit and
    app/health/anomaly.py's "never take down the additive path" convention."""
    if state == "INFESTED":
        a, b = config.a_infested, config.b_infested
    elif state == "HEALTHY":
        a, b = config.a_healthy, config.b_healthy
    else:
        return 0.5, "medium"

    feature = config.feature if config.feature in _VALID_FEATURES else DEFAULT_FEATURE
    rms_value = mean_rms if feature == "mean_rms" else peak_rms

    if rms_value is None or not math.isfinite(rms_value) or rms_value <= 0.0:
        return 0.5, "medium"

    z = a * math.log(rms_value) + b
    probability = _sigmoid(z)
    tier = _tier_for(probability, config.tier_edges)
    return probability, tier
