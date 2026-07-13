"""Session-level decision: EWMA-peak score vs. a data-driven cutoff.

Phase A's offline evaluation (see evaluate_decision_rules.py) compared this against the
existing hand-tuned threshold+count-band rule and several more elaborate alternatives
(SPRT evidence accumulation, fixed-sample quantile bands) across ~6,500 labeled sessions.
None of the more elaborate methods clearly beat this one — real recordings here are a
fixed ~20s/40 windows, not open-ended, so sequential evidence accumulation (which wants to
keep sampling until confident) is a structural mismatch; a single EWMA-smoothed peak over
the full fixed window, with a cutoff derived from data instead of hand-picked, matched or
slightly exceeded the old method's accuracy while replacing three manually-tuned numbers
(score_thresh, susp_limit, inf_limit) with one. The cutoff below was cross-validated two
independent ways (ROC-optimal threshold and a collapsed fixed-sample quantile band) that
agreed to within 0.0004 of each other.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

# Fit against models/9_1_2 from the full labeled corpus (test_data/{T,F} +
# 9_1_4/audio_data), 70/30 train/test split, seed=42. See evaluate_decision_rules.py.
DEFAULT_CUTOFF = 0.5696557745704712
DEFAULT_SPAN = 5.0


@dataclass(frozen=True)
class ThresholdConfig:
    cutoff: float = DEFAULT_CUTOFF
    span: float = DEFAULT_SPAN

    def to_json(self) -> str:
        return json.dumps({"method": "ewma_peak", "cutoff": self.cutoff, "span": self.span})

    @staticmethod
    def from_json(text: str) -> "ThresholdConfig":
        data = json.loads(text)
        return ThresholdConfig(cutoff=float(data["cutoff"]), span=float(data.get("span", DEFAULT_SPAN)))


def default_config(threshold_path: Optional[str] = None) -> ThresholdConfig:
    """Loads a frozen calibration JSON if present, otherwise falls back to the shipped
    default fit against models/9_1_2 (relevant when a different/uncalibrated model is
    loaded at runtime — a fitted cutoff is specific to the model/scaler pair it came
    from)."""
    if threshold_path is not None and os.path.exists(threshold_path):
        with open(threshold_path, "r", encoding="utf-8") as f:
            return ThresholdConfig.from_json(f.read())
    return ThresholdConfig()


class EwmaPeakDecision:
    """Streaming EWMA-peak session decision.

    Tracks the running EWMA-smoothed score and its peak-so-far; always resolves to
    HEALTHY or INFESTED once at least one score has been seen — unlike the SPRT
    accumulator this never abstains, matching a fixed-length session that always
    completes rather than an open-ended one that can wait for more evidence.
    """

    def __init__(self, config: ThresholdConfig):
        self.config = config
        self._alpha = 2.0 / (config.span + 1.0)
        self._smoothed: Optional[float] = None
        self._peak = 0.0

    def update(self, score: float) -> float:
        if self._smoothed is None:
            self._smoothed = score
        else:
            self._smoothed = self._alpha * score + (1.0 - self._alpha) * self._smoothed
        self._peak = max(self._peak, self._smoothed)
        return self._peak

    @property
    def peak(self) -> float:
        return self._peak

    @property
    def state(self) -> str:
        return "INFESTED" if self._peak > self.config.cutoff else "HEALTHY"
