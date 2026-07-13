"""Sequential Probability Ratio Test (SPRT) evidence accumulation over classifier scores.

Replaces a hand-tuned per-window threshold plus two absolute count bands with two
interpretable error-rate knobs: alpha (target false-positive rate) and beta (target
false-negative rate). See docs/superpowers/specs for the full rationale.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

from .likelihood import FittedLikelihood, fitted_llr_increment

StepState = Literal["CONTINUE", "HEALTHY", "INFESTED"]
FinalState = Literal["HEALTHY", "INFESTED", "SUSPICIOUS"]

DEFAULT_ALPHA = 0.05
DEFAULT_BETA = 0.05
DEFAULT_WEIGHT = 0.2


def sprt_boundaries(alpha: float, beta: float) -> tuple[float, float]:
    """Wald's SPRT decision boundaries in log-likelihood-ratio (nats).

    Crossing cumulative LLR >= A decides INFESTED; crossing LLR <= B decides HEALTHY.
    """
    if not (0.0 < alpha < 1.0) or not (0.0 < beta < 1.0):
        raise ValueError(f"alpha and beta must be in (0, 1); got alpha={alpha}, beta={beta}")
    a = math.log((1.0 - beta) / alpha)
    b = math.log(beta / (1.0 - alpha))
    return a, b


@dataclass(frozen=True)
class SPRTConfig:
    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA
    eps: float = 1e-6
    weight: float = DEFAULT_WEIGHT
    likelihood_mode: Literal["logit", "fitted"] = "logit"
    fitted_params: Optional[FittedLikelihood] = None

    def __post_init__(self) -> None:
        if self.likelihood_mode == "fitted" and self.fitted_params is None:
            raise ValueError("likelihood_mode='fitted' requires fitted_params")

    @property
    def boundaries(self) -> tuple[float, float]:
        return sprt_boundaries(self.alpha, self.beta)


def default_config(
    likelihood_path: Optional[str] = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    weight: float = DEFAULT_WEIGHT,
) -> SPRTConfig:
    """Default SPRT configuration.

    If `likelihood_path` points at an existing fitted-likelihood JSON (see
    `likelihood.FittedLikelihood`), uses fitted-mode evidence; otherwise falls back to
    logit-mode, which needs no calibration file (e.g. a different/uncalibrated model was
    loaded at runtime — a fitted Beta calibration is specific to the model/scaler pair it
    was fit against).
    """
    if likelihood_path is not None and os.path.exists(likelihood_path):
        with open(likelihood_path, "r", encoding="utf-8") as f:
            fitted = FittedLikelihood.from_json(f.read())
        return SPRTConfig(alpha=alpha, beta=beta, weight=weight, likelihood_mode="fitted", fitted_params=fitted)
    return SPRTConfig(alpha=alpha, beta=beta, weight=weight, likelihood_mode="logit")


def llr_increment(score: float, config: SPRTConfig) -> float:
    """Evidence contributed by a single window's score, in nats."""
    p = min(max(score, config.eps), 1.0 - config.eps)
    if config.likelihood_mode == "fitted":
        assert config.fitted_params is not None
        raw = fitted_llr_increment(p, config.fitted_params, config.eps)
    else:
        raw = math.log(p / (1.0 - p))
    return config.weight * raw


@dataclass(frozen=True)
class StepResult:
    index: int
    score: float
    llr_increment: float
    cumulative_llr: float
    state: StepState


@dataclass(frozen=True)
class SessionResult:
    final_state: FinalState
    cumulative_llr: float
    decided_at_index: Optional[int]
    trace: list[StepResult] = field(default_factory=list)


class SPRTAccumulator:
    """Streaming SPRT evidence accumulator over one session's window scores.

    Latches at HEALTHY/INFESTED on first boundary crossing; further `update()` calls keep
    accumulating and logging evidence but no longer change the decided state.
    """

    def __init__(self, config: SPRTConfig):
        self.config = config
        self._a, self._b = config.boundaries
        self._cumulative_llr = 0.0
        self._state: StepState = "CONTINUE"
        self._decided_at_index: Optional[int] = None
        self._trace: list[StepResult] = []

    def update(self, score: float) -> StepResult:
        increment = llr_increment(score, self.config)
        self._cumulative_llr += increment
        index = len(self._trace)
        if self._state == "CONTINUE":
            if self._cumulative_llr >= self._a:
                self._state = "INFESTED"
                self._decided_at_index = index
            elif self._cumulative_llr <= self._b:
                self._state = "HEALTHY"
                self._decided_at_index = index
        step = StepResult(
            index=index,
            score=score,
            llr_increment=increment,
            cumulative_llr=self._cumulative_llr,
            state=self._state,
        )
        self._trace.append(step)
        return step

    def result(self) -> SessionResult:
        final_state: FinalState = self._state if self._state != "CONTINUE" else "SUSPICIOUS"
        return SessionResult(
            final_state=final_state,
            cumulative_llr=self._cumulative_llr,
            decided_at_index=self._decided_at_index,
            trace=list(self._trace),
        )
