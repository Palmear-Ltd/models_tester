"""Conceptual data objects for the Audio Signal Health Monitoring subsystem.

Pure-data, NumPy-only. This module imports nothing from the tester or UI so the
logic stays portable. See the Phase 0 design spec for the conceptual model (Ch. 5).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class HealthState(Enum):
    OK = "OK"
    WARNING = "WARNING"
    FAULT = "FAULT"
    UNKNOWN = "UNKNOWN"


class CheckStatus(Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    NOT_EXECUTED = "NOT_EXECUTED"


class CheckCategory(Enum):
    """Importance of a check for Decision Fusion (spec Ch. 10.3)."""

    CRITICAL = "CRITICAL"
    PRIMARY = "PRIMARY"
    SUPPORTING = "SUPPORTING"


@dataclass(frozen=True)
class Measurement:
    """A single numerical value produced by a Signal Health Check."""

    name: str
    value: float
    unit: str = ""


@dataclass(frozen=True)
class AudioWindow:
    """Immutable 2.5s mono analysis window. The fundamental pipeline input."""

    samples: np.ndarray
    sample_rate: int
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        incoming = np.asarray(self.samples)
        if incoming.ndim != 1:
            raise ValueError(
                f"AudioWindow expects a 1-D mono sample array, got shape {incoming.shape}"
            )
        # Store a private, read-only float32 copy so the window is immutable
        # and decoupled from the caller's buffer.
        arr = np.array(incoming, dtype=np.float32)
        arr.setflags(write=False)
        object.__setattr__(self, "samples", arr)

    @property
    def sample_count(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channel_count(self) -> int:
        return 1

    @property
    def window_duration(self) -> float:
        return self.sample_count / float(self.sample_rate)


@dataclass
class SignalCheckResult:
    """Outcome of one Signal Health Check for one Audio Window.

    Intentionally mutable (not frozen): the SignalCheckManager sets ``executed``
    and ``execution_time`` after the check runs. Do not convert this to frozen.
    """

    check_id: str
    check_name: str
    status: CheckStatus = CheckStatus.NOT_EXECUTED
    category: CheckCategory = CheckCategory.PRIMARY
    executed: bool = False
    execution_time: float = 0.0
    measurements: list[Measurement] = field(default_factory=list)
    diagnostic_messages: list[str] = field(default_factory=list)


@dataclass
class HealthReport:
    """Final output of the Health Analysis Pipeline for one Audio Window."""

    timestamp: float
    window_id: str
    check_results: list[SignalCheckResult] = field(default_factory=list)
    calibration_evaluation: Optional[Any] = None
    anomaly_result: Optional[Any] = None
    final_state: HealthState = HealthState.UNKNOWN
    confidence: float = 0.0
    diagnostic_summary: str = ""
