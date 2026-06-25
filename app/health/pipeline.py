"""Health Analysis Pipeline — evaluates one Audio Window into one Health Report.

Seven stages (spec Ch. 3). In Phase 0 the stages are wired but preprocessing,
feature preparation, calibration, and anomaly detection are no-ops.
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from typing import Any, Optional

from app.health.feature_prep import prepare_features
from app.health.calibration_eval import CalibrationEvaluation, evaluate_calibration
from app.health.fusion import decide
from app.health.manager import SignalCheckManager
from app.health.models import AudioWindow, HealthReport


class HealthAnalysisPipeline:
    def __init__(self, manager: Optional[SignalCheckManager] = None, calibration_profile=None, history_length: int = 20):
        self.manager = manager if manager is not None else SignalCheckManager()
        self.calibration_profile = calibration_profile
        self._history: deque = deque(maxlen=history_length)

    def analyze(self, window: AudioWindow) -> HealthReport:
        # Stage 1 — Preprocessing (Phase 0: no-op)
        window = self._preprocess(window)
        # Stage 2 — Feature Preparation
        features = self._prepare_features(window)
        features["history"] = list(self._history)  # prior windows (oldest -> newest)
        # Stage 3 — Signal Health Checks
        results = self.manager.run_checks(window, features)
        # Record this window's measurements as the next entry of the stability history.
        self._history.append(
            {r.check_id: {m.name: m.value for m in r.measurements} for r in results}
        )
        # Stage 4 — Calibration Evaluation
        calibration_evaluation = self._evaluate_calibration(results)
        # Stage 5 — Anomaly Detection
        anomaly_result = self._detect_anomalies(features, results)
        # Stage 6 — Decision Fusion
        final_state, confidence, summary = decide(results, calibration_evaluation)
        # Stage 7 — Health Report Generation
        return HealthReport(
            timestamp=time.time(),
            window_id=uuid.uuid4().hex,
            check_results=results,
            calibration_evaluation=calibration_evaluation,
            anomaly_result=anomaly_result,
            final_state=final_state,
            confidence=confidence,
            diagnostic_summary=summary,
        )

    def _preprocess(self, window: AudioWindow) -> AudioWindow:
        return window

    def _prepare_features(self, window: AudioWindow) -> dict[str, Any]:
        return prepare_features(window)

    def _evaluate_calibration(self, results: list) -> Optional[CalibrationEvaluation]:
        if self.calibration_profile is None:
            return None
        return evaluate_calibration(results, self.calibration_profile)

    def _detect_anomalies(self, features: dict, results: list) -> Optional[Any]:
        # Phase 0 no-op; Phase 6 scores the feature vector against healthy behaviour.
        return None
