"""Base interface every Signal Health Check implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.health.models import AudioWindow, SignalCheckResult


class SignalHealthCheck(ABC):
    """Evaluates exactly one property of an Audio Window.

    Subclasses set ``check_id`` / ``check_name`` and implement ``run``. Checks must
    not depend on one another; the manager runs each in isolation.
    """

    check_id: str = ""
    check_name: str = ""

    @abstractmethod
    def run(self, window: AudioWindow, features: dict[str, Any]) -> SignalCheckResult:
        """Measure the property and return a SignalCheckResult."""
        raise NotImplementedError
