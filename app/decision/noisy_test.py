"""Noisy test handling: warn, extend, repeat
(design: docs/superpowers/specs/2026-10-28-noisy-test-handling-design.md).

Environmental noise changes 9_1_2's window scores: passing cars push them DOWN, so an
infested tree near a road can come out HEALTHY, and other noises (wind, pumps, engines,
or several at once) may push them up. So a window recorded during noise counts for
nothing, in either direction (owner decision D2), and the verdict is the EWMA-peak rule
over exactly the first 36 QUIET windows (D1) -- the window count the cutoff was fitted on.
With no noise, those are today's 36 windows after 20 s and nothing changes.

When the normal 20 s ends short of 36 quiet windows the session asks the caller to
EXTEND (keep recording, up to `max_extra_sec`), then to REPEAT (a fresh attempt, up to
`max_attempts`), and finally answers TOO_NOISY. Which of these happen is configuration.

PORTABILITY. Stdlib + NumPy only; no UI, no audio capture, no timing, no file I/O. The
caller owns the microphone and the clock and feeds one 0.5 s hop at a time:
`feed_hop(samples, window_score)` or, if it computes the floor itself,
`feed_floor(floor_db, window_score)`. That split is what the Flutter port reproduces; the
golden vectors in tests/data/noisy_test_vectors.json drive `feed_floor` directly.

THE NOISE MEASURE is the hop's TRIMMED FLOOR: RMS after clipping every sample to the hop's
own `trim_percentile` of |x| (from features/noise_gated_windows:app/decision/noise_gate.py).
Bite clicks occupy a few percent of a hop, so trimming removes them; environmental noise
IS the floor. Click counts, template matches, crest factor and untrimmed peaks are the
larva signature and must never be used here.

A window (5 hops) is NOISY if any of its hops has a floor
    * more than `noise_db` above the attempt's REFERENCE -- the `reference_quantile` of
      all finite floors seen so far in this attempt (a passing car against the quiet
      moments between cars); or
    * above `abs_noise_db` in absolute dBFS (noise that never stops).
The quiet/noisy mask is recomputed from all hops each time, so a test that starts during a
car is judged against the quiet part that follows. Silent hops (floor -inf) are not noise.

SHIPPED DEFAULTS ARE A STRUCTURAL NO-OP: `enabled = False` (the session then ends after
`normal_hops` with today's all-window EWMA verdict), and both thresholds are infinite.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional

import numpy as np

from app.decision.threshold import DEFAULT_CUTOFF, DEFAULT_SPAN, EwmaPeakDecision, ThresholdConfig

HEALTHY = "HEALTHY"
INFESTED = "INFESTED"
TOO_NOISY = "TOO_NOISY"

CONTINUE = "CONTINUE"
EXTENDING = "EXTENDING"
REPEAT = "REPEAT"
DONE = "DONE"

HOP_SEC = 0.5


def hop_floor_db(hop, trim_percentile: float = 90.0) -> float:
    """Trimmed floor of one hop in dBFS. -inf for empty, non-finite or all-zero input."""
    x = np.asarray(hop, dtype=np.float64).ravel()
    if x.size == 0 or not np.all(np.isfinite(x)):
        return -math.inf
    limit = float(np.percentile(np.abs(x), trim_percentile))
    clipped = np.clip(x, -limit, limit) if limit > 0.0 else x
    rms = float(np.sqrt(np.mean(clipped * clipped)))
    if rms <= 0.0:
        return -math.inf
    return 20.0 * math.log10(rms)


def _enc(v: float):
    """JSON has no infinity; store it as null."""
    return None if isinstance(v, float) and math.isinf(v) else v


@dataclass(frozen=True)
class NoisyTestConfig:
    enabled: bool = False
    trim_percentile: float = 90.0
    reference_quantile: float = 0.2
    noise_db: float = math.inf          # FIT: dB above the attempt's reference
    abs_noise_db: float = math.inf      # FIT: absolute dBFS
    min_quiet_windows: int = 36         # = the window count the cutoff was fitted on (D1)
    hops_per_window: int = 5            # 2.5 s window / 0.5 s hop
    normal_hops: int = 40               # 20 s
    max_extra_sec: float = 0.0
    max_attempts: int = 1
    repeat_pause_sec: float = 3.0
    cutoff: float = DEFAULT_CUTOFF
    span: float = DEFAULT_SPAN

    @property
    def extra_hops(self) -> int:
        return int(round(self.max_extra_sec / HOP_SEC))

    def to_json(self) -> str:
        d = {k: _enc(v) for k, v in asdict(self).items()}
        d["method"] = "quiet_windows_v1"
        return json.dumps(d, indent=2)

    @staticmethod
    def from_json(text: str) -> "NoisyTestConfig":
        data = json.loads(text)
        base = NoisyTestConfig()
        kw = {}
        for f in fields(NoisyTestConfig):
            if f.name not in data:
                continue
            v = data[f.name]
            default = getattr(base, f.name)
            if isinstance(default, bool):
                kw[f.name] = bool(v)
            elif isinstance(default, int):
                kw[f.name] = int(v)
            else:
                kw[f.name] = math.inf if v is None else float(v)
        return NoisyTestConfig(**kw)


def default_config(config_path: Optional[str] = None) -> NoisyTestConfig:
    """Loads `models/<version>/noisy_test.json` if present, else the shipped no-op."""
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return NoisyTestConfig.from_json(f.read())
    return NoisyTestConfig()


@dataclass
class AttemptResult:
    verdict: str                 # HEALTHY / INFESTED / TOO_NOISY
    provisional_verdict: str     # EWMA peak over ALL windows of the attempt (today's answer)
    quiet_peak: float            # EWMA peak over the first min_quiet_windows quiet windows
    all_peak: float
    n_quiet: int
    n_windows: int
    n_hops: int
    attempt: int
    reference_db: float
    quiet_mask: List[bool] = field(default_factory=list)
    floors_db: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reference_db"] = _enc(d["reference_db"])
        d["floors_db"] = [None if math.isinf(v) else v for v in d["floors_db"]]
        return d


@dataclass
class Action:
    kind: str                            # CONTINUE / EXTENDING / REPEAT / DONE
    result: Optional[AttemptResult] = None


class NoisyTestSession:
    """One test, possibly over several attempts. Feed every 0.5 s hop; act on the returned
    Action. After REPEAT the caller pauses `repeat_pause_sec` and calls `start_attempt()`."""

    def __init__(self, config: NoisyTestConfig):
        self.config = config
        self.attempt = 0
        self.start_attempt()

    def start_attempt(self) -> None:
        self.attempt += 1
        self._floors: List[float] = []
        self._scores: List[float] = []
        self._extending = False
        self._closed = False

    # --- feeding -----------------------------------------------------------------------

    def feed_hop(self, hop_samples, window_score: Optional[float]) -> Action:
        return self.feed_floor(hop_floor_db(hop_samples, self.config.trim_percentile), window_score)

    def feed_floor(self, floor_db: float, window_score: Optional[float]) -> Action:
        """`window_score` is the score of the window ending with this hop, or None while the
        first window is still filling (the first hops_per_window - 1 hops of an attempt)."""
        if self._closed:
            raise RuntimeError("attempt already finished; call start_attempt()")
        c = self.config
        self._floors.append(float(floor_db))
        if window_score is not None:
            self._scores.append(float(window_score))
        n_hops = len(self._floors)

        if not c.enabled:
            if n_hops >= c.normal_hops:
                return self._close(DONE)
            return Action(CONTINUE)

        if self.n_quiet() >= c.min_quiet_windows:
            return self._close(DONE)
        if n_hops < c.normal_hops:
            return Action(CONTINUE)
        if n_hops == c.normal_hops and c.extra_hops > 0:
            self._extending = True
            return Action(EXTENDING)
        if n_hops < c.normal_hops + c.extra_hops:
            return Action(CONTINUE)
        return self._close(REPEAT if self.attempt < c.max_attempts else DONE)

    def finish(self) -> AttemptResult:
        """The caller stopped early (operator pressed stop, or a replayed file ended)."""
        self._closed = True
        return self._result()

    # --- state -------------------------------------------------------------------------

    @property
    def extending(self) -> bool:
        return self._extending

    def reference_db(self) -> float:
        finite = [f for f in self._floors if math.isfinite(f)]
        if not finite:
            return -math.inf
        return float(np.quantile(np.asarray(finite), self.config.reference_quantile))

    def hop_noisy(self) -> List[bool]:
        c = self.config
        ref = self.reference_db()
        out = []
        for f in self._floors:
            if not math.isfinite(f):
                out.append(False)
                continue
            rel = math.isfinite(ref) and (f - ref) > c.noise_db
            out.append(bool(rel or f > c.abs_noise_db))
        return out

    def quiet_mask(self) -> List[bool]:
        """One entry per scored window; window w covers hops w .. w + hops_per_window - 1."""
        hn = self.hop_noisy()
        k = self.config.hops_per_window
        offset = len(self._floors) - len(self._scores) - (k - 1)
        mask = []
        for w in range(len(self._scores)):
            first = w + offset
            mask.append(not any(hn[max(first, 0):first + k]))
        return mask

    def n_quiet(self) -> int:
        return sum(self.quiet_mask())

    # --- internals ---------------------------------------------------------------------

    def _peak(self, scores) -> float:
        acc = EwmaPeakDecision(ThresholdConfig(cutoff=self.config.cutoff, span=self.config.span))
        for s in scores:
            acc.update(s)
        return acc.peak

    def _state(self, peak: float) -> str:
        return INFESTED if peak > self.config.cutoff else HEALTHY

    def _result(self) -> AttemptResult:
        c = self.config
        all_peak = self._peak(self._scores)
        provisional = self._state(all_peak)
        if not c.enabled:
            mask = [True] * len(self._scores)
            quiet_scores = list(self._scores)
            verdict = provisional
        else:
            mask = self.quiet_mask()
            quiet_scores = [s for s, q in zip(self._scores, mask) if q][: c.min_quiet_windows]
            verdict = (self._state(self._peak(quiet_scores))
                       if len(quiet_scores) >= c.min_quiet_windows else TOO_NOISY)
        return AttemptResult(
            verdict=verdict,
            provisional_verdict=provisional,
            quiet_peak=self._peak(quiet_scores),
            all_peak=all_peak,
            n_quiet=sum(mask),
            n_windows=len(self._scores),
            n_hops=len(self._floors),
            attempt=self.attempt,
            reference_db=self.reference_db(),
            quiet_mask=mask,
            floors_db=list(self._floors),
        )

    def _close(self, kind: str) -> Action:
        self._closed = True
        return Action(kind, self._result())
