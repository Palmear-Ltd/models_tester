# Root-cause threshold recalibration — design

## Problem

`app/health/rootcause.py`'s SENSOR_LINK attribution (surfaced to the tester via the
"Likely cause" label, the health panel, and the "Validate Acquisition" button) is
supposed to tell a field tester whether an unhealthy signal reading reflects a real
sensor/cable problem, as distinct from the CNN's own classification. Verified against
14 local WAV files (8 known-SENSOR_LINK-fault `test_data/audio_signal_health/fp/F/`,
4 clean `test_data/F/` TN reference, 2 `test_data/T/` TP) it currently returns
**SENSOR_LINK at confidence 1.00 on every single file**, fault or clean. It cannot
currently be used to tell a tester "trust this" — it says "sensor problem" regardless
of ground truth.

Two independent, compounding bugs, both against `app/health/checks/time_domain.py`,
`app/health/checks/stability.py`, and `app/health/rootcause.py`:

### Bug 1 — per-window check thresholds sit inside normal piezo noise, not above it

`ClickTransientCheck` (T009) warns at `click_count >= 3`, fails at `>= 15`
(`checks/time_domain.py:399-409`, both hardcoded `__init__` defaults). Measured
per-window `click_count` across our local corpora:

| Corpus | median | mean | p95 | % windows ≥3 (WARN) | % windows ≥15 (FAIL) |
|---|---|---|---|---|---|
| TN clean reference | 0 | 51.5 | 51 | 25% | 6% |
| fp/F known fault | 2 | 115.9 | 90 | 47% | 24% |
| TP infested | 3 | 150.1 | 396 | 51% | 25% |

There is real separation between clean and faulty (roughly 2x the flag rate), but the
absolute cutoffs (3 / 15) were set well inside the noise floor of genuinely healthy
piezo contact, not above it — a quarter of clean windows already clear WARN.

### Bug 2 — `assess_many` has no session-level floor; it saturates on any nonzero sum

`rootcause.py:189-214` sums each window's weighted score across a whole ~40-window
session (`combined_score += score`), then `_finalize` (rootcause.py:132-160) does:

```python
elif score <= 0.0:
    primary_cause = RootCause.UNKNOWN
else:
    primary_cause = RootCause.SENSOR_LINK
    confidence = min(1.0, score / MAX_SCORE)   # MAX_SCORE=10.0, documented as a
                                                # PER-WINDOW cap, applied to a
                                                # 40-window SUM
```

Any positive combined score confirms SENSOR_LINK — there is no minimum bar relative
to session length or a healthy baseline. Since even clean recordings flag ~25% of
windows at WARN-or-worse (weight 1-3 each per the `_WEIGHT_TABLE`), the combined score
clears `MAX_SCORE=10` within the first handful of windows out of 40, saturating
confidence at 1.00 regardless of whether the true per-window rate is 6% (clean) or
24% (fault).

## Why this matters (motivating use case)

Field workflow: a tester straps the sensor to a tree and runs a live test — WAV
testing is developer/analyst-only. The goal of this whole subsystem is to let a
tester (or an analyst reviewing saved reports) distinguish *"the model's verdict
reflects the actual acoustic signal"* from *"the sensor/cable fed the model garbage,
discard this result"* — i.e. answer "can I trust this prediction" for model
**performance evaluation** purposes (was a wrong verdict the model's fault, or the
environment's). The mechanism built for exactly this (Phase 5 `Validate Acquisition`
button, Phase 8d live "Likely cause" label — both already wired into `main.py`) is
unusable in its current state because it never says anything but SENSOR_LINK.

## Data source for recalibration

`calibration_profiles/multiyear_healthy_v1.json` — 45,872 windows, 5 hardware
variants, 2021-2026, generated from `TN_`-only recordings. **The raw corpus behind
it (`/home/bashar/workspace/palmear/9_1_4`) is out of scope going forward per
explicit direction — do not read from it, and do not regenerate/resample this
profile from it.** Use only the summary statistics already baked into the committed
profile JSON (`statistics[check_id][measurement].{mean,median,std,p5,p95,minimum,maximum}`
— note there is no p99; p95 is the finest available percentile).

Relevant existing stats (from the committed profile):

| check.measurement | mean | median | p95 | max |
|---|---|---|---|---|
| T008.dropout_event_count | 0.012 | 0 | 0 | 14 |
| T008.max_dropout_run_ms | 0.794 | 0 | 0 | 960 |
| T008.dropout_frame_ratio | ~0 | 0 | 0 | 0.384 |
| T009.click_count | 10.166 | 0 | 43 | 2532 |
| T009.click_rate | 4.066 | 0 | 17.2 | 1012.8 |
| S004.recurrence_ratio | 0.009 | 0 | 0.050 | 0.650 |

S004 (`max_recurrence=0.3` vs baseline p95=0.050) is already well-calibrated — leave
it alone. T008's WARNING path (`min_event_ms=30`, vs baseline p95=0 for
`max_dropout_run_ms`) is defensible as-is (only the rare tail crosses it) — T009's
`warn_count=3`/`fault_count=15` are the clear outlier relative to their own baseline
(p95=43) and are the primary fix target. T008's FAIL path (`fault_ratio=0.15`) should
be spot-checked too but is lower priority (p95 of `dropout_frame_ratio` is 0, so it's
likely already conservative).

## Fix design

### 1. Retune T009 defaults directly (not gated behind a loaded calibration profile)

`calibration_profile` is never auto-loaded (`self.calibration_profile = None` at
startup, per existing project memory) — a fix that only takes effect when a tester
manually loads a profile in Settings ships zero value to the common case. The
per-check **default** constructor values in `checks/time_domain.py` must move to
better-calibrated numbers directly; a loaded calibration profile may *additionally*
refine them later, but the shipped defaults are the primary fix.

Proposed (anchored on the broad baseline's p95=43, doubled for FAIL since fp/F's own
p95 is ~2x the clean baseline's — exact multiplier is a config value, not hardcoded,
so it's cheaply retunable without a code change once a finer-grained profile exists):
`warn_count` p95-anchored (~43→ round to 40), `fault_count` ~2x that (~85). Land on
exact numbers during implementation with the synthetic + corpus-replay tests below —
these are starting points, not final.

### 2. Persist thresholds as data, not hardcoded Python literals

Add a small persisted JSON config (mirrors the existing `app/decision/threshold.py`
`ThresholdConfig`/`default_config()` idiom already in this codebase) holding the
per-check constructor overrides, loaded once at pipeline-build time and threaded
through `app/health/config.py`'s existing `CheckConfig(params=...)` mechanism (this
plumbing already exists — `build_manager` already does
`params = cc.params if cc else {}` then `spec.factory(**params)`). No new plumbing
needed for *applying* the config, only for *loading* it from a file instead of a
hardcoded dict.

### 3. Fix `assess_many` to require a session-level floor, not "any positive sum"

Replace "any positive combined score ⇒ SENSOR_LINK, confidence = min(1, score/10)"
with a persisted, data-driven session cutoff — same `ThresholdConfig`-style pattern
as `app/decision/threshold.py`'s `EwmaPeakDecision` (cutoff fit from data, not
hand-picked, loaded from JSON with a shipped fallback default). The exact statistic
to threshold (combined score normalized by windows-considered, or fraction of
windows non-PASS) and its cutoff value should be derived from what a genuinely clean
~40-window session typically produces (using the local TN reference + the broad
profile's implied per-window WARN rates as a sanity bound), then validated by
replaying the local corpora: TN clean should mostly resolve to NONE/UNKNOWN, fp/F
should still mostly resolve to SENSOR_LINK.

### Non-goals

- Not touching `calibration_eval.py` / `anomaly.py` (the Mahalanobis anomaly-count
  signal) — that one already showed real, if noisy, separation in prior testing and
  is out of scope here.
- Not attempting to regenerate `multiyear_healthy_v1.json` with finer percentiles —
  9_1_4 is out of scope. If finer-grained cutoffs are wanted later, that's a
  follow-up against whatever corpus replaces it.

## Acceptance / validation

Replay `test_data/audio_signal_health/fp/F/*.wav`, `test_data/F/*.wav`,
`test_data/T/*.wav` through `rootcause.assess_many` with the new defaults + session
cutoff:
- fp/F (8 files): should still mostly resolve SENSOR_LINK (this is a real, confirmed
  fault — don't regress detection of the thing we already root-caused).
- test_data/F TN (4 files): should mostly resolve NONE/UNKNOWN, not SENSOR_LINK.
- test_data/T TP (2 files): informative only (n=2, thin) — no strong assertion.

Plus synthetic-signal unit tests per project TDD convention (silence, sine, noise,
clipped, DC-offset, hum, and specifically synthetic click-burst / dropout-burst
signals at controlled rates straddling the old vs. new thresholds) in
`tests/health/test_time_domain.py`, `tests/health/test_rootcause.py`.
