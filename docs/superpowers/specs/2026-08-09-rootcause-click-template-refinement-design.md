# Root-cause SENSOR_LINK refinement: locally-fit click spectral template — design

## Problem

`app/health/rootcause.py`'s SENSOR_LINK attribution — the "Validate Acquisition" button
and live "Likely cause" label — currently scores click activity using T009
(`ClickTransientCheck`)'s raw click *count* against a session-mean cutoff
(`rootcause_session_config.json`, `DEFAULT_SESSION_CUTOFF = 0.25`, refit 2026-07-30). Raw
click count cannot tell a real bite click apart from a sensor/cable contact-noise click —
this session's investigation found T-labeled (infested) sessions have meaningfully more
clicks than F sessions (rank-AUC 0.68), but confirmed SENSOR_LINK-fault sessions have
*even more* clicks than genuinely infested ones (mean click count: TN 14.7, TP 36.1, FP
40.0 per session) — so click count alone cannot distinguish the fault SENSOR_LINK exists
to detect from the real signal it must not fire on.

## Evidence (this session's investigation)

Working backward through what was tried and what survived honest, held-out testing:

1. **Individual click shape** (attack/decay envelope timing) looked promising on a 2-file
   sample (AUC 0.35–0.41) but **collapsed to chance (AUC 0.48–0.49, p>0.9) at proper
   file-level power** (40 T files) — a pseudo-replication artifact, discarded.
2. **Literature frequency-band matching** (RPW eating/biting peaks ~1,651/2,219 Hz, per
   Mankin et al.'s bioacoustic RPW detection work) also failed at file-level power — T vs.
   FAULT AUC 0.46–0.53, indistinguishable from chance. Sensor-noise clicks land in the same
   generic low-kHz range as real clicks; a fixed frequency box doesn't separate them.
3. **A locally-fit differential spectral template** — built from *our own* corpus, not
   imported literature constants — did separate them. Method: average the (L2-normalized,
   40-bin, 0–8000Hz) power spectrum of clicks from T-train sessions, subtract the average
   spectrum of TN-train (clean baseline) sessions, use the residual as a matched-filter
   direction. Score each click by its dot product with this template; fit a per-click
   match threshold via Youden's J on train data only.

   At small scale (40 T / 40 TN / 8 FAULT files, single-pass whole-file click detection):
   T-test vs. FAULT-gold burst-rate AUC 0.81. **Rescaled to 183 T / 200 TN / 8 FAULT-gold +
   150 FP-silver files** (stratified 2022–2026, 2021 excluded — deliberately-injected noise
   that year, see prior turns), properly held out:

   | comparison | raw click rate AUC | matched-click-rate AUC | burst-rate AUC |
   |---|---|---|---|
   | T vs. **FAULT-gold** (n=8, confirmed root-caused SENSOR_LINK) | 0.50 | 0.58 | **0.71–0.72** |
   | T vs. **FP-silver** (n=150, model-flagged FPs, *not* individually confirmed) | 0.63 | 0.65 | 0.58–0.62 |
   | TN vs. FAULT-gold | 0.24–0.28 | 0.28 | **0.43–0.47 (collapses toward chance)** |
   | TN vs. FP-silver | 0.34–0.39 | 0.35–0.39 | 0.37–0.44 (stays separated) |

**The critical, scope-defining finding:** this template is genuinely effective against the
*narrow, confirmed* SENSOR_LINK contact-noise signature (the burst-rate metric brings
FAULT-gold's click activity down to statistically indistinguishable from clean TN — AUC
collapses to ~0.45), but **it is not a general false-positive fix** — against the broader,
heterogeneous population of model-flagged FPs it barely beats raw click count. That
heterogeneity is expected: not every FP is a SENSOR_LINK fault (weak/ambiguous real
signal, other environmental confounds, etc. are also in that bucket, and this template was
never fit to recognize those). This refinement's validated scope is exactly
`rootcause.py`'s own stated scope — SENSOR_LINK specifically, not general FP explanation
— which is why it belongs here and not, e.g., as a CNN confidence gate.

## Why this matters

`rootcause.py`'s own docstring already states its narrow purpose: distinguish "the
model's verdict reflects the actual acoustic signal" from "the sensor/cable fed the model
garbage, discard this result." Its current click signal (T009 count) cannot make that
distinction reliably — this refinement gives it a signal that, on held-out confirmed-fault
data, actually can.

## Design

### 1. A new, independent check — not an extension of T009

T009's own `click_count`/`click_rate` measurements and PASS/WARNING/FAIL thresholds
(`check_thresholds.json`, recalibrated 2026-07-16) serve the general-purpose "is there
excessive clicking" health signal used elsewhere in the system (per-check panel, live
indicator) — that calibration is validated for its own purpose and shouldn't be
disturbed. `rootcause.py`'s `_WEIGHT_TABLE` only ever consumes `(check_id, status)` pairs
from already-scored, independent checks (`app/health/`'s "checks are independent,
isolated" convention) — so the cleanest fit is a **new check** (tentatively `T010`,
category `PRIMARY` like T009) that:
- Re-runs the same click-detection algorithm T009 uses (robust-MAD-sigma first-difference
  threshold, `click_k`/`merge_gap`) on `window.samples` — a small, cheap duplication, not
  a T009 dependency, preserving check isolation.
- Scores each detected click against a **persisted differential template** (see §2),
  emitting a new measurement (`matched_click_count`, and/or `burst_rate` — see §3's open
  question).
- Has its own PASS/WARNING/FAIL thresholds, in the same `check_thresholds.json` idiom as
  T009's `warn_count`/`fault_count`, fit empirically (not guessed) during implementation.

### 2. The template is a new, fitted, shippable data file

`app/health/click_template.json` (or filed under `calibration_profiles/` if that's a
better fit for the project's existing data-file conventions — decide during
implementation): `{"n_bins": 40, "freq_max_hz": 8000.0, "template": [...40 floats...],
"match_threshold": <float>, "fitted_against": "<corpus description>", "fitted_date":
"..."}`. Loaded via the standard load-JSON-with-fallback idiom (`app/decision/threshold.py`
pattern) — missing file means the new check simply reports UNKNOWN/PASS-through rather
than raising, so a stale or absent template can never break a session.

Fitting happens via a new offline script (mirrors `calibrate.py`), **not** inside
`app/health/` — `app/health/` stays stdlib+NumPy only, but *fitting* the template needs
`soundfile`/`librosa` for corpus I/O, same split as everything else in this codebase.

### 3. Open question: per-window measurement vs. true session-level burst rate

What was actually validated this session used **single-pass detection over each file's
whole (up to 20s) captured audio** — not the live per-window rolling-buffer computation
T009 actually runs (2.5s window, 0.5s hop, ~5x overlap, so the same physical click is
seen — and would be recounted — in ~5 consecutive windows). That's not automatically
disqualifying: T009's own existing `click_count` already has this exact redundancy, and
`rootcause.py`'s `assess_many` sum→mean→cutoff architecture was already fit *around* it
(see the 2026-07-16/2026-07-30 recalibration specs). The straightforward path is:
compute **`matched_click_count` per window** (T010, mirroring T009's cadence exactly) and
let `assess_many`'s existing aggregation handle it identically to T009 today — no new
plumbing, no raw-audio access needed inside `rootcause.py`.

The literature's actual *burst rate* (gap-grouped bursts/sec) is a coarser, more
session-level statistic that a 2.5s window can barely express. **This must be verified,
not assumed**, in Phase A (see plan): refit and re-evaluate the per-window
`matched_click_count` aggregate under the *real* production windowing (reuse
`offline_score.py`/`scripts/rms_scan.py`'s exact windowing-replication pattern) and
confirm the T-vs-FAULT-gold separation found this session (with single-pass detection)
survives. If it doesn't survive, the fallback is giving `assess_many` access to raw
per-window click timestamps (a bigger, deferred change) so true burst-grouping can happen
at the session level — do not build that unless the simpler path is shown to fail.

### 4. `rootcause.py` weight-table and session-cutoff refit

Add `(T010, WARNING)` / `(T010, FAIL)` entries to `_WEIGHT_TABLE`, and refit
`DEFAULT_SESSION_CUTOFF` (currently 0.25) against the new combined weight table — the
existing cutoff was fit to the *old* score distribution and adding a new scoring
dimension changes it. Refit against the same local corpora used for every prior
recalibration of this module (`test_data/F` TN ×4, `test_data/audio_signal_health/fp/F`
FAULT ×8), replayed through the *actual* per-window pipeline. Whether T010 supplements or
partially supersedes T009's weight in this table is an empirical question for the refit
step, not a decision made here.

## Non-goals

- **Not a general false-positive fix.** Evidence explicitly shows this does not
  generalize to the broader FP-silver population (AUC 0.58–0.65, barely above raw click
  count). Do not market or wire this as "reduces false positives" broadly — it's scoped to
  the confirmed SENSOR_LINK contact-noise signature only.
- **Not a CNN input.** Per the prior turn's discussion: nothing in this investigation
  clears the bar for a model change (no ablation evidence, training happens outside this
  repo anyway). This is purely an `app/health/`/`app/decision/` pipeline addition.
- **Not touching T009's own status/thresholds** — those stay as recalibrated 2026-07-16
  for their own, separate purpose.
- **Not attempting true multi-source burst discrimination** (the "many asynchronous
  larvae" complication raised earlier in this investigation) — this template distinguishes
  bite-like spectral content from contact-noise spectral content; it does not attempt to
  count or characterize individual larvae.

## Risks / open questions

- **Windowing-parity risk (§3)** is the biggest unresolved question — the validated result
  used a different (single-pass, whole-file) computation than what production would
  actually run per-window. Phase A must close this gap and re-validate before anything
  ships; do not skip straight to wiring T010 into `rootcause.py` on the strength of this
  session's numbers alone.
- **FP-silver's heterogeneity is itself informative** but out of scope here — a natural
  follow-up (not part of this plan) would be characterizing what *does* distinguish the
  FP-silver cases that don't look like contact noise, but that's a separate investigation.
- **Template staleness** — like every other fitted config in this codebase, the template
  is specific to the hardware/corpus it was fit against (piezo needle sensor, this
  project's recording chain) and would need refitting if hardware changes materially (see
  `project-piezo-hardware-and-app-defaults` memory on the china-needle → sanded-needle
  evolution).

## Acceptance / validation

- Refit the template on a proper train split, evaluate held-out on T-test/TN-test/
  FAULT-gold/FP-silver **using the real per-window production windowing**, not the
  single-pass method used for this session's exploratory numbers.
- `rootcause.assess_many` replay (the same protocol as the 2026-07-16/2026-07-30
  recalibrations): FAULT-gold (8 files) should mostly resolve SENSOR_LINK; local TN clean
  reference (4 files) should mostly resolve NONE/UNKNOWN. Do not regress either bar
  relative to the current shipped behavior.
- Confirm FP-silver's aggregate SENSOR_LINK rate does **not** rise (this refinement isn't
  meant to relabel general false positives as SENSOR_LINK) and ideally note, descriptively,
  how it behaves — informative, not a pass/fail bar given the non-goal above.
- Synthetic-signal unit tests per project TDD convention for the new check (silence, sine,
  noise, clipped, synthetic click bursts at controlled match-scores) plus a
  missing-template-file fallback test.
