# Changelog

Notable changes to the Palmear Audio Testing Tool. Dates are when the work landed on
`main`. This project doesn't use version numbers yet, so entries are grouped by date.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## 2026-09-13 — 9_1_5 candidate model (thin-cable retrain)

Added `models/9_1_5/` — a candidate retrain for the new thin piezo→preamp cable,
selectable in the tester's model dropdown next to 9_1_2. Trained in the sibling
`training/9_1_5/` project (app-parity features, 2025–2026 field data plus the
Aug/Sep 2026 cable/housing sessions; full write-up and every number in that
project's `README.md`). Ships with its own refit `scaler.json`, an EWMA-peak
`decision_threshold.json` (span 8, cutoff 0.50 — the owner's choice, not the
val-fitted 0.742) and `model_params.json` with `labelScore` refit to 0.17. No
`rms_confidence.json` / `calibration.json` yet, so the tester logs the
shipped-default fallback for those. Session-level on the 9_1_5 test split it
runs FNR 0.158 / FPR 0.188 (bal. acc 0.827, AUC 0.902) against 9_1_2's
0.115 / 0.336 (0.774, 0.865) at 9_1_2's shipped cutoff. See
`models/9_1_5/README.md` for the known soft spot.

## 2026-08-12 — RMS verdict confidence + refined sensor-link click detection

Both features grew out of a corpus-wide investigation into whether signal properties
already computed live (RMS energy, click timing) but never used for anything beyond
display could help judge trustworthiness. Several hypotheses were tried and discarded
along the way (individual click waveform shape, literature-derived frequency-band
matching) before landing on what actually held up under held-out testing — see
`docs/superpowers/specs/2026-08-09-rms-verdict-confidence-design.md` and
`docs/superpowers/specs/2026-08-09-rootcause-click-template-refinement-design.md` for
the full evidence trail.

**⚠️ Read the "New Advisory Signals" section in [`README.md`](README.md) before relying
on either of these in the field — both are real, evidence-based, and purely additive
(neither ever changes a diagnosis), but neither has been through a field validation round
yet.**

### Added
- **Verdict confidence**: a low/medium/high trust tier shown alongside the session
  diagnosis (e.g. *"INFESTED (EWMA peak: 0.71) — verdict confidence: low"*), built from
  two logistic curves (one per verdict branch) mapping session RMS energy to P(verdict
  correct). Session RMS turned out to be a weak *direct* predictor of infestation but a
  much stronger predictor of *which way* the model's verdict tends to be wrong — false
  positives run unusually loud, false negatives run unusually quiet.
  (`app/decision/rms_confidence.py`, `models/9_1_2/rms_confidence.json`)
- **T010 `ClickSpectralMatchCheck`**: refines the sensor/cable-link "Likely cause"
  detector. The previous version judged clicking by raw count, which can't tell a real
  insect bite from sensor/cable contact noise — confirmed-fault recordings click *even
  more* than genuinely infested ones. T010 instead scores each click's spectrum against
  a template fit from this project's own labeled recordings (real-infestation click
  spectra vs. clean-baseline click spectra) and judges a window by how much of its
  clicking actually looks bite-shaped. Validated specifically against confirmed
  sensor-fault recordings — see the advisory note above for its scope.
  (`app/health/checks/time_domain.py`, `app/health/click_template.json`,
  `fit_click_template.py`)
- Both signals' saved session results now include the new fields (`verdict_confidence_tier`,
  `verdict_confidence_probability`) alongside the existing decision fields.
- New corpus tooling: `scripts/rms_scan.py`, `scripts/click_scan.py`,
  `scripts/click_template_scan*.py` — backfill RMS/click features onto an
  already-scored manifest for offline analysis and refitting.

### Changed
- `rootcause.py`'s SENSOR_LINK scoring weight table gained a T010 entry; the session-level
  decision cutoff was refit accordingly (`0.25` → `0.75`) against the same confirmed-fault
  and clean-reference recordings used for every prior recalibration of this module.
- `evaluate_decision_rules.py` gained `--rms-manifest`/`--rms-confidence-out` flags to fit
  and evaluate the verdict-confidence curves on the same train/test split already used for
  the decision cutoff.

## 2026-08-06/07 — decision-cutoff tooling + manual test procedure

### Added
- Per-window EWMA/peak/cutoff recording and plotting in the live tester, so a session's
  decision trace is visible after the fact, not just the final verdict.
- EWMA span sweep in the offline cutoff-refit tooling — different models turned out to
  want different smoothing spans rather than one span fixed across all of them.
- Fitted decision cutoffs shipped for the `9_1_1` and `one_shot` models.
- A documented manual test procedure for live-mic + WAV-replay parity checks.

### Fixed
- One-shot models now score correctly in `offline_score.py` (previously used the
  sliding-window path unconditionally).

## 2026-07-30 — buffer hold-off fix

### Fixed
- A session's first ~4 windows were being built from a still-partly-zero rolling audio
  buffer, spuriously tripping the click detector and inflating every session's
  sensor-link score by a roughly constant floor. Inference and health analysis now wait
  until the rolling 2.5s buffer has filled once with real audio before scoring anything.
  The sensor-link session cutoff was recalibrated accordingly (this is the fix the
  2026-08-12 refit above built further on top of).

## 2026-07-16 — sensor-link detector recalibration

### Fixed
- `rootcause.py`'s SENSOR_LINK attribution (behind "Validate Acquisition" and the live
  "Likely cause" label) was firing at confidence 1.00 on *every* recording — fault,
  clean, and infested alike — making it useless as a signal. Root cause: the click-count
  thresholds sat inside the noise floor of genuinely healthy piezo contact, and the
  session-level rule fired on any positive score with no baseline-relative floor. Fixed
  by recalibrating the per-window thresholds (informed by a new multi-year healthy
  calibration profile) and requiring a session's *mean* score to clear a data-driven
  cutoff instead of merely being positive.

### Added
- A "Refresh" button for the microphone device list.
- A transparency log line reporting which health-threshold config is active.

## 2026-07-13 — EWMA-peak decision rule

### Changed
- Replaced the hand-tuned threshold + positive-count-band infestation decision with an
  EWMA-smoothed peak score compared against a single data-driven cutoff — matched or
  slightly exceeded the old rule's accuracy across ~6,500 labeled sessions while
  replacing three manually-tuned numbers with one.

## 2026-06-23 → 2026-07-07 — Audio Signal Health Monitoring (Phases 0–7)

Built the `app/health/` subsystem: a signal-quality layer that judges whether a captured
recording is *trustworthy* (sensor/cable/hardware health), running alongside the
classifier and never blocking, slowing, or changing its results.

### Added
- Time-domain checks (flatline, clipping, DC bias, dropout, click transients, crest
  factor) and frequency-domain checks (hum, spectral drift, harmonic resonance), fused
  into a single health state with a live indicator.
- Calibration: generate a per-device baseline profile from known-good recordings,
  evaluate new recordings against it. Later upgraded to full-covariance Mahalanobis
  distance for anomaly detection.
- Stability checks across consecutive recordings, a debounced runtime monitor, and a
  health timeline plot.
- Startup validation ("Validate Acquisition" button) — a ~20s capture scored for overall
  signal health before a real session starts.
- Root-cause attribution: a plain-language "Likely cause: SENSOR_LINK — ..." explanation
  when the health checks point at a physical sensor/cable fault specifically, as distinct
  from other signal-health issues.

(Deferred at the time, since addressed above: persisting startup/anomaly reports landed
2026-07-06; the sensor-link false-positive-on-everything bug was fixed 2026-07-16.)

## Earlier

Model support (one-shot models, model dropdown in Settings), audio preprocessing parity
with the mobile app's Dart implementation, PCEN normalization, and the initial
Tkinter tester UI — see `git log` for the full history predating this changelog.
