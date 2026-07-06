# Phase 7 — Pre-merge Cleanup (Design)

**Date:** 2026-07-06
**Branch:** `features/audio_signal_health`
**Goal:** Finish the three deferred follow-ups from Phases 1–6, then merge the
Audio Signal Health Monitoring subsystem into `main`.

This ships as one effort in three sub-phases, following the repo convention
(headless engine first, one visible feature per sub-phase, TDD throughout).
The owner commits/pushes manually and performs GUI verification before merge.

## Context

Phases 0–6 are complete (137 tests pass). Three items were explicitly deferred:

1. Wire the classifier's `fmin`/`fmax` into the Settings UI (the lone code TODO
   at `app/audio/features.py:24`).
2. Persist startup ("Validate Acquisition") and anomaly reports to disk.
3. Replace the diagonal RMS z-distance anomaly detector with a true
   full-covariance Mahalanobis distance.

All three are in scope for this pre-merge batch.

---

## 7a — Wire fmin/fmax into Settings UI (smallest)

**What:** Expose the classifier feature-extraction bounds `fmin`/`fmax`
(currently hard-defaulted to 50 / 10000 inside `FeatureExtractor.extract_features`)
as editable Settings fields.

**Design:**
- Add `fmin_var` and `fmax_var` tk variables in `main.py.__init__`, defaulting to
  **50** and **10000** — the current `extract_features` defaults — so classification
  output is unchanged unless the user edits them.
- Add two entry fields to the **Preprocessing** tab of
  `app/ui/settings_dialog.py`, beside the existing `low_cut` / `up_cut` fields.
- Thread `fmin`/`fmax` from those vars through the inference call path in
  `main.py` into `extract_features`.
- Remove the `#TODO` comment at `app/audio/features.py:24`.

**Boundary note:** This item modifies the *classifier's own* feature extraction,
not the additive health layer. Editing the values *can* change classification
output — that is intended and user-controlled. Defaults preserve current behavior.

**Tests:**
- Settings-dialog test asserts the new vars and fields exist and bind.
- A plumbing test confirms the values reach `extract_features` (assert the call
  is constructed with the var values, or via a stubbed extractor).

---

## 7b — Persist reports to `reports/`

**What:** Save each Validate-Acquisition startup result and each anomaly
rising-edge event to disk as timestamped JSON files.

**Design:**
- **Portable core stays pure.** Add `app/health/serialization.py` containing only
  pure functions that turn duck-typed reports into plain dicts
  (`startup_result_to_dict(result)`, `anomaly_event_to_dict(...)`, and any shared
  `report_to_dict` helper). Standard-library `json`-compatible output only; no
  file I/O, no non-stdlib/NumPy imports. Testable under the numpy-only `.venv`.
- **File writing lives in `main.py`** (an app concern): build the `reports/`
  directory on demand, stamp a `YYYYMMDD_HHMMSS` filename with `datetime`, and
  write the serialized dict.
- **Triggers:**
  - Validate Acquisition → after `run_validation(...)`, write
    `reports/startup_YYYYMMDD_HHMMSS.json`.
  - Anomaly rising edge → at the existing `_last_anomalous` rising-edge point in
    `_update_health_indicator`, write `reports/anomaly_YYYYMMDD_HHMMSS.json`.
- **Content:**
  - Startup: overall state, per-window summary, aggregate signal, timestamp,
    source (mic vs WAV path).
  - Anomaly: distance, threshold, is_anomalous, contributors, confidence,
    timestamp, source.
- Add `reports/` to `.gitignore`.

**Tests:**
- Serialization schema / round-trip tests (headless): dict shape, keys, JSON
  serializability, stable ordering.

---

## 7c — Full-covariance Mahalanobis (biggest; profile format bump v1 → v2)

**What:** Replace the diagonal RMS z-distance anomaly detector with a true
Mahalanobis distance that accounts for correlations between measurements.

**Scope boundary:** This replaces the *anomaly* distance path only
(`app/health/anomaly.py`). The per-measurement percentile statistics used by
`calibration_eval.py` (the panel **Cal** column) are untouched. "Replace diagonal
entirely" means: profiles now carry covariance, `anomaly.py` always uses full
Mahalanobis, and there is **no diagonal fallback**. Any profile lacking covariance
(v1) is no longer valid for anomaly detection and must be regenerated.

**Design:**

- **Profile format (`calibration.py`), version → 2:**
  - Add an ordered feature index: a list of `(check_id, measurement_name)` pairs
    defining the vector layout.
  - Add a **mean vector** (length D) and a **covariance matrix** (D×D) computed
    over every calibration window's full feature vector.
  - Existing per-measurement stats/percentiles remain (still used by
    `calibration_eval`).
  - `save`/`load` serialize the new fields; `load` rejects/errors clearly on a v1
    profile used for anomaly (or the anomaly path treats absence of covariance as
    "no profile → unchanged behavior", consistent with the fusion contract).

- **`calibrate.py`:** while iterating windows, accumulate each window's full
  feature vector in the fixed index order; after the pass, compute the mean vector
  and covariance (`np.cov`, rows=features). Store both plus the index in the JSON.

- **`anomaly.py` — `detect_anomaly(results, profile, *, p=0.001)`:**
  - Build the current window's feature vector in the profile's index order.
  - If some measurements are missing this window, subselect the available
    dimensions and the corresponding covariance submatrix.
  - Regularize Σ with `εI` (small ε relative to its trace) before inversion to
    handle singular/ill-conditioned matrices; use a solve rather than explicit
    inverse where practical.
  - Distance: `d² = (x − μ)ᵀ Σ⁻¹ (x − μ)`, `d = sqrt(d²)`.
  - **Threshold via chi-square p-value:** treat `d²` as chi-square with
    `df = D` (the number of dimensions actually used). Flag anomalous when
    `d² > chi2_crit(1 − p, df)`, default `p = 0.001`. Compute the critical value
    with a small **pure-Python** inverse regularized-incomplete-gamma helper
    (`math` only — preserves the stdlib+NumPy rule). `d_thresh = sqrt(chi2_crit)`.
  - **Confidence** keeps its current linear shape, rebased on the chi-square
    threshold: `confidence = max(0.0, 1.0 − d / (2 · d_thresh))`
    (1.0 at the centroid, 0.5 at the threshold, 0.0 at twice the threshold).
  - **Contributors:** top-3 dimensions by their component of the quadratic form
    (per-dimension contribution to `d²`), reported as `(label, contribution)`.
  - Returns `None` when no profile covariance is available or no measurements
    match (consistent with today's "no profile → unchanged" behavior).

- **`fusion.py` unchanged:** `decide(results, calibration_evaluation=None,
  anomaly_result=None)` still sets **confidence only, never state**. `AnomalyResult`
  keeps its field shape (`distance`, `threshold`, `is_anomalous`, `contributors`,
  `confidence`) so `main.py`/pipeline/UI need no structural change (confidence is
  already surfaced from Phase 6b).

- **Regenerate `models/9_1_2/calibration.json`** from
  `test_data/audio_signal_health/sanded_needle_1/` via `calibrate.py`, run headless
  in `.venv` (numpy-only, no tflite/librosa needed for the health feature path).
  Bump the profile `version` to 2. **Assumption:** `sanded_needle_1` is the correct
  source for the `sanded_piezo_9_1_2` profile; the regenerated profile is visible in
  owner review before merge.

**Tests:**
- Calibration: covariance + mean vector + index are stored, round-trip through
  save/load, and have correct shape/order.
- Anomaly: centroid → `d ≈ 0`, `confidence ≈ 1`; a point at the chi-square
  threshold sits at the `is_anomalous` boundary; a **correlated-feature** case that
  full covariance flags but diagonal z-distance would miss; singular-Σ input is
  regularized without error; missing-feature subselection uses the right submatrix.
- Chi-square helper: `chi2_crit` matches known values (e.g. `chi2_ppf(0.95, 1) =
  3.841`, `chi2_ppf(0.95, 2) = 5.991`) within tolerance.
- Update `tests/health/test_anomaly.py`: replace diagonal-specific assertions.

---

## Ordering & completion

1. 7a → 2. 7b → 3. 7c → 4. regenerate `9_1_2` profile → 5. full `pytest`
(target: all prior 137 tests plus new tests green) → 6. hand to owner for GUI
verification and merge to `main`.

## Non-goals / deferred (unchanged)

- Reworking `calibration_eval`'s percentile rule.
- Any classifier/model changes beyond exposing existing `fmin`/`fmax`.
- A reports viewer/UI (files are written to `reports/`; browsing is out of scope).
