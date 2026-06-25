# Phase 6 — Anomaly Detection & Confidence

**Date:** 2026-06-25
**Status:** Approved (design); ready for implementation planning
**Scope:** Fill pipeline Stage 5 (anomaly detection) and the deferred "confidence reflects checks only (Phase 6)" gap. Split into 6a (headless detection + fusion confidence) and 6b (surface confidence/anomaly in the UI). No change to checks, calibration evaluation, or the state machine.

## 1. Purpose

Anomaly detection gives a single holistic "how normal is this window overall" measure relative to the calibration profile, and turns it into the report's **confidence**. It complements — does not duplicate — `calibration_eval` (Phase 3b), which already drives *state* from per-measurement percentile deviations.

## 2. Decisions (from brainstorming)

- **Diagonal Mahalanobis = RMS z-distance.** Calibration profiles store only per-measurement stats (`mean, std, min, max, p5, p95`), no covariance matrix, so full Mahalanobis is out of scope. For each measurement present in the profile with `std > 0`: `z = (value − mean) / std`; `distance = sqrt(mean(zᵢ²))` (RMS keeps the threshold interpretable regardless of measurement count).
- **Anomaly feeds CONFIDENCE, not state.** `final_state` is unchanged by anomaly (calibration_eval owns profile-driven escalation). No profile → confidence stays checks-based (today's behavior); profile present → confidence reflects the anomaly distance.

## 3. Phase 6a — Headless detection + fusion (`app/health/anomaly.py`)

Pure NumPy + `app.health.models`/`app.health.calibration` types.

- `AnomalyResult(distance: float, threshold: float, is_anomalous: bool, contributors: list[tuple[str, float]], confidence: float)` — `contributors` = top-3 `(label, z)` by `|z|`, `label = "{check_id}.{measurement}"`.
- `detect_anomaly(results, profile, *, threshold=3.0) -> AnomalyResult | None`:
  - For each result's measurement, look up `profile.statistics[check_id][measurement_name]`; if found and `std > 0`, collect `z = (value − mean)/std`.
  - If no measurements matched → return `None` (cannot score).
  - `distance = sqrt(mean(z²))`; `is_anomalous = distance > threshold`.
  - `confidence = max(0.0, 1.0 − distance / (2·threshold))` — `1.0` at the centroid, `0.5` at the threshold, `0.0` at twice the threshold.
  - `contributors` = top-3 measurements by `|z|`.
- **Pipeline Stage 5:** `_detect_anomalies(features, results)` returns `detect_anomaly(results, self.calibration_profile)` when a profile is set, else `None`.
- **Fusion:** `decide(results, calibration_evaluation=None, anomaly_result=None)`. After the existing state/confidence/calibration logic, if `anomaly_result is not None`: set `confidence = anomaly_result.confidence` and append an anomaly note to the summary (distance + ANOMALOUS/normal + top contributor). State is untouched. Duck-typed (no anomaly import in fusion). The pipeline passes `anomaly_result` into `decide`.

## 4. Phase 6b — Surface confidence/anomaly (`main.py`)

- The Signal Health indicator shows the confidence, e.g. **"Signal Health: OK · conf 0.92"**.
- When `report.anomaly_result` is anomalous, log it once on transition (e.g. `[anomaly] distance 4.1 — F001.spectral_centroid z=4.8`).
- No new plots; engine untouched. (Visible feature for the phase.)

## 5. Out of Scope

- No covariance matrix / true Mahalanobis (would require a profile-format change + recompute).
- Anomaly does not change `final_state`.
- No persistence of anomaly history.

## 6. Testing

- **6a (headless):** `detect_anomaly` — no profile / no matching measurements → `None`; a window at the profile mean → `distance≈0`, `confidence≈1`, not anomalous; a window many sigma out → `distance` large, `is_anomalous True`, `confidence 0`, correct top contributor; `std≤0` measurements skipped. `decide` — `anomaly_result=None` → unchanged confidence; with an `AnomalyResult` → `confidence == anomaly_result.confidence`, state unchanged, summary contains the anomaly note. Pipeline — with a profile, `report.anomaly_result` is populated and `report.confidence` matches it; without a profile, `anomaly_result is None` and behavior is unchanged.
- **6b (UI):** `ast.parse` + `import main` + unchanged suite; manual GUI — load a profile, run; the indicator shows `· conf X.XX`, and an off-profile signal logs an `[anomaly]` line and drops the confidence.
