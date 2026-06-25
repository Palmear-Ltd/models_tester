# Phase 4a — History Mechanism & Stability Checks

**Date:** 2026-06-25
**Status:** Approved (design); ready for implementation planning
**Scope:** Headless. Adds cross-window history to the pipeline and the three stability checks (S001–S003). No UI; Runtime Monitoring is 4b, the timeline plot is 4c.

## 1. Purpose

Stability checks (spec §4.10) detect intermittent faults and gradual degradation by looking at how measurements evolve **across windows**, not within a single one. The current check interface (`run(window, features)`) only sees one window, so the pipeline must carry a bounded history that stability checks can read.

## 2. Pipeline History (decision: pipeline-maintained, injected via `features`)

`HealthAnalysisPipeline.__init__(manager=None, calibration_profile=None, history_length=20)` holds `self._history`, a `collections.deque(maxlen=history_length)` of **measurement snapshots**, one per analysed window: `{check_id: {measurement_name: float}}`.

In `analyze`:
1. `features = self._prepare_features(window)`.
2. Inject `features["history"] = list(self._history)` — the prior windows, oldest→newest (does **not** include the current window).
3. Run checks (stability checks read `features["history"]`).
4. Build the current window's snapshot from `results` (`{r.check_id: {m.name: m.value for m in r.measurements}}`) and `self._history.append(...)`.

History resets when the pipeline is rebuilt (profile/calibration change) — acceptable. During calibration generation the history accumulates across the calibration windows, so stability measurements get characterized too.

## 3. Stability Checks (`app/health/checks/stability.py`)

All **`category = CheckCategory.SUPPORTING`**, stateless, NumPy-only. Each reads `features.get("history", [])` and extracts its source measurement series; if the series has fewer than `min_samples` values (warm-up) or the source measurement is absent (that check disabled), it returns **PASS** with the measurement set to `0.0` and no diagnostics.

- **S001 Energy Stability** (`check_id "S001"`, name "Energy Stability"): series = `T002.rms` across history. `energy_cv = std/mean` (0 if mean≤0). WARNING if `energy_cv > max_variation` (default `0.5`). Measurement: `energy_cv`. Params: `min_samples=5`, `max_variation=0.5`.
- **S002 Spectral Stability** (`check_id "S002"`, name "Spectral Stability"): series = `F001.spectral_centroid`. `centroid_cv = std/mean`. WARNING if `> max_variation` (default `0.3`). Measurement: `centroid_cv`. Params: `min_samples=5`, `max_variation=0.3`.
- **S003 Long-Term Noise Floor** (`check_id "S003"`, name "Long-Term Noise Floor"): series = `T002.rms`. `noise_floor = 25th percentile` of the series. WARNING if `noise_floor > max_noise_floor` (default `0.05`). Measurement: `noise_floor`. Params: `min_samples=5`, `max_noise_floor=0.05`.

Series extraction: for each snapshot in history, read `snap.get(<check_id>, {}).get(<measurement>)`; keep non-`None` floats. Thresholds are provisional manual defaults (calibration can characterize/tune later, as for other checks).

## 4. Registry / Profiles

Add to `app/health/config.py` `REGISTRY` (a new config group `"stability"`, none mandatory):
`CheckSpec("S001", EnergyStabilityCheck, "stability")`, `("S002", SpectralStabilityCheck, "stability")`, `("S003", LongTermNoiseFloorCheck, "stability")`.

Resulting profile check counts: `development` 14, `diagnostic` 14, `production` 13 (still drops F004), `minimal` 3 (categories disabled → only mandatory). The count assertions in `tests/health/test_config.py` (and any pipeline/defaults tests asserting 11/`set` of ids) are updated accordingly. `app/health/defaults.py` is left as-is (legacy 11-check helper used only by its own tests, which keep asserting 11 — it is not the live path).

## 5. Out of Scope

- Runtime Monitoring (events, persistence, recovery, smoothed state) — Phase 4b.
- Health timeline plot — Phase 4c.
- No change to existing checks, fusion, or calibration logic.

## 6. Testing (headless)

- **Stability checks:** fed synthetic `features["history"]` (a list of snapshots) — a stable `T002.rms` series → PASS; a jittery series → WARNING; fewer than `min_samples` → PASS; missing source measurement → PASS. Direct assertions on the measurement value (`energy_cv`, `centroid_cv`, `noise_floor`).
- **Pipeline history:** repeated `analyze` calls accumulate snapshots (bounded by `history_length`); `features["history"]` passed to checks contains prior windows only; with stability checks registered, a run after enough windows produces their results.
- **Config:** registry has 14 entries; profile counts as in §4.
- Full suite stays green (no UI touched).
