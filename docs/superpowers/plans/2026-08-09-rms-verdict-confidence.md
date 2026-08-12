# Plan: RMS-based verdict confidence

Spec: `docs/superpowers/specs/2026-08-09-rms-verdict-confidence-design.md`

Ships in two sub-phases, headless engine first (per project convention), each leaving the
tester working.

## Phase A — headless: RMS capture, curve fitting, core module, minimal wiring

**Goal:** a session's verdict is accompanied by a fitted low/medium/high trust tier
(`main.py:diag_label` text, no new UI widgets). `state`/`predicted_infested` unchanged on
every replayed file. No new manual GUI verification needed beyond the usual headless
checks (owner does the live GUI smoke test per `CLAUDE.md`).

1. **Formalize `scripts/rms_scan.py` as the permanent RMS-backfill tool.**
   - Already exists (this session's investigation) and already mirrors
     `offline_score.py`'s exact windowing/hold-off/cap logic, plus a `--bandpass` flag.
     Clean up: module docstring should point at this plan/spec instead of reading as a
     one-off investigation script; confirm `.rms_cache/` (currently untracked, 28MB) gets
     a `.gitignore` entry alongside `.score_cache/`.
   - No change to `offline_score.py` / `SCORING_REV` — keep RMS scanning a separate, much
     cheaper pass (see spec §6) rather than forcing a full model rescore to add it.

2. **Extend `evaluate_decision_rules.py` with an RMS-confidence fitting stage.**
   - After the existing cutoff fit (`stratified_split`, seed=42 — reuse the identical
     split, don't re-derive it), join each record's session mean-RMS (from
     `scripts/rms_scan.py`'s output, joined by `path`) to its `SessionRecord`.
   - Using the *train* split only: compute each session's verdict under the *already
     fitted* `ewma_peak` cutoff, split into the INFESTED pool (TP/FP) and HEALTHY pool
     (TN/FN), and fit `sigmoid(a*log(mean_rms)+b)` separately per pool (try mean_rms and
     peak_rms, keep whichever gets the better held-out-*test*-split AUC — see spec §1).
   - Evaluate on the *test* split only (same discipline as the existing cutoff eval — no
     data-driven choice touches the test fold) and print pool AUCs, matching the existing
     `evaluation.log` reporting style.
   - Choose trust-tier cutoffs (low/medium/high) on the fitted probability using the test
     split's distribution — document the chosen edges and *why* in a code comment (mirrors
     `ThresholdConfig`'s own docstring-caveat style), not silently hardcoded.
   - Add `--rms-manifest <path>` and `--rms-confidence-out <path>` CLI flags, writing a
     JSON with schema `{a_infested, b_infested, a_healthy, b_healthy, feature: "mean_rms"|"peak_rms", tier_edges: [lo, hi], n_infested_fit, n_healthy_fit}` —
     the `n_*_fit` fields matter for the FN-thinness risk flagged in the spec (§Risks):
     if `n_healthy_fit`'s minority class (FN) count is very small, note that in the
     written JSON so it's inspectable later, and prefer widening the trust-tier band
     (biasing toward "medium" over a confident "low"/"high") rather than a sharp cutoff
     when the fit is that thin — decide the exact tier-edge behavior from the actual fit
     diagnostics at implementation time.

3. **New module `app/decision/rms_confidence.py`** (mirrors `app/decision/threshold.py`):
   - `RmsConfidenceConfig` frozen dataclass: `a_infested, b_infested, a_healthy, b_healthy,
     feature, tier_edges` + shipped `DEFAULT_*` constants (fit against `models/9_1_2`,
     documented the same way `DEFAULT_CUTOFF`/`DEFAULT_SPAN` are).
   - `to_json`/`from_json`, `default_config(path=None)` — identical fallback-on-missing-
     file idiom to `app/decision/threshold.py:default_config`.
   - `estimate(state: str, mean_rms: float, peak_rms: float, config: RmsConfidenceConfig) ->
     tuple[float, str]` returning `(probability, tier)`, selecting `mean_rms` or
     `peak_rms` per `config.feature`, applying the branch matching `state`
     (`"INFESTED"` → `a_infested/b_infested`, `"HEALTHY"` → `a_healthy/b_healthy`).
     Never raises on `state` values outside `{"INFESTED","HEALTHY"}` — return a neutral
     `(0.5, "medium")` (matches `HealthState.UNKNOWN`'s spirit) rather than crash the
     session-end path.

4. **Wire into `main.py`.**
   - `load_resources()`: load `models/<model>/rms_confidence.json` next to
     `decision_threshold.json` (same directory-relative lookup as
     `main.py:536-549`), with the identical transparency log pattern — "fitted for this
     model" vs. "SHIPPED DEFAULT — not fitted for `<model>`".
   - `calculate_diagnosis()` (`main.py:629-651`, sliding-window branch only — single-shot
     mode has no `energy_history`-style session RMS trace to condition on, leave it
     unchanged): after `state`/`peak` are resolved, compute
     `mean_session_rms = mean(v for _, v in self.energy_history)` (guard the empty-history
     case — return neutral tier, don't divide by zero), call
     `rms_confidence.estimate(state, mean_session_rms, peak_session_rms, self.rms_confidence_config)`,
     and append the tier to `self.diag_label`'s existing text, e.g.
     `f"{state} (EWMA peak: {peak:.3f}) — verdict confidence: {tier}"`. Do **not** change
     `predicted_infested`'s return value or any other branch's behavior.

5. **Tests (write first, per project TDD convention).**
   - `tests/decision/test_rms_confidence.py`: monotonicity (higher RMS → lower confidence
     for `state="INFESTED"`, higher confidence for `state="HEALTHY"`, and vice versa at low
     RMS); JSON round-trip; missing-file fallback to shipped default; unknown `state` →
     neutral tier, no exception.
   - `tests/test_main_decision_wiring.py` (existing file, extend): `calculate_diagnosis`
     appends a trust tier to `diag_label` text without changing `predicted_infested`'s
     return value, across a few synthetic `energy_history` traces.
   - Corpus-replay check (skip-gracefully pattern, same as `test_rootcause.py`'s corpus
     tests): on the local `test_data/{T,F}` sample, mean fitted confidence should be
     higher for TP than FP within the INFESTED pool, and higher for TN than FN within the
     HEALTHY pool — direction-only assertion, explicitly documented as thin (small local n,
     matches the spec's own caveat).
   - Full suite must still pass: `.venv/bin/python -m pytest tests/ -q`.

6. **Verify.** `.venv/bin/python -c "import ast; ast.parse(open('main.py').read())"` then
   `.venv/bin/python -c "import main"`; confirm `state`/`predicted_infested` are unchanged
   from pre-Phase-A on a small before/after replay of `test_data/{T,F}`.

## Phase B — transparency + persistence polish

**Goal:** the fitted config in use is auditable and the trust tier survives into saved
session results, matching how `decision_method`/`decision_ewma_peak` are already persisted
(`main.py:1166-1170`). No behavior change to the verdict itself.

1. Add `"verdict_confidence_tier"` and `"verdict_confidence_probability"` to the
   `save_results()` JSON payload (`main.py:1166-1170` area), alongside the existing
   `decision_method`/`decision_ewma_peak` fields.
2. One-line startup log for which `rms_confidence.json` was loaded (or "shipped default"),
   matching the existing `Decision cutoff: ...` / `source: ...` two-line block at
   `main.py:539-549` — same visual style, placed right after it.
3. Headless verify only (owner does the manual GUI check per `CLAUDE.md`): re-run the
   Phase A verify steps, confirm test count only increased.

## Explicitly deferred (not in this plan)

- Any fusion between this confidence and `app/health/`'s anomaly-confidence — spec flags
  this as a follow-up if the two independent numbers prove confusing side by side in
  practice, not a design decision to make blind now.
- Surfacing the trust tier inside the `app/health/` panel/report itself — this signal is
  session-level and decision-owned, not per-window and health-owned (spec §3); if the
  owner later wants it visually grouped with the health panel, that's a UI-only follow-up,
  not a re-architecture.
