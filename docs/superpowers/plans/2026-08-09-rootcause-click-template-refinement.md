# Plan: root-cause SENSOR_LINK refinement via locally-fit click spectral template

Spec: `docs/superpowers/specs/2026-08-09-rootcause-click-template-refinement-design.md`

Ships in two sub-phases, headless engine first (per project convention). **Phase A's
first job is closing the windowing-parity gap flagged in the spec (§3) — do not proceed
to wiring anything into `rootcause.py` until that's verified**, since everything after it
depends on the per-window statistic actually preserving the signal found this session.

## Phase A — headless: windowing-parity check, template fit, new check, wiring

**Goal:** `rootcause.assess_many` scores SENSOR_LINK using a locally-fit click spectral
template instead of raw click count alone, validated under real production windowing —
or, if that doesn't hold up, a documented decision to defer (see step 1's fallback).

1. **Formalize this session's scan/fit scripts and re-run under real windowing.**
   - `scripts/click_template_scan.py` (already added) extracts per-click spectra via
     **single-pass whole-file detection** — this must be changed to replicate
     `offline_score.py`'s exact rolling-buffer windowing (2.5s window, 0.5s hop,
     buffer-fill hold-off, 20s cap — same pattern already used in `scripts/rms_scan.py`
     and `scripts/click_scan.py`) so per-window `matched_click_count` is computed exactly
     as the live pipeline would produce it.
   - Rebuild the differential template and refit the per-click match threshold (Youden's
     J on train) using this real-windowed extraction, same train/test split discipline
     (stratified by label, 2021 excluded, seed=42) as this session's exploration.
   - **Decision gate:** re-run the T-test vs. FAULT-gold vs. FP-silver evaluation from
     this session under the real windowing. If the AUC lift over raw click count
     (0.71–0.72 vs. FAULT-gold in this session's exploration) survives, proceed. If it
     collapses the way the attack/decay-shape hypothesis did in an earlier round of this
     investigation, stop and report back — the fallback (session-level burst-grouping
     with raw click timestamps threaded through `assess_many`) is a materially bigger
     change and shouldn't be built speculatively.
   - Scale the train corpus further if useful (this session used 183 T / 200 TN files;
     more is available in `9_1_4/audio_data/wav/{T,F}` if the smaller sample proves
     unstable across reruns).

2. **New check: `T010` (tentative ID — confirm it's free in the registry).**
   - `app/health/checks/time_domain.py` (or a new module if that file is getting
     crowded): re-run the same click-detection algorithm as `ClickTransientCheck`
     (independent, not a T009 dependency — see spec §1) on `window.samples`, score each
     click against the loaded template, emit `Measurement("matched_click_count", ...)`.
   - PASS/WARNING/FAIL thresholds on `matched_click_count`, loaded from
     `check_thresholds.json` (new `"T010"` entry) via the existing loader — fit the
     actual warn/fault numbers from the same corpus replay as step 1, not guessed.
   - Declare `category = CheckCategory.PRIMARY` as a class attribute only (existing hard
     convention — do not pass `category=` into the result).

3. **Template data file + loader.**
   - `app/health/click_template.json`: `{n_bins, freq_max_hz, template, match_threshold,
     fitted_against, fitted_date}` per the spec's schema.
   - Loader in `app/health/config.py` (or co-located with T010's check class), same
     load-JSON-with-fallback idiom as `app/decision/threshold.py:default_config` — a
     missing/malformed file must never raise, just disables T010's matching (falls back
     to PASS-through / UNKNOWN territory, not a crash).

4. **`rootcause.py` weight table + session cutoff refit.**
   - Add `(T010, WARNING)` / `(T010, FAIL)` rows to `_WEIGHT_TABLE` (`rootcause.py:78`).
   - Refit `DEFAULT_SESSION_CUTOFF` (`rootcause.py:131`, currently 0.25) against the new
     combined table, replayed through the *actual* per-window pipeline on the same local
     corpora as the 2026-07-16/2026-07-30 recalibrations (`test_data/F` TN ×4,
     `test_data/audio_signal_health/fp/F` FAULT ×8). Document the refit informally, same
     caveat style as the existing `DEFAULT_SESSION_CUTOFF` comment block.
   - Whether T009's existing weight in the table should shrink now that T010 covers the
     same failure mode with a better signal is an empirical call — decide from the refit
     replay, not by intuition; document either way in a code comment.

5. **Tests (write first, per project TDD convention).**
   - `tests/health/test_time_domain.py` (or wherever T010 lands): synthetic click bursts
     with controlled spectral content straddling the fitted match threshold — assert
     PASS/WARNING/FAIL land where expected. Include a template-missing-file fallback
     test.
   - `tests/health/test_rootcause.py`: extend the existing corpus-replay tests
     (skip-gracefully pattern already used there) to assert FAULT-gold mostly resolves
     SENSOR_LINK and local TN clean mostly resolves NONE/UNKNOWN with T010 wired in — the
     spec's acceptance bar, not a regression of the current 5/8 and 4/4 rates from the
     2026-07-30 refit.
   - Full suite must still pass: `.venv/bin/python -m pytest tests/ -q`.

6. **Verify.** Headless checks per `CLAUDE.md`
   (`ast.parse`/`import main`), plus the `rootcause.assess_many` replay from step 4/5
   printed for manual review before calling Phase A done.

## Phase B — transparency + FP-silver characterization note

**Goal:** the new signal's provenance is auditable, and its known limitation (doesn't
generalize past confirmed SENSOR_LINK) is visible in-repo, not just in this plan.

1. One-line startup log for which `click_template.json` was loaded (or "shipped
   default"/"disabled — no template"), matching the existing transparency pattern next to
   the decision-cutoff log block (`main.py:539-549`).
2. Add a short note to `app/health/rootcause.py`'s module docstring (or a comment near
   `_WEIGHT_TABLE`) stating T010's validated scope explicitly — confirmed SENSOR_LINK
   contact-noise signature only, not general FP explanation — so a future reader doesn't
   assume it does more than it does (mirrors this plan's own non-goals section).
3. Headless verify only; owner does the manual GUI check per `CLAUDE.md`.

## Explicitly deferred (not in this plan)

- Session-level true burst-grouping (raw click timestamps threaded through
  `assess_many`) — only pursued if Phase A step 1's decision gate fails.
- Characterizing what actually drives the FP-silver cases that aren't SENSOR_LINK-like —
  a separate investigation, not this refinement.
- Any CNN-facing change — out of scope per the spec's non-goals and the prior turn's
  broader conclusion in this investigation.
