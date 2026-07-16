# Plan: root-cause threshold recalibration + persisted config

Spec: `docs/superpowers/specs/2026-07-16-rootcause-threshold-recalibration-design.md`

Ships in two sub-phases, headless engine first (per project convention), each
leaving the tester working. A third, unrelated small UI item (refresh-devices
button) rides along per owner request — implemented directly, no sub-phase needed.

## Phase A — headless: persisted thresholds + fixed session-level decision

**Goal:** `rootcause.assess_many` stops firing SENSOR_LINK on every recording;
clean-vs-fault separation on the local corpora becomes real (see acceptance below).
No UI changes in this phase.

1. **New persisted config: check-threshold overrides.**
   - `app/health/check_thresholds.json` (new file, repo-committed default)
     schema: `{"T009": {"warn_count": <int>, "fault_count": <int>}, ...}` — only
     entries that differ from a check's class defaults need to be present.
   - Loader in `app/health/config.py`: `load_check_thresholds(path=None) ->
     dict[str, dict]`, falling back to a shipped default dict (mirrors
     `app/decision/threshold.py`'s `default_config()` fallback-on-missing-file
     idiom) if the file is absent — never raises for a missing file.
   - Wire into `_profile_configs()` (or a new helper `build_manager`/
     `pipeline_for_profile` calls into) so `HealthConfig.checks["T009"].params`
     gets populated from this loader automatically — every existing caller
     (`main.py`'s `pipeline_for_profile("development", ...)`, tests) picks up the
     new defaults with no call-site changes required.
   - Derive the actual numbers from `calibration_profiles/multiyear_healthy_v1.json`'s
     already-committed `statistics.T009.click_count` (mean 10.166, p95 43) per the
     spec's proposal (`warn_count` anchored near p95, `fault_count` ~2x that) —
     land on exact values via the synthetic + corpus-replay tests in step 3, not
     by inspection alone. Leave S004 untouched (already well-calibrated per spec).
     Spot-check T008's `fault_ratio` against `dropout_frame_ratio` p95=0 but only
     change it if the corpus replay in step 3 shows it's actually contributing to
     the saturation (T009 is the confirmed primary offender).

2. **Fix `assess_many`'s session-level decision.**
   - Add `app/health/rootcause_session_config.json` (new file) + a
     `RootCauseSessionConfig` dataclass in `rootcause.py` (or a new small module
     if that gets crowded) mirroring `ThresholdConfig`/`default_config()` in
     `app/decision/threshold.py` exactly — `cutoff` + `to_json`/`from_json` +
     `default_config(path=None)` with a shipped fallback constant.
   - Replace the current `elif score <= 0.0: UNKNOWN else: SENSOR_LINK` /
     `confidence = min(1.0, score / MAX_SCORE)` in `_finalize` (called from both
     `assess_results` — single window — and `assess_many` — session) with a
     comparison against the persisted session cutoff. Keep `assess_results`
     (single-window, used for the *live* per-window "Likely cause" label) working
     the way it currently does or nearly so — the saturation bug is specifically
     `assess_many` summing across ~40 windows against a per-window-sized
     `MAX_SCORE`; a single window's score against `MAX_SCORE=10` is not obviously
     broken the same way, confirm with a test either way before changing it.
   - Fit the actual cutoff value empirically: replay the local TN reference
     (`test_data/F/*.wav`) and fp/F fault corpus
     (`test_data/audio_signal_health/fp/F/*.wav`) through the *new* per-window
     thresholds from step 1, compute session-combined scores for each, and pick a
     cutoff that separates them (informal, n=12 — good enough to fix "saturates on
     everything," not a statistically rigorous cutoff; document the informality in
     a code comment, same spirit as `ThresholdConfig`'s own docstring caveats).

3. **Tests (write first, per project TDD convention).**
   - `tests/health/test_time_domain.py`: synthetic click-burst signals at
     controlled rates straddling old (3/15) and new thresholds — assert PASS
     below new `warn_count`, WARNING between, FAIL above. Keep/adapt any existing
     T009 tests that assumed the old defaults.
   - `tests/health/test_rootcause.py`: synthetic multi-window sessions — an
     all-PASS session, a session with occasional isolated WARNINGs (simulating
     baseline piezo noise), and a session with a persistent fault pattern —
     assert the first two resolve NONE/UNKNOWN and the third resolves
     SENSOR_LINK. Add a corpus-replay test (skip gracefully if `soundfile`/the
     WAV files aren't available in a minimal env — follow whatever skip pattern
     `tests/health/` already uses for I/O-dependent tests, check
     `test_calibration.py` for precedent) asserting: fp/F mostly SENSOR_LINK,
     `test_data/F` TN mostly NOT SENSOR_LINK.
   - Full suite must still pass (currently 239 tests): `.venv/bin/python -m
     pytest tests/ -q`.

4. **Verify.** Re-run the exact `rootcause.assess_many` replay from this
   conversation (fp/F, `test_data/F`, `test_data/T`, no calibration profile
   loaded) and confirm TN no longer saturates at SENSOR_LINK(1.00) while fp/F
   still mostly does.

## Phase B — wiring check + transparency log line

**Goal:** confirm the fix reaches the actual tester-facing UI paths with zero
UI code changes needed (the label/panel/popup already read `rootcause.assess`/
`assess_many` — Phase A's fix should be invisible plumbing to them), and add one
log line so the active threshold config is auditable (matches the existing
`decision_threshold.json` auto-load transparency pattern in `main.py`).

1. In `main.py`, wherever the health pipeline is (re)built
   (`__init__`/`_rebuild_health_pipeline`, main.py:57, 703-704), log which
   check-threshold config path was loaded (or "shipped defaults" if the file's
   absent) — one line, same style as the existing `self.log(f"Health profile:
   ...")` at main.py:761.
2. Headless verify only (per CLAUDE.md — GUI manual check is the owner's job):
   `.venv/bin/python -c "import ast; ast.parse(open('main.py').read())"` then
   `.venv/bin/python -c "import main"`, plus unchanged/increased test count.

## Unrelated: refresh-devices button (implemented directly, no sub-phase)

Add a small `ttk.Button` next to the device combo in the Run bar (`main.py`
~line 218-220, alongside the existing `file_btn`) that calls
`self.refresh_devices()` — currently that method only runs on init and on
Mic/Wav-File radio toggle, with no way to re-scan after e.g. plugging in a USB
audio interface mid-session.
