# Noisy test handling — implementation plan

Spec: `docs/superpowers/specs/2026-10-28-noisy-test-handling-design.md` (decisions D1–D5 settled 2026-10-28).
Branch: `features/noisy_test_handling` (from `main` at `b03efc4`).
Python: `../.venv/bin/python`. Tests: `../.venv/bin/python -m pytest tests/ -q`.

## RESUME HERE
Update this block at the end of each task: what is done, the commit, the test count, and the next task.

- Task 0 (spec + plan) done: `1fb6a9d`, `2f604a3`.
- **Task 1 done:** `app/decision/noisy_test.py` and `tests/decision/test_noisy_test.py` (22 tests); full suite 373 passed, 2 skipped.
- **Task 3 run early, and it FAILED bar 1. Stopped for the owner**, per this plan. See the spec section "Fit result, 2026-10-28". In short, a trimmed-floor level detector cannot tell car noise from the normal 11 dB variation of field recordings. With an oracle detector, extend 40 s halves the misses (21.8% → 10.5%).
- Tasks 2 and 4–6 are **not started**. The owner chooses the next step:
  - (a) a learned spectral car detector, fitted by injection (needs more noise recordings to avoid fitting one road);
  - (b) ship the mechanism with the detector off, i.e. the plumbing for the app, the detector later;
  - (c) another detection idea.

## Task 1 — Portable core `app/decision/noisy_test.py` (TDD)

Stdlib + NumPy only. No UI, no file I/O, no timing.

- `hop_floor_db(hop, trim_percentile=90.0) -> float`: the trimmed floor, ported from `features/noise_gated_windows:app/decision/noise_gate.py:hop_features`. Silence returns `-inf`.
- `NoisyTestConfig` (dataclass, `to_json` / `from_json`, `default_config(path)` with the load-JSON-with-fallback idiom of `threshold.py`). Fields:
  - `enabled=False`
  - `trim_percentile=90.0`
  - `reference="session_quantile"`, `reference_quantile=0.2`, `reference_min_hops=5`
  - `noise_db=inf` (unfitted means nothing is ever noisy)
  - `min_quiet_windows=36`, `hops_per_window=5`, `normal_hops=40`
  - `max_extra_sec=0.0`, `max_attempts=1`, `repeat_pause_sec=3.0`
  - `cutoff`, `span` (copied from the model's `decision_threshold.json` by the caller)
- `NoisyTestSession(config)`: `feed_hop(hop_samples, window_score)` → `Action`.
  - `window_score` is `None` until the first full buffer.
  - A window is noisy if any of its 5 hops has a floor more than `noise_db` above the reference.
  - The reference is the `reference_quantile` of the floors so far in this attempt. Before `reference_min_hops` hops, use what exists.
  - Quiet window scores feed an EWMA-peak accumulator (`EwmaPeakDecision` from `threshold.py`). All window scores feed a second accumulator: the provisional verdict for D5.
  - Actions: `CONTINUE`; `EXTENDING` (emitted once, when the normal hops end short of quiet windows and extension is allowed); `REPEAT(attempt_result)`; `DONE(result)`.
  - `result`: verdict (`HEALTHY` / `INFESTED` / `TOO_NOISY`), provisional verdict, quiet peak, all-window peak, n_quiet, n_windows, attempt, quiet_mask, floors.
  - `start_attempt()` resets per-attempt state; the caller calls it after the pause.
  - **With `enabled=False`:** DONE after exactly `normal_hops`, the verdict equals the all-window verdict, and the attempt count is 1. A test proves this.
- Tests in `tests/decision/test_noisy_test.py`, on synthetic hops (a quiet floor plus loud bursts, clicks for click-neutrality):
  - the off config equals today's EWMA verdict on random scores;
  - all quiet → DONE at hop 40 with the same verdict;
  - a burst → EXTENDING → DONE once 36 quiet windows exist;
  - extension exhausted → REPEAT → TOO_NOISY after `max_attempts`;
  - warn-only (0 extra, 1 attempt) → TOO_NOISY at hop 40;
  - sparse clicks don't make a window noisy;
  - the provisional verdict uses all windows;
  - JSON round trip.
- Commit.

## Task 2 — Golden vectors for the Dart port

- `scripts/make_noisy_test_vectors.py` generates `tests/data/noisy_test_vectors.json`. Each vector holds per-hop floors (as floats, so the port can skip the audio), scores, config, the expected action per hop, and the final result. There are about 12 vectors covering every config row in the spec table, plus edge cases (noise in the first hops, silence).
- The session gets `feed_floor(floor_db, window_score)`, so vectors can drive it without audio. `feed_hop` = `hop_floor_db` + `feed_floor`.
- A test replays every vector and asserts exact actions and results.
- Commit.

## Task 3 — Fit and evidence (`scripts/noisy_test_injection.py`, outside `app/`)

- **Clean corpus:** 2026 files from `../training/one_shot_1/dataset.csv`, roles val + recent + test (~2,200, no exclusions). Compute per-hop floors (no model needed) and reuse 9_1_2 window scores, computing them where missing.
- **Injection tracks:**
  - **Infested:** join 4 consecutive clean T recordings of the same session (80 s, 50 ms crossfade); 9_1_2 never saw 2026.
  - **Healthy:** the same with F recordings.
  - Each track is mixed with 2 random 80 s stretches of the long car recording at its recorded level, plus the clean track itself. Cache the per-hop floors and per-window scores (BLAS pinned, 16 workers).
- **Sweep:** `noise_db` (e.g. 3–15 dB), `reference_quantile` (0.1 / 0.2 / 0.3) and configs (warn / extend 40 s / repeat 2). Results go into the spec's Evidence section, which also states that joined tracks are a stand-in.
  - **Bar 1:** trigger rate on clean 20 s tests ≤ 5%.
  - **Bar 2:** trigger rate on clean T minus clean F ≤ 2 pp.
  - **Bar 3:** infested tracks reported HEALTHY cut to less than half of today's extra misses; healthy tracks' false alarms ≤ today's clean rate.
- Choose `noise_db` and the reference; write `models/9_1_2/noisy_test.json` with `enabled: false`.
- **If no setting passes bars 1–3: stop and report to the owner.** Don't wire the UI on a failed fit.
- Commit.

## Task 4 — Tester wiring (`main.py`)

- Load the config with the model (as for `decision_threshold.json`); the settings panel exposes enabled, max_extra_sec, max_attempts and repeat_pause_sec (persisted with the other settings).
- **mic_loop / file_loop:** the capture cap becomes dynamic.
  - On EXTENDING, raise it to `normal + max_extra_sec`. File replay obeys the same cap; a file that ends early ends the attempt as if the extension ran out.
  - On REPEAT: save the attempt, show "Repeating test k of n", pause, then start a new capture. For files, the next attempt continues from where the file is, so a long WAV replays like a live road.
- **handle_audio_chunk:** pass each 0.5 s hop and the window score (from `run_inference`) to the session.
- **Display:** a "Noise detected – keeping the test running" line, the final verdict TOO NOISY in orange, and noisy windows shaded on the score plot.
- **Saving:**
  - the WAV holds the attempt's full audio (fixes the 2.5 s-buffer save bug);
  - the JSON gains `verdict`, `provisional_verdict`, `attempt`, `n_attempts`, `duration_sec_actual`, `quiet_mask`, `noise_floor_db` and the noisy-test config;
  - the filename gets `_NZ` after the outcome prefix for TOO NOISY (D5).
- **With `enabled=False`:** behaviour is byte-identical except the save fix. Check with `tests/test_offline_score_parity.py` and `tests/test_input_path_parity.py` unchanged and green.
- **Headless checks:** `ast.parse`, `import main`, full test count. The GUI check is the owner's.
- Commit.

## Task 5 — Offline replay (`offline_score.py --noisy-test CONFIG`)

- Runs the same session over each WAV with no duration cap and writes one row per attempt (verdict, provisional, n_quiet, duration).
- Run it on the 43-minute recording with the three configs, and report how often each one extends, repeats or gives up on that road.
- A test checks that replaying a short synthetic WAV through `--noisy-test` matches `NoisyTestSession` fed directly.
- Commit.

## Task 6 — Handover and records

- `docs/noisy-test/handover-mobile.md`: the state machine diagram (mermaid), config fields, the vectors and how to run them, the 36-vs-40 parity point (D3), and what to save.
- Update `CLAUDE.md` (status line, test count), this plan's RESUME block, and the memory note.
- Commit. Pushing and merging stay with the owner.
