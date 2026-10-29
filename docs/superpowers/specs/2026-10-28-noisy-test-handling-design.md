# Noisy test handling: warn, extend, repeat (design)

**Status:** decisions D1–D5 settled by the owner on 2026-10-28 (see "Decisions"). Ready for the plan. The feature ships switched off.
**Replaces in part:** `2026-09-24-noise-gated-windows-design.md` (branch `features/noise_gated_windows`, not merged). That design assumed noise causes **false alarms** and tried to remove noisy windows from the verdict. The first real car recording shows the opposite for cars (see Evidence). This spec reuses that branch's noise measurement (the trimmed floor), but changes what the tester does with it.
**Who it is for:** the tester is the reference implementation. The mobile app developer replicates it in Flutter from this spec, the portable module and its test vectors.

## The problem in one paragraph

Near a road, a passing car makes 9_1_2 score **lower**, not higher. An infested tree tested while cars pass can come out HEALTHY. A healthy tree comes out healthy either way. So car noise makes a **HEALTHY** answer untrustworthy. Other noises (wind, pumps, engines, or several at once) may instead push scores up and fake an **INFESTED** answer, so the owner decided (D2) that **a noisy test is not trusted either way**. Operators don't reliably follow "wait for a quiet moment", so the tester must handle it: notice the noise, then (by configuration) extend the test, repeat it, or tell the operator the result can't be trusted.

## Evidence (2026-10-28, all measured on this machine)

**Source:** `palmear/long_recording_for_cars/2026-10-21 10.29.32.wav`.
- 43.5 min, recorded at the Musa farm roadside on a healthy tree, with our sensor and a different recording app (ours stops at 20 s).
- A car passes every 20–30 s: the level rises from about −60 to −35 dBFS, then falls back.
- Scripts and results are in the session scratchpad (`cars/`). They move into `scripts/` when this is built.

| Check | Result |
|---|---|
| 9_1_2 on the recording itself, as 130 separate 20 s tests | 0 alarms (app rule and tester rule). Louder windows score **lower** (correlation −0.29). |
| Real car stretches mixed into 126 infested val recordings (3 each) | App rule (22 of 36 windows ≥ 0.45): **71% → 48%** caught. Tester rule (EWMA peak): **89% → 62%**. |
| The same mixing into 113 healthy val recordings | False alarms fall (app 13% → 4%). Noise does not create alarms. |
| Score change per window vs car level | −0.08 when the car is faint (< −55 dBFS), −0.35 when loud (> −40 dBFS) |
| Tests with no car close (loudest moment < −50 dBFS) | 61% caught (app rule), against 40% when a car passes close. Some masking remains from road rumble. |
| Dropping loud windows only, without real extra time (the old design) | 48% → 55% caught, with 20% of tests left without enough quiet windows |
| How long until a car-free 20 s comes on that road (no car above −45 dBFS) | Median 32 s, 90th percentile 130 s. Only 12% of 20 s stretches qualify. |

**Scope:** one road, one tree, cars only. Pumps, engines and wind are untested. The owner reported in September that some noise causes false alarms, and wind looks like an infested tree in its spectrum. That is why the rule for INFESTED answers is a setting (D2), not hard-coded.

## What the tester does (behaviour)

### Today
Record 20 s, score 36 windows, give HEALTHY or INFESTED.

### With this feature (when enabled)
For every 0.5 s hop the tester measures **noise** alongside the model score. Each window is marked **quiet** or **noisy** (see "The noise measure"). **Noisy windows never count, in either direction.** The verdict is always the EWMA-peak rule over **exactly 36 quiet windows** (D1), taken in the order they were recorded.

1. **Record and collect quiet windows.** With no noise, the 36 quiet windows are exactly today's 36 windows after 20 s, so the test, verdict and duration are identical to today.
2. **When 36 quiet windows have been collected,** stop and report HEALTHY or INFESTED from them.
3. **If the normal 20 s ends with fewer than 36 quiet windows:**
   - **Extend** (if `max_extra_sec` > 0): keep recording until 36 quiet windows exist, up to `max_extra_sec` more seconds.
   - **Repeat** (if `max_attempts` > 1): if there still aren't enough, start a fresh test automatically after a short pause (`repeat_pause_sec`), up to `max_attempts` tests in total. Each attempt starts its quiet-window count from zero, because the operator may have moved the sensor.
   - **Give up:** report **"TOO NOISY – result not reliable, test again later"**. This is never reported as HEALTHY or INFESTED.

`min_quiet_windows` is fixed at 36 in this version, matching the window count the cutoff was fitted on (D1). It appears in the config only so that a later refit can change it together with the cutoff.

The PM's three wishes are all settings of one mechanism:

| Configuration | `max_extra_sec` | `max_attempts` | Result when noise persists |
|---|---|---|---|
| Off (today's behaviour) | — | — | HEALTHY / INFESTED as today (`enabled = false`) |
| Warn only | 0 | 1 | TOO NOISY after 20 s |
| Extend | e.g. 40 | 1 | Up to 60 s, then TOO NOISY |
| Repeat | 0 | e.g. 3 | Up to 3 × 20 s, then TOO NOISY |
| Extend + repeat | e.g. 20 | e.g. 2 | Up to 2 × 40 s, then TOO NOISY |

**Why extend or repeat at all, if dropping windows helped little:** dropping loud windows inside a fixed 20 s leaves too few quiet windows (20% of tests). Extending gives real extra quiet time, which is what a careful operator would do by waiting for a car-free moment. The car-free tests in the evidence (61% vs 40%) are the estimate of what extending can recover. They won't recover the full 71%, because road rumble still masks a little.

**A note on the EWMA over quiet windows:** the smoothing runs over the quiet windows as one sequence, skipping the noisy ones. A quiet window just after a car is therefore smoothed with the last quiet window before it, not with the car.

### Messages to the operator (live)
- During a test: "Noise detected – keeping the test running" (extend) or "Noise detected – repeating test 2 of 3" (repeat).
- At the end: HEALTHY, INFESTED or TOO NOISY. The result also states how many quiet windows the verdict used and how long the test lasted.

## Decisions (owner, 2026-10-28)

### D1. The verdict uses the first 36 quiet windows
Noisy windows are left out and the verdict waits for 36 quiet ones. The EWMA cutoff (0.5697, span 5) was fitted on 36 windows, so its false-alarm rate stays as fitted. Scoring more windows would raise false alarms, because a peak over more windows crosses the cutoff more easily.

### D2. A noisy test is rechecked whatever the answer
INFESTED under noise is not trusted either. Wind and other noises, alone or together, may push scores up, even though cars push them down. So noisy windows count for nothing in either direction, and there is no early INFESTED exit from noisy windows.

### D3. The phone uses the EWMA peak rule
The mobile app already implements the EWMA peak rule, so the module implements only that rule. There is no count rule. **One parity point for the app developer to confirm:** the tester scores from the first full 2.5 s buffer (36 windows per 20 s), while the phone's FIFO historically started zero-filled (40 hops per 20 s). Under this feature the quiet-window count starts at the first full buffer on both, and the test vectors pin that down.

### D4. Off by default
`enabled = false` in the shipped config. It is a structural no-op, proven by a test, and it stays off until the owner turns it on or approves it. The tester exposes all settings so the PM can try configurations. Suggested starting values once fitted: `max_extra_sec = 40`, `max_attempts = 2`, `repeat_pause_sec = 3`.

### D5. TOO NOISY in results and filenames
The TP/TN/FP/FN prefix is only the field engineers' data-collection labelling: it combines the operator's known label with the tester's verdict. It stays as it is, and noise is added as **extra information**:
- **JSON:** a new `verdict` field: `"HEALTHY" | "INFESTED" | "TOO_NOISY"`. `predicted_infested` stays for compatibility. For TOO NOISY it holds the provisional verdict (below), and `verdict` says it was not reliable.
- **Filename:** for a TOO NOISY test, `_NZ` is added after the outcome prefix, e.g. `FN_NZ_20261028_101530.wav`. The outcome letters come from the **provisional verdict**: the EWMA-peak rule over all windows of that attempt, noisy ones included, exactly as today's tester would have said. The audio ingest matches `^(TP|FN|TN|FP|X)_` (`audio/catalog_lib.py:154`), so the label (T/F) is still read correctly with no ingest change. A later ingest change may record the `NZ` tag as a column; that is optional.
- **Each repeat attempt** is saved as its own file. Attempts that reached a verdict get no tag; attempts that ran out of quiet windows get `_NZ`.

## The noise measure

A per-hop number that rises with environmental noise but not with larva clicks. It reuses the **trimmed floor** from `app/decision/noise_gate.py` (branch `features/noise_gated_windows`): the hop's level after clipping each sample to the hop's own 90th percentile, so short bite clicks are removed and steady noise is kept.

A window (2.5 s = 5 hops) is **noisy** if its trimmed floor is more than `noise_db` above a **quiet reference**. Candidates for the reference, chosen by the fit:
1. **Session-relative:** a low quantile of the floors seen so far in this test (and the previous attempt). This catches a passing car against the road's quiet moments. It cannot see noise that never stops (a pump running throughout).
2. **Absolute:** a fixed dBFS floor level. This catches steady noise, but the corpus spans 45 dB of level across sessions and hardware, so a fixed number is a site lottery. It's allowed only in combination with 1, and only if the fit shows it is safe on the clean corpus.
3. **Low-frequency share** (energy below 300 Hz, about 0.39 at the roadside farm vs 0.16 on the quiet farm) as a second signal for traffic rumble.

**Explicitly excluded as noise inputs:** click counts, template matches, crest factor and untrimmed peak level. These are the larva signature: using them would mark infested windows as noisy.

### Fitting the threshold (no infested roadside trees needed)
Fitting is done by **injection**: real car noise from the long recording, mixed into real clean recordings, both infested and healthy, from the 2026 val split. Every hop's true noise level is then known.
- `noise_db` is set to the level where the window score starts to drop meaningfully, measured from the injected infested windows (e.g. a mean drop of 0.05).
- **Bar 1: few false triggers on clean recordings.** On the unmixed 2026 corpus, at most 5% of tests may trigger an extension. Otherwise operators wait constantly.
- **Bar 2: click-neutral.** The trigger rate on clean infested recordings must not exceed the rate on clean healthy ones by more than 2 percentage points. Otherwise heavy chewing would look like noise.
- **Bar 3: worth it.** On injected tests, the share of infested recordings caught (or flagged TOO NOISY instead of HEALTHY) must clearly beat today's rule. Target: fewer than half the extra misses. On injected healthy recordings, false alarms must not rise above today's clean rate.
  - **The number that matters for customers:** infested trees reported as HEALTHY. Today about 52% of infested roadside tests would be (100% − 48%). After the change, those cases should become INFESTED or TOO NOISY.
- **Extending can only be measured on long recordings.** For the fit, the 43-minute healthy recording supplies the continuing noise. The infested signal is taken from consecutive 20 s recordings of the same infested session joined end to end (a stand-in, stated in the results). A real check needs long recordings on infested trees, which is a field request.

## Code layout (what the app developer reads)

- **`app/decision/noisy_test.py`** (portable: stdlib and NumPy only, no UI, no file I/O):
  - `trimmed_floor_db(hop_samples)` and `lf_share(hop_samples, sr)`: the per-hop measurements.
  - `NoisyTestConfig`: the settings above, loaded from JSON.
  - `NoisyTestSession`: a state machine fed one hop at a time (`feed(hop_samples, window_score_or_None)`). It returns an action: `CONTINUE`, `EXTEND`, `REPEAT`, or `DONE(verdict, details)`. It holds the quiet-window list and applies the EWMA peak rule over it (D1, D3), and it also tracks the provisional all-window verdict for the filename (D5). It does no timing and no audio capture: the caller does that, which is exactly the split the phone needs.
- **`models/9_1_2/noisy_test.json`**: fitted thresholds and defaults (`enabled: false` until approved). It sits alongside the existing configs and replaces none of them.
- **`tests/decision/test_noisy_test.py`**: unit tests, plus **golden test vectors**, `tests/data/noisy_test_vectors.json`. Each vector gives per-hop floor values and scores, the config, and the expected action sequence and final verdict, covering every configuration in the table above. The app developer runs the same vectors against the Dart port; a port is correct when every vector matches.
- **`main.py`**: wiring only.
  - It feeds each hop into `NoisyTestSession`.
  - On EXTEND it raises the capture cap, both for the mic and for file replay; a file shorter than the extension ends as if the operator stopped.
  - On REPEAT it saves the attempt, pauses and starts a new attempt.
  - It shows the messages and draws noisy windows shaded on the score plot.
  - A settings panel holds the configuration.
- **Saving:** each attempt is saved as its own WAV (the **full** audio, including any extension) plus JSON. The JSON gains `verdict`, `attempt`, `n_attempts`, `duration_sec_actual`, `quiet_mask` (per window) and `noise_floor_db` (per hop), so a saved recording can always be re-analysed. This also fixes the known bug where the saved WAV holds only the last 2.5 s buffer, which is needed anyway once tests can run longer than 20 s.
- **`offline_score.py`** gains `--max-duration-sec` use for long files (it exists already) and a `--noisy-test` replay mode that runs the same state machine over a long WAV, so long field recordings can be scored offline exactly as the tester would.

## What does not change

- The model, features, windows and cutoffs when the feature is off. `enabled = false` is a guaranteed no-op, and a test proves it: identical verdicts on the test corpus with the feature off.
- The health subsystem (`app/health/`). The noise measure lives in `app/decision/` because it changes the verdict; health never does.
- The phone, until the owner approves and the app developer ports it.

## Order of work (each step leaves a working tester)

1. **Measure and fit:** the injection scripts, the noise measure, the threshold choice against bars 1–3, and a written result. Stop for the owner if a bar fails.
2. **Portable module and tests:** `noisy_test.py`, unit tests and golden vectors, with no UI change.
3. **Tester wiring:** extend and repeat in mic and file mode, messages, plot shading, settings, saving (with the save fix) and the TOO NOISY answer.
4. **Offline replay:** `offline_score.py --noisy-test`, plus a run over the 43-minute recording to show how the configurations behave on a real road (how often each one extends, repeats or gives up).
5. **Handover note for the app developer:** the state machine as a diagram, the config fields, the vectors, and the open D3 number.

## Field requests (to make this trustworthy beyond one road)

- More **long recordings (≥ 10 min) on healthy trees** near other roads, a water pump, an engine, and on windy days, with the same other app. These are cheap and needed for the fit's generality and for D2.
- **Any long recording on an infested tree** in any noise, even a short drive from the farm, would let extension be measured directly instead of by stand-in.

## Fit result, 2026-10-28: the level-based noise measure fails bar 1 (stopped for the owner)

`scripts/noisy_test_injection.py` (`build`, `eval`). Cache: `../training/9_1_5/.noisy_test_cache/`.

**Tracks:**
- 62 infested and 90 healthy joined tracks: 4 consecutive 2026 recordings of the same session, 80 s each, from val, recent and test.
- 51 infested and 9 healthy real long recordings from 2025-12 (60–120 s, possibly seen by 9_1_2).
- Each track was mixed with 2 random stretches of the car recording at its recorded level.

**1. Normal field recordings vary as much as a car does.** Within ordinary 2026 20 s recordings (456), the trimmed floor swings by a median of **11 dB** above the recording's 20th percentile (90th percentile: 21 dB), at every position in the 20 s. Mixed in at its recorded level, the car raises a typical recording's floor by only 0.4 dB (faint car) to 12 dB (loud car), yet scores still drop. Consequences:

| Measure | Normal tests flagged | Car-affected tests flagged |
|---|---|---|
| Floor > reference + 6 dB, any hop (q 0.2) | 82% | ~95% |
| Floor > reference + 15 dB, any hop | 33% | — (misses most masking) |
| Floor > reference + 10 dB for ≥ 12 consecutive hops | 5% | 5–9% |
| Absolute floor > −35 dBFS, any hop | 36% | — |

The low-frequency share (< 300 Hz) does not separate either: normal hops span 0.02–0.99, car hops 0.24–0.37. **No setting meets bar 1 (≤ 5% of normal tests extended) while catching car-affected tests.**

**2. The mechanism itself works, given a good detector.** These results use an *oracle* detector that knows the true car level (noisy = car above −45 dBFS) and the extend-40 s configuration. Joined 2026 tracks:

| | Infested reported HEALTHY | Infested INFESTED | Infested TOO NOISY | Healthy false alarms |
|---|---|---|---|---|
| Today, clean | 6.5% | 93.5% | — | 27.8% |
| Today, with cars | **21.8%** | 78.2% | — | 12.2% |
| Oracle + extend 40 s | **10.5%** | 83.1% | 6.5% | 18.3% |

So, with a good detector, misses are halved and false alarms stay below the clean rate (bar 3 met).

On the 2025 long recordings, whose infested signal is weaker (today: 78% caught clean, **9%** with cars), extending recovers almost nothing: even the road's quiet moments mask them. Only a strict TOO NOISY protects those trees.

**What this means:** the state machine (`app/decision/noisy_test.py`, 22 tests) is sound and ready. What is missing is a noise detector that tells car noise apart from the ordinary variation of field recordings. Level alone cannot do it. Options are for the owner (see the plan's RESUME block).

## Background level on real field recordings, 2026-10-29

Question from the owner: is there a background level above which 9_1_2 stops working, whatever the noise type? Data: every 2026 20 s recording with 9_1_2 window scores (5,757: 5,045 healthy, 712 infested, about 40 sessions), no mixing. Scripts: `scripts/noisy_test_background/` (the table lives in the session scratchpad and is rebuilt by `build.py`).

- **Infested trees are not missed in louder backgrounds.** Catch rate by median background: 74% below −60 dBFS (35 recordings), 89–92% from −60 to −45, 95–100% above −45. Within sessions, the loudest third vs the quietest third: 82.5% vs 87.4% (17 sessions, not consistent: 6 lower, 3 higher). Exception, with small n: tests whose background swings 20–30 dB caught 59% (17 recordings, 7 sessions).
- **Healthy trees get more false alarms in louder backgrounds.** By median background: 13% below −60 dBFS, 23% at −55 to −50, 34% at −45 to −40, 51–55% above −35. Within sessions, the loudest third vs the quietest third: 24.9% vs 16.0%, louder worse in 29 of 40 sessions. Within a recording, healthy windows more than 6 dB above the recording's own median score +0.06 higher (34 of 41 sessions).
- **Dropping the loud windows** (any hop more than X dB above the recording's 20th percentile), compared against dropping the same number of windows at random: X = 6 gives healthy positives 16.1% vs 21.3% and infested 87.0% vs 88.8%; X = 10 gives 21.1% vs 26.0% and 87.4% vs 89.5%. So the effect is real but modest: about 5 points fewer false alarms, for about 2 points fewer catches.
- The false-alarm rate in these sets (24% overall; 15–33% by role) is far above the field estimate of about 3–4%; this has not been explained.
- **Conclusion:** everyday background noise (not cars) causes **false alarms**, not misses. That supports D2: noisy windows should count for nothing in either direction. Cars, as mixed into recordings, cause misses. Both effects matter.
