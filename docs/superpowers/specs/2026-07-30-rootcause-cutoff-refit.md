# Rootcause session cutoff refit (2026-07-30)

## Context

`main.py:handle_audio_chunk` (and its offline replication, `offline_score.py`'s
`score_wav_file`) now hold inference and health/rootcause analysis until the
persistent 2.5s rolling buffer has filled once with real audio, instead of
running on every 0.5s chunk from the first hop. This removes the artifact
documented in `app/health/rootcause.py`'s previous `DEFAULT_SESSION_CUTOFF`
comment: every session's first ~4 windows used to be built from a
mostly/partly zero buffer, which spuriously tripped ClickTransientCheck /
DropoutSegmentCheck and inflated every session's mean rootcause score
(clean or faulty alike) by a roughly constant floor.

`DEFAULT_SESSION_CUTOFF` (0.55, in `app/health/rootcause_session_config.json`)
was fit *against* sessions that included that inflated floor. With the
artifact gone, session mean-scores are structurally lower, so the cutoff
needed refitting rather than just carrying the old value forward.

## Method

Replayed the same two local corpora used for the 2026-07-16 recalibration
through the new hold-off windowing (`tests/health/test_rootcause.py`'s
`_session_audio_windows`, updated to match `main.py`) and computed each
session's mean per-window rootcause score (`app/health/rootcause._score`,
summed across executed-check windows and divided by window count — the same
arithmetic `assess_many` uses):

- `test_data/audio_signal_health/fp/F` — 8 confirmed sensor-link fault recordings
- `test_data/F` — 4 clean (TN) reference recordings

Each session now yields 36 windows instead of 40 (the leading 4 of 40 hops
are skipped while the buffer is still filling).

```
fault scores: 0.0556, 0.0556, 0.2222, 0.3333, 0.4444, 0.5000, 0.6389, 1.0556
clean scores: 0.0556, 0.0556, 0.1667, 0.2222
```

## Finding

The zero-padding artifact is gone, but the underlying per-check signal is
still noisy at n=12: two confirmed-fault recordings tie with two clean
recordings at the same minimum score (0.0556). No cutoff cleanly separates
the two groups.

A cutoff sweep found the best achievable total accuracy is 9/12, achieved
either just above 0.2222 (6/8 fault, 3/4 clean) or just above 0.3333 minus
margin, i.e. in the (0.2222, 0.3333) gap (5/8 fault, 4/4 clean).

## Decision

Picked **0.25**, in the (0.2222, 0.3333) gap: 5/8 fault recordings correctly
resolve SENSOR_LINK, 4/4 clean recordings do not. This is deliberately biased
toward not reintroducing false positives on clean recordings — the original
2026-07-16 recalibration exists because the prior rule fired SENSOR_LINK on
*every* session, so a refit that trades a bit of fault recall to keep clean
recordings clean is the right direction, not a regression. 5/8 is comfortably
clear of the "mostly" test bar (`>= (8+1)//2 = 4`).

Informal, not statistically rigorous — same caveat that applied to the 0.55
value this replaces. If a larger/labeled corpus becomes available, or a
finer-grained calibration profile is added, this is worth refitting properly
rather than by inspection of a 12-file sweep.

## Changes

- `app/health/rootcause.py`: `DEFAULT_SESSION_CUTOFF` 0.55 → 0.25, comment rewritten.
- `app/health/rootcause_session_config.json`: `cutoff` 0.55 → 0.25.
- `tests/health/test_rootcause.py`: `_session_audio_windows` updated to skip
  windows until the buffer fills (matching the new `main.py`/`offline_score.py`
  behavior); stale comment referencing the old 0.55 cutoff/rationale fixed.

Full test suite (`pytest tests/ -q`): 263 passed, 3 skipped, including both
corpus replay tests (`test_corpus_replay_fp_f_mostly_resolves_sensor_link`,
`test_corpus_replay_tn_f_mostly_not_sensor_link`).
