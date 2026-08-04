"""offline_score.py must emit the same window stream as main.py's live loops.

A decision cutoff is fitted against this script's output and then used unmodified by the
live app, so any divergence between the two silently miscalibrates the cutoff. See
main.py:mic_loop / main.py:file_loop and tests/test_input_path_parity.py for the other
two loops in this contract.
"""
import numpy as np

import offline_score
from offline_score import HOP_SEC, TARGET_SR, WINDOW_SEC

BLOCK = int(TARGET_SR * HOP_SEC)
BUFFER = int(TARGET_SR * WINDOW_SEC)
HOLD_OFF_HOPS = BUFFER // BLOCK  # 5 blocks fill the buffer; the 5th is the first scored


def _expected_windows(n_hops):
    """Scoring starts on the hop that completes the buffer, so hop indices
    HOLD_OFF_HOPS-1 .. n_hops-1 are scored. Matches main.py:handle_audio_chunk, where a
    20s/40-chunk session scores 36 windows."""
    return max(0, n_hops - HOLD_OFF_HOPS + 1)


class _FakeExtractor:
    """Records every buffer it is asked to featurise, so tests can assert on windows."""

    def __init__(self):
        self.buffers = []

    def extract_features(self, buffer, sr, n_mels, seq_len, **prep):
        self.buffers.append(np.array(buffer, copy=True))
        return np.zeros((seq_len, n_mels), dtype=np.float32)


class _FakeScaler:
    def apply(self, specs):
        return specs


class _FakeModel:
    def predict(self, input_data):
        return np.array([[0.5]], dtype=np.float32)


def _score(audio, monkeypatch, max_duration_sec=20.0):
    monkeypatch.setattr(offline_score, "_load_wav_mono", lambda _p: np.asarray(audio, dtype=np.float32))
    extractor = _FakeExtractor()
    scores = offline_score.score_wav_file(
        "any.wav", _FakeModel(), _FakeScaler(), extractor, 98, 32, {},
        max_duration_sec=max_duration_sec,
    )
    return scores, extractor


def test_twenty_second_file_yields_the_same_window_count_as_a_live_session(monkeypatch):
    # 20s / 0.5s = 40 hops, minus the 5 consumed by the buffer-fill hold-off = 36.
    scores, _ = _score(np.ones(TARGET_SR * 20), monkeypatch)
    assert len(scores) == _expected_windows(40)


def test_long_file_is_capped_to_the_live_capture_duration(monkeypatch):
    # Without a cap a 60s file scored 120 hops. EwmaPeakDecision.peak is a running max,
    # so that gave a fitted cutoff far more windows than any live session produces.
    scores, _ = _score(np.ones(TARGET_SR * 60), monkeypatch)
    assert len(scores) == _expected_windows(40)


def test_max_duration_is_configurable_to_match_a_changed_live_setting(monkeypatch):
    scores, _ = _score(np.ones(TARGET_SR * 60), monkeypatch, max_duration_sec=10.0)
    assert len(scores) == _expected_windows(20)


def test_trailing_partial_block_is_dropped_not_zero_padded(monkeypatch):
    # main.py:file_loop emits whole blocks only; a zero-padded tail would put a hard
    # zero edge into the final windows that no live session ever sees.
    audio = np.ones(BLOCK * 8 + 5000)
    _, extractor = _score(audio, monkeypatch)

    assert len(extractor.buffers) == _expected_windows(8)
    assert all(np.all(b == 1.0) for b in extractor.buffers)


def test_scored_windows_are_full_length_and_hold_only_real_audio(monkeypatch):
    _, extractor = _score(np.ones(TARGET_SR * 20), monkeypatch)

    assert all(len(b) == BUFFER for b in extractor.buffers)
    assert all(np.all(b == 1.0) for b in extractor.buffers)  # no zero-buffer remnants


def test_file_shorter_than_the_rolling_buffer_scores_nothing(monkeypatch):
    scores, _ = _score(np.ones(int(TARGET_SR * 1.5)), monkeypatch)
    assert scores == []


def test_load_wav_mono_takes_channel_zero_like_the_mic(monkeypatch):
    # mic_loop opens InputStream(channels=1) -> channel 0, not a mean downmix.
    stereo = np.stack([np.full(1000, 0.1), np.full(1000, 0.9)], axis=1)
    monkeypatch.setattr(offline_score.sf, "read", lambda _p, always_2d=False: (stereo, TARGET_SR))

    mono = offline_score._load_wav_mono("any.wav")

    assert np.allclose(mono, 0.1)


def test_cache_key_changes_when_scoring_semantics_change(tmp_path):
    # The cache survives across runs; without this, a stale entry scored under the old
    # zero-padded/uncapped rules would be silently reused by a refit.
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x" * 16)
    key = lambda **kw: offline_score._cache_key(str(wav), "m.tflite", "s.json", {}, **kw)

    assert key(max_duration_sec=20.0) != key(max_duration_sec=10.0)
