"""The WAV and live-mic input paths must be equivalent.

Both loops only produce 0.5s chunks onto `audio_queue`; everything downstream
(handle_audio_chunk -> health pipeline + run_inference + EWMA decision) is shared, so
equivalence is entirely a question of *what chunks each loop emits*. These tests pin
that contract: same count (capped by the same duration rule), same shape, same dtype,
same channel, and no synthetic zero padding on either side.

The count cap matters most: EwmaPeakDecision.peak is a running max over the session, so
a longer session is strictly more likely to cross the cutoff. A WAV replay that ran to
the end of a 5-minute file was not comparable to a 20s live test of the same signal.
"""
import queue

import numpy as np
import pytest

import main
from main import ModelsTesterApp, capture_duration_sec

SR = 44100
BLOCK = SR // 2  # 0.5s


class _FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _FakeApp:
    def __init__(self, inference_mode="sliding", sliding_test_duration_sec=20.0):
        self.is_running = True
        self.inference_mode_var = _FakeVar(inference_mode)
        self.single_shot_duration_sec = 20
        self.sliding_test_duration_var = _FakeVar(sliding_test_duration_sec)
        self.audio_queue = queue.Queue()
        self.logs = []

    def log(self, msg):
        self.logs.append(msg)


def _drain(q):
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


# --- shared duration rule -------------------------------------------------------


def test_capture_duration_sec_uses_single_shot_value_in_single_mode():
    assert capture_duration_sec("single", 20, 45.0) == 20


def test_capture_duration_sec_uses_sliding_value_in_sliding_mode():
    assert capture_duration_sec("sliding", 20, 45.0) == 45.0


# --- file_loop ------------------------------------------------------------------


@pytest.fixture
def fake_soundfile(monkeypatch):
    """Patches sf.read / time.sleep so file_loop runs synchronously off in-memory audio."""
    state = {}

    def _install(data, fs=SR):
        state["data"] = np.asarray(data, dtype=np.float64)
        state["fs"] = fs

    def _read(path, always_2d=False):
        data = state["data"]
        if always_2d and data.ndim == 1:
            data = data.reshape(-1, 1)
        return data, state["fs"]

    monkeypatch.setattr(main.sf, "read", _read)
    monkeypatch.setattr(main.time, "sleep", lambda _s: None)
    return _install


def test_file_loop_sliding_mode_caps_at_the_sliding_test_duration(fake_soundfile):
    # 60s of audio, 20s sliding cap -> 40 chunks, same as a 20s mic run. Before this
    # cap existed the whole file was replayed (120 chunks), inflating the EWMA peak.
    fake_soundfile(np.ones(SR * 60, dtype=np.float64))
    app = _FakeApp(inference_mode="sliding", sliding_test_duration_sec=20.0)

    ModelsTesterApp.file_loop(app, "any.wav")

    assert app.audio_queue.qsize() == 40


def test_file_loop_single_shot_mode_caps_at_the_single_shot_duration(fake_soundfile):
    fake_soundfile(np.ones(SR * 60, dtype=np.float64))
    app = _FakeApp(inference_mode="single", sliding_test_duration_sec=5.0)

    ModelsTesterApp.file_loop(app, "any.wav")

    assert app.audio_queue.qsize() == 40  # 20s / 0.5s


def test_file_loop_drops_a_trailing_partial_chunk_instead_of_zero_padding(fake_soundfile):
    # A zero-padded tail rolls into the session buffer and trips the click/dropout
    # checks (see rootcause.py's DEFAULT_SESSION_CUTOFF comment). The mic never emits
    # a partial block, so neither may the file path.
    fake_soundfile(np.ones(BLOCK * 3 + 5000, dtype=np.float64))
    app = _FakeApp(sliding_test_duration_sec=20.0)

    ModelsTesterApp.file_loop(app, "any.wav")

    chunks = _drain(app.audio_queue)
    assert len(chunks) == 3
    assert all(len(c) == BLOCK for c in chunks)
    assert all(np.all(c == 1.0) for c in chunks)  # no zero padding anywhere


def test_file_loop_emits_the_same_chunk_shape_and_dtype_as_the_mic_callback(fake_soundfile):
    # sounddevice hands the queue float32 (frames, channels); the file path must match
    # so handle_audio_chunk sees identical data either way.
    fake_soundfile(np.ones(SR * 2, dtype=np.float64))
    app = _FakeApp(sliding_test_duration_sec=20.0)

    ModelsTesterApp.file_loop(app, "any.wav")

    chunks = _drain(app.audio_queue)
    assert all(c.shape == (BLOCK, 1) for c in chunks)
    assert all(c.dtype == np.float32 for c in chunks)


def test_file_loop_takes_the_first_channel_like_a_mono_mic_stream(fake_soundfile):
    # mic_loop opens InputStream(channels=1), which takes channel 0 -- not a downmix.
    stereo = np.stack(
        [np.full(SR * 2, 0.1), np.full(SR * 2, 0.9)], axis=1
    )
    fake_soundfile(stereo)
    app = _FakeApp(sliding_test_duration_sec=20.0)

    ModelsTesterApp.file_loop(app, "any.wav")

    chunks = _drain(app.audio_queue)
    assert chunks
    assert all(np.allclose(c, 0.1) for c in chunks)  # 0.5 would mean a mean-downmix


def test_file_loop_stops_early_when_the_file_is_shorter_than_the_cap(fake_soundfile):
    fake_soundfile(np.ones(SR * 2, dtype=np.float64))
    app = _FakeApp(sliding_test_duration_sec=20.0)

    ModelsTesterApp.file_loop(app, "any.wav")

    assert app.audio_queue.qsize() == 4
    assert app.is_running is False


def test_file_loop_logs_its_capture_duration_like_the_mic_loop(fake_soundfile):
    fake_soundfile(np.ones(SR * 60, dtype=np.float64))
    app = _FakeApp(inference_mode="sliding", sliding_test_duration_sec=20.0)

    ModelsTesterApp.file_loop(app, "any.wav")

    assert any("Sliding-window mode: capturing 20.0s" in msg for msg in app.logs)


# --- mic_loop -------------------------------------------------------------------


class _FakeInputStream:
    """Delivers `chunks_to_deliver` blocks synchronously on __enter__."""

    def __init__(self, device, channels, samplerate, blocksize, callback):
        self._callback = callback

    def __enter__(self):
        for _ in range(_FakeInputStream.chunks_to_deliver):
            self._callback(np.zeros((BLOCK, 1), dtype=np.float32), BLOCK, None, None)
        return self

    def __exit__(self, *exc_info):
        return False


def test_mic_loop_enqueues_exactly_the_capped_chunk_count(monkeypatch):
    # The outer poll loop only checks every 0.1s, so callbacks could overshoot the cap
    # and queue extra windows the file path would never produce. The callback itself
    # now enforces the cap.
    monkeypatch.setattr(main.sd, "InputStream", _FakeInputStream)
    _FakeInputStream.chunks_to_deliver = 45
    app = _FakeApp(inference_mode="sliding", sliding_test_duration_sec=20.0)

    ModelsTesterApp.mic_loop(app, device_idx=0)

    assert app.audio_queue.qsize() == 40
    assert app.is_running is False
