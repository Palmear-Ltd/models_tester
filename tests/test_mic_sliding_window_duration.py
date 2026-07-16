"""Sliding-window mic tests now auto-stop after a configurable duration (default 20s),
the same way single-shot mode already did, so a tester isn't stuck manually clicking
STOP TEST every run. See main.py:mic_capture_duration_sec / main.py:mic_loop."""
import queue

import numpy as np
import pytest

import main
from main import ModelsTesterApp, mic_capture_duration_sec


def test_mic_capture_duration_sec_uses_single_shot_value_in_single_mode():
    assert mic_capture_duration_sec("single", 20, 45.0) == 20


def test_mic_capture_duration_sec_uses_sliding_value_in_sliding_mode():
    assert mic_capture_duration_sec("sliding", 20, 45.0) == 45.0


def test_mic_capture_duration_sec_defaults_to_20_for_sliding():
    # main.py's ModelsTesterApp seeds sliding_test_duration_var at 20.0 (main.py:__init__).
    assert mic_capture_duration_sec("sliding", 20, 20.0) == 20.0


class _FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _FakeInputStream:
    """Stand-in for sounddevice.InputStream. Delivers exactly `chunks_to_deliver` audio
    blocks synchronously on __enter__ (matching the duration cap under test), so the
    outer polling loop sees chunk_count reach max_chunks on its very first check --
    no real-time sleeps needed in the test."""

    instances = []

    def __init__(self, device, channels, samplerate, blocksize, callback):
        self._callback = callback
        _FakeInputStream.instances.append(self)

    def __enter__(self):
        for _ in range(_FakeInputStream.chunks_to_deliver):
            self._callback(np.zeros(1, dtype=np.float32), 1, None, None)
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeApp:
    def __init__(self, inference_mode, sliding_test_duration_sec):
        self.is_running = True
        self.inference_mode_var = _FakeVar(inference_mode)
        self.single_shot_duration_sec = 20
        self.sliding_test_duration_var = _FakeVar(sliding_test_duration_sec)
        self.audio_queue = queue.Queue()
        self.logs = []

    def log(self, msg):
        self.logs.append(msg)


@pytest.fixture(autouse=True)
def _patch_input_stream(monkeypatch):
    _FakeInputStream.instances = []
    monkeypatch.setattr(main.sd, "InputStream", _FakeInputStream)
    yield


def test_sliding_mode_mic_loop_auto_stops_at_configured_duration():
    # 1s sliding cap / 0.5s blocks => 2 chunks. Previously sliding mode had no cap at
    # all (max_chunks stayed None), so mic_loop would just poll forever; now it must
    # flip is_running off on its own once chunk_count reaches the cap.
    _FakeInputStream.chunks_to_deliver = 2
    app = _FakeApp(inference_mode="sliding", sliding_test_duration_sec=1.0)
    ModelsTesterApp.mic_loop(app, device_idx=0)

    assert app.is_running is False
    assert app.audio_queue.qsize() == 2


def test_single_shot_mode_mic_loop_still_uses_its_own_fixed_duration():
    # Unaffected by the new sliding_test_duration_var: single-shot's cap still comes
    # from single_shot_duration_sec (20s / 0.5s blocks = 40 chunks).
    _FakeInputStream.chunks_to_deliver = 40
    app = _FakeApp(inference_mode="single", sliding_test_duration_sec=1.0)
    ModelsTesterApp.mic_loop(app, device_idx=0)

    assert app.is_running is False
    assert app.audio_queue.qsize() == 40


def test_sliding_mode_mic_loop_logs_capture_duration():
    _FakeInputStream.chunks_to_deliver = 40
    app = _FakeApp(inference_mode="sliding", sliding_test_duration_sec=20.0)
    ModelsTesterApp.mic_loop(app, device_idx=0)

    assert any("Sliding-window mode: capturing 20.0s" in msg for msg in app.logs)
