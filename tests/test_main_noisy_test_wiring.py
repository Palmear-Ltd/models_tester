"""End-to-end wiring of the noisy-test feature in main.py, headless: a real ModelsTesterApp
(Tk root withdrawn, 9_1_2 model), capture threads stubbed out, chunks fed straight into
handle_audio_chunk as the queue would. Checks the feature-off path is today's behaviour
(plus the full-length WAV save fix), and that the on path extends, repeats, names TOO NOISY
files with _NZ and agrees with NoisyTestSession fed directly."""
import json
import os

import numpy as np
import pytest

main = pytest.importorskip("main", reason="main.py needs the full GUI/tflite dependency stack")

from app.decision.noisy_test import TOO_NOISY, NoisyTestSession  # noqa: E402
from app.decision.threshold import EwmaPeakDecision  # noqa: E402

SR = 44100
HOP = SR // 2
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _NoThread:
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass

    def join(self, timeout=None):
        pass


@pytest.fixture
def app(tmp_path, monkeypatch):
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    monkeypatch.setattr(main, "USER_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(main.threading, "Thread", _NoThread)
    a = main.ModelsTesterApp(root)
    monkeypatch.setattr(a.root, "after", lambda *args, **kw: None)
    a.model_path_var.set(os.path.join(REPO, "models", "9_1_2", "model.tflite"))
    a.scaler_path_var.set(os.path.join(REPO, "models", "9_1_2", "scaler.json"))
    a.output_dir_var.set(str(tmp_path / "out"))
    a.save_results_var.set(True)
    a.input_type_var.set("mic")
    a.device_var.set("0: fake")
    a.inference_mode_var.set("sliding")
    a.duration_var.set(2.5)
    yield a
    root.destroy()


def audio(seconds, bursts=(), seed=0):
    """Quiet noise at about -55 dBFS with loud stretches [(start_s, end_s), ...] at -25."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 10 ** (-55 / 20), int(seconds * SR)).astype(np.float32)
    for a, b in bursts:
        x[int(a * SR):int(b * SR)] *= 10 ** (30 / 20)
    return x


def run(app, x):
    """Starts a test and feeds 0.5 s chunks until the test ends itself or audio runs out,
    then stops it as process_queue would. Returns the chunks fed."""
    assert app.load_resources()
    app.start_test()
    fed = []
    for h in range(len(x) // HOP):
        if not app.is_running:
            break
        chunk = x[h * HOP:(h + 1) * HOP]
        app.handle_audio_chunk(chunk.reshape(-1, 1))
        fed.append(chunk)
    app.is_running = False
    app.stop_test()
    return fed


def saved(app):
    out = app.output_dir_var.get()
    return sorted(f for f in os.listdir(out))


def test_feature_off_is_todays_verdict_and_saves_the_full_20s(app):
    app.noisy_enabled_var.set(False)
    run(app, audio(20, bursts=[(5, 8)]))
    assert app.noisy_session is None
    acc = EwmaPeakDecision(app.decision_config)
    for s in app.score_history:
        acc.update(float(s))
    assert len(app.score_history) == 36
    files = saved(app)
    assert len(files) == 2 and "_NZ" not in files[0]
    wav = [f for f in files if f.endswith(".wav")][0]
    import soundfile as sf
    assert len(sf.read(os.path.join(app.output_dir_var.get(), wav))[0]) == 20 * SR
    data = json.load(open(os.path.join(app.output_dir_var.get(), wav[:-4] + ".json")))
    assert data["verdict"] == acc.state and data["predicted_infested"] == (acc.state == "INFESTED")
    assert data["noisy_test"] is None and data["duration_sec_actual"] == 20.0


def test_feature_on_extends_and_matches_the_session_fed_directly(app):
    app.noisy_enabled_var.set(True)
    app.noisy_max_extra_var.set(40.0)
    app.noisy_max_attempts_var.set(1)
    fed = run(app, audio(60, bursts=[(5, 8)]))
    r = app.noisy_attempts[-1]["result"]
    assert len(fed) > 40 and r.n_hops == len(fed) and r.n_quiet == 36
    # Same hops and the app's own window scores through a fresh session: same answer.
    s = NoisyTestSession(app.noisy_config)
    for h, chunk in enumerate(fed):
        a = s.feed_hop(chunk, float(app.score_history[h - 4]) if h >= 4 else None)
    assert a.kind == "DONE" and a.result.verdict == r.verdict and a.result.quiet_mask == r.quiet_mask
    files = saved(app)
    data = json.load(open(os.path.join(app.output_dir_var.get(), [f for f in files if f.endswith(".json")][0])))
    assert data["verdict"] == r.verdict and data["noisy_test"]["n_quiet"] == 36
    assert data["duration_sec_actual"] == len(fed) * 0.5
    assert data["noisy_test"]["config"]["noise_db"] == 10.0


def test_feature_on_without_extension_names_too_noisy_files_nz(app):
    app.noisy_enabled_var.set(True)
    app.noisy_max_extra_var.set(0.0)
    app.noisy_max_attempts_var.set(1)
    app.user_label_var.set("healthy")
    run(app, audio(20, bursts=[(5, 8)]))
    r = app.noisy_attempts[-1]["result"]
    assert r.verdict == TOO_NOISY
    files = saved(app)
    assert len(files) == 2 and all("_NZ_" in f for f in files)
    prefix = files[0].split("_")[0]
    infested = r.provisional_verdict == "INFESTED"
    assert prefix == ("FP" if infested else "TN")       # outcome letters from the provisional verdict (D5)
    data = json.load(open(os.path.join(app.output_dir_var.get(), [f for f in files if f.endswith(".json")][0])))
    assert data["verdict"] == TOO_NOISY and data["predicted_infested"] == infested
    assert app.diag_label.cget("text").startswith("TOO NOISY")


def test_feature_on_repeat_saves_each_attempt(app):
    app.noisy_enabled_var.set(True)
    app.noisy_max_extra_var.set(0.0)
    app.noisy_max_attempts_var.set(2)
    app.noisy_pause_var.set(1.0)
    bursts = [(t, t + 0.5) for t in np.arange(0, 60, 1.0)]   # noisy throughout
    run(app, audio(60, bursts=bursts))
    assert [a["result"].attempt for a in app.noisy_attempts] == [1, 2]
    assert all(a["saved"] for a in app.noisy_attempts)
    files = saved(app)
    assert len([f for f in files if f.endswith(".json")]) == 2
    assert all("_NZ_" in f for f in files)
