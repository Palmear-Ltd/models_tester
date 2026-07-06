import numpy as np
import soundfile as sf

import calibrate
from app.health.calibration import load_profile

SR = 44100


def test_cli_run_generates_profile(tmp_path):
    n = int(6.0 * SR)
    sig = (0.3 * np.sin(2 * np.pi * 1000.0 * np.arange(n) / SR)).astype(np.float32)
    wav = tmp_path / "healthy.wav"
    sf.write(str(wav), sig, SR)
    out = tmp_path / "profile.json"

    calibrate.run(str(wav), str(out), profile_id="cli_test", sensor_info="piezo")

    profile = load_profile(str(out))
    assert profile.profile_id == "cli_test"
    assert profile.window_count == 8
    assert "T002" in profile.statistics
