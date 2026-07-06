import numpy as np

from app.health.defaults import default_manager, default_pipeline
from app.health.models import AudioWindow, HealthState

SR = 44100
N = 110250


def _win(x):
    return AudioWindow(samples=np.asarray(x, dtype=np.float32), sample_rate=SR)


def test_default_manager_registers_all_checks():
    assert len(default_manager().checks) == 11  # 7 time-domain + 4 frequency-domain


def test_default_pipeline_ok_on_clean_sine():
    t = np.arange(N) / SR
    sig = 0.3 * np.sin(2 * np.pi * 1000.0 * t)
    report = default_pipeline().analyze(_win(sig))
    assert report.final_state is HealthState.OK
    assert len(report.check_results) == 11


def test_default_pipeline_fault_on_silence():
    report = default_pipeline().analyze(_win(np.zeros(N)))
    assert report.final_state is HealthState.FAULT
    assert report.diagnostic_summary
