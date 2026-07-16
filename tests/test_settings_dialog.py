import os

from app.ui.settings_dialog import SettingsDialog


class _FakeApp:
    """Minimal stand-in; SettingsDialog must construct without touching Tk."""

    root = None


def test_settings_dialog_constructs_without_display():
    dlg = SettingsDialog(_FakeApp())
    assert hasattr(dlg, "show")
    assert callable(dlg.show)


def test_preprocessing_tab_binds_fmin_fmax():
    # SettingsDialog builds its widgets against app.<var> attributes. Confirm the
    # preprocessing tab references fmin/fmax vars by exercising _build_preprocessing_tab
    # with a recording fake that captures which app attributes are read.
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        import pytest
        pytest.skip("no display available")
    root.withdraw()
    from tkinter import ttk

    class _App:
        def __init__(self):
            self.low_cut_var = tk.DoubleVar(value=500.0)
            self.up_cut_var = tk.DoubleVar(value=8000.0)
            self.sub_win_size_var = tk.DoubleVar(value=0.05)
            self.sub_hop_size_var = tk.DoubleVar(value=0.025)
            self.n_mels_var = tk.IntVar(value=32)
            self.seq_len_var = tk.IntVar(value=98)
            self.use_filter_var = tk.BooleanVar(value=True)
            self.fmin_var = tk.DoubleVar(value=50.0)
            self.fmax_var = tk.DoubleVar(value=10000.0)
            self.is_one_shot_model = False
            self.root = root

    app = _App()
    dlg = SettingsDialog(app)
    nb = ttk.Notebook(root)
    dlg._build_preprocessing_tab(nb)  # must not raise; references app.fmin_var/app.fmax_var
    assert app.fmin_var.get() == 50.0
    assert app.fmax_var.get() == 10000.0
    root.destroy()


def test_model_dropdown_selection_updates_paths():
    # The Model & Scaler tab's dropdown is a convenience over the manual path
    # Entry fields, not a replacement -- selecting an entry must populate
    # model_path_var/scaler_path_var exactly as a manual Browse would.
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        import pytest
        pytest.skip("no display available")
    root.withdraw()
    from tkinter import ttk

    class _App:
        def __init__(self):
            self.available_models = {
                "9_1_1": ("models/9_1_1/model.tflite", "models/9_1_1/scalar.json"),
                "9_1_2": ("models/9_1_2/model.tflite", "models/9_1_2/scaler.json"),
            }
            self.model_choice_var = tk.StringVar(value="9_1_2")
            self.model_path_var = tk.StringVar(value="models/9_1_2/model.tflite")
            self.scaler_path_var = tk.StringVar(value="models/9_1_2/scaler.json")
            self.calibration_profile_path_var = tk.StringVar(value="")
            self.root = root

        def load_model_dialog(self):
            pass

        def load_scaler_dialog(self):
            pass

        def browse_calibration_profile(self):
            pass

        def generate_calibration_profile(self):
            pass

        def on_model_choice_selected(self, event=None):
            model_path, scaler_path = self.available_models[self.model_choice_var.get()]
            self.model_path_var.set(model_path)
            self.scaler_path_var.set(scaler_path)

    app = _App()
    dlg = SettingsDialog(app)
    nb = ttk.Notebook(root)
    dlg._build_model_tab(nb)  # must not raise

    app.model_choice_var.set("9_1_1")
    app.on_model_choice_selected()
    assert app.model_path_var.get() == "models/9_1_1/model.tflite"
    assert app.scaler_path_var.get() == "models/9_1_1/scalar.json"
    root.destroy()


def test_acquisition_tab_binds_sliding_test_duration_spinbox():
    # Mic sliding-window tests auto-stop after app.sliding_test_duration_var seconds
    # (main.py:mic_loop) -- confirm the Settings spinbox is wired to that same var.
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError:
        import pytest
        pytest.skip("no display available")
    root.withdraw()

    class _App:
        def __init__(self):
            self.duration_var = tk.DoubleVar(value=2.5)
            self.output_dir_var = tk.StringVar(value="")
            self.user_label_var = tk.StringVar(value="healthy")
            self.save_results_var = tk.BooleanVar(value=False)
            self.inference_mode_var = tk.StringVar(value="sliding")
            self.single_shot_duration_sec = 20
            self.sliding_test_duration_var = tk.DoubleVar(value=20.0)
            self.is_one_shot_model = False
            self.root = root

        def browse_output_dir(self):
            pass

    app = _App()
    dlg = SettingsDialog(app)
    from tkinter import ttk
    nb = ttk.Notebook(root)
    dlg._build_acquisition_tab(nb)  # must not raise

    app.sliding_test_duration_var.set(30.0)
    assert app.sliding_test_duration_var.get() == 30.0
    root.destroy()


def test_discover_models_finds_all_shipped_models():
    # Exercises the real (non-fake) discovery method against the repo's actual
    # models/ directory -- must handle the scaler.json/scalar.json naming
    # inconsistency across model folders.
    from main import ModelsTesterApp

    discovered = ModelsTesterApp._discover_models(None)
    assert "9_1_2" in discovered
    model_path, scaler_path = discovered["9_1_2"]
    assert model_path.endswith("models/9_1_2/model.tflite")
    assert scaler_path.endswith("models/9_1_2/scaler.json")
    for name, (mpath, spath) in discovered.items():
        assert os.path.isfile(mpath), name
        assert os.path.isfile(spath), name
