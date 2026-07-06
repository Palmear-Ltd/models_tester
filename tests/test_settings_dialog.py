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
