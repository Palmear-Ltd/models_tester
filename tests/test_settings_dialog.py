from app.ui.settings_dialog import SettingsDialog


class _FakeApp:
    """Minimal stand-in; SettingsDialog must construct without touching Tk."""

    root = None


def test_settings_dialog_constructs_without_display():
    dlg = SettingsDialog(_FakeApp())
    assert hasattr(dlg, "show")
    assert callable(dlg.show)
