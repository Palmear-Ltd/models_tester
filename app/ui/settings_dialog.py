"""Tabbed Settings dialog for the model tester.

A single modal window consolidating all configuration that does not live on the
main run bar. It binds directly to the `tk` variables already on the app, so
edits take effect on the next test start (same as before) — there is no separate
apply/cancel buffering. Built lazily when opened.
"""

import tkinter as tk
from tkinter import ttk


class SettingsDialog:
    """Modal, 3-tab settings window bound to the app's tk variables."""

    def __init__(self, app):
        self.app = app
        self.window = None

    def show(self):
        app = self.app
        self.window = tk.Toplevel(app.root)
        self.window.title("Settings")
        self.window.geometry("480x420")
        self.window.transient(app.root)
        self.window.grab_set()

        notebook = ttk.Notebook(self.window)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self._build_model_tab(notebook)
        self._build_acquisition_tab(notebook)
        self._build_preprocessing_tab(notebook)

        ttk.Button(self.window, text="Close", command=self.window.destroy).pack(
            fill="x", padx=10, pady=(0, 10)
        )

    def _build_model_tab(self, notebook):
        app = self.app
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text="Model & Scaler")
        tab.columnconfigure(1, weight=1)

        ttk.Label(tab, text="Model Path (.tflite):").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.model_path_var).grid(row=0, column=1, padx=5, sticky="ew")
        ttk.Button(tab, text="Browse", command=app.load_model_dialog).grid(row=0, column=2)

        ttk.Label(tab, text="Scaler Path (.json/.npz):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.scaler_path_var).grid(row=1, column=1, padx=5, sticky="ew")
        ttk.Button(tab, text="Browse", command=app.load_scaler_dialog).grid(row=1, column=2)

        ttk.Label(tab, text="Calibration Profile (.json):").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.calibration_profile_path_var).grid(row=2, column=1, padx=5, sticky="ew")
        ttk.Button(tab, text="Browse", command=app.browse_calibration_profile).grid(row=2, column=2)
        ttk.Button(
            tab, text="Generate from recordings…", command=app.generate_calibration_profile
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=5)

    def _build_acquisition_tab(self, notebook):
        app = self.app
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text="Acquisition & Output")
        tab.columnconfigure(1, weight=1)

        # Window duration is irrelevant for a one-shot model (it ignores it).
        if not getattr(app, "is_one_shot_model", False):
            ttk.Label(tab, text="Window Duration (sec):").grid(row=0, column=0, sticky="w", pady=5)
            ttk.Spinbox(tab, from_=0.5, to=10.0, increment=0.5, textvariable=app.duration_var, width=12).grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(tab, text="Output Dir:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.output_dir_var).grid(row=1, column=1, padx=5, sticky="ew")
        ttk.Button(tab, text="Browse", command=app.browse_output_dir).grid(row=1, column=2)

        ttk.Label(tab, text="User Label:").grid(row=2, column=0, sticky="w", pady=5)
        label_frame = ttk.Frame(tab)
        label_frame.grid(row=2, column=1, sticky="w", padx=5)
        ttk.Radiobutton(label_frame, text="Infested", variable=app.user_label_var, value="infested").pack(side="left", padx=5)
        ttk.Radiobutton(label_frame, text="Healthy", variable=app.user_label_var, value="healthy").pack(side="left", padx=5)

        ttk.Checkbutton(tab, text="Save results and audio", variable=app.save_results_var).grid(row=3, column=0, columnspan=3, sticky="w", pady=5)

        ttk.Label(tab, text="Inference Mode:").grid(row=4, column=0, sticky="w", pady=5)
        mode_frame = ttk.Frame(tab)
        mode_frame.grid(row=4, column=1, sticky="w", padx=5)
        ttk.Radiobutton(mode_frame, text="Sliding Window", variable=app.inference_mode_var, value="sliding").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text=f"Single Shot ({app.single_shot_duration_sec}s)", variable=app.inference_mode_var, value="single").pack(side="left", padx=5)

    def _build_preprocessing_tab(self, notebook):
        app = self.app
        tab = ttk.Frame(notebook, padding=12)
        notebook.add(tab, text="Preprocessing")
        tab.columnconfigure(1, weight=1)

        ttk.Checkbutton(tab, text="Use Bandpass Filter", variable=app.use_filter_var).grid(row=0, column=0, columnspan=2, sticky="w", pady=5)

        ttk.Label(tab, text="Low Cut (Hz):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.low_cut_var, width=15).grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(tab, text="Up Cut (Hz):").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.up_cut_var, width=15).grid(row=2, column=1, sticky="w", padx=5)

        ttk.Label(tab, text="Mel fmin (Hz):").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.fmin_var, width=15).grid(row=3, column=1, sticky="w", padx=5)

        ttk.Label(tab, text="Mel fmax (Hz):").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.fmax_var, width=15).grid(row=4, column=1, sticky="w", padx=5)

        ttk.Label(tab, text="FFT Win Size (s):").grid(row=5, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.sub_win_size_var, width=15).grid(row=5, column=1, sticky="w", padx=5)

        ttk.Label(tab, text="Hop Size (s):").grid(row=6, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.sub_hop_size_var, width=15).grid(row=6, column=1, sticky="w", padx=5)

        ttk.Label(tab, text="Mel Bands:").grid(row=7, column=0, sticky="w", pady=5)
        ttk.Entry(tab, textvariable=app.n_mels_var, width=15).grid(row=7, column=1, sticky="w", padx=5)

        # Sequence length does not apply to a one-shot model.
        if not getattr(app, "is_one_shot_model", False):
            ttk.Label(tab, text="Seq Len (frames):").grid(row=8, column=0, sticky="w", pady=5)
            ttk.Entry(tab, textvariable=app.seq_len_var, width=15).grid(row=8, column=1, sticky="w", padx=5)
