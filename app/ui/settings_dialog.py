"""
Settings dialog for preprocessing pipeline configuration.

Provides a comprehensive interface for configuring:
- Model parameters (duration, sequence length, mel bands, FFT settings)
- Bandpass filter settings (optional)
- Sample rate and normalization options (downsampling, PCEN)
"""

import tkinter as tk
from tkinter import ttk


class SettingsDialog:
    """
    Settings dialog for preprocessing pipeline configuration.
    
    Creates a modal dialog window with three main sections:
    1. Model Settings
    2. Bandpass Filter (Optional)
    3. Sample Rate & Normalization
    """
    
    def __init__(self, parent, config_vars):
        """
        Initialize settings dialog.
        
        Args:
            parent: Parent window (root or Toplevel)
            config_vars: Dictionary containing all configuration variables:
                - duration_var: tk.DoubleVar for audio duration
                - seq_len_var: tk.IntVar for sequence length
                - n_mels_var: tk.IntVar for mel bands
                - sub_win_size_var: tk.DoubleVar for FFT window size
                - sub_hop_size_var: tk.DoubleVar for hop size
                - use_filter_var: tk.BooleanVar for filter enable
                - low_cut_var: tk.DoubleVar for low cut frequency
                - up_cut_var: tk.DoubleVar for high cut frequency
                - enable_downsample_var: tk.BooleanVar for downsample enable
                - downsample_sr_var: tk.IntVar for target sample rate
                - use_pcen_var: tk.BooleanVar for PCEN normalization
        """
        self.parent = parent
        self.config_vars = config_vars
        self.window = None
    
    def show(self):
        """Display the settings dialog."""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Pipeline Configuration")
        self.window.geometry("500x640")
        self.window.transient(self.parent)
        self.window.grab_set()

        # Create scrollable frame
        canvas = tk.Canvas(self.window, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")

        # Build sections
        self._build_model_settings(scrollable_frame)
        self._build_filter_settings(scrollable_frame)
        self._build_sample_rate_settings(scrollable_frame)
        self._build_buttons(scrollable_frame)
    
    def _build_model_settings(self, parent):
        """Build the Model Settings section."""
        model_frame = ttk.LabelFrame(parent, text="Model Settings", padding=12)
        model_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(model_frame, text="Duration (sec):").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Spinbox(model_frame, from_=0.5, to=10.0, increment=0.5, 
                    textvariable=self.config_vars['duration_var'], width=20).grid(row=0, column=1, padx=5, sticky="ew")

        ttk.Label(model_frame, text="Sequence Length (frames):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(model_frame, textvariable=self.config_vars['seq_len_var'], width=20).grid(row=1, column=1, padx=5, sticky="ew")

        ttk.Label(model_frame, text="Mel Bands:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(model_frame, textvariable=self.config_vars['n_mels_var'], width=20).grid(row=2, column=1, padx=5, sticky="ew")

        ttk.Label(model_frame, text="FFT Window Size (s):").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(model_frame, textvariable=self.config_vars['sub_win_size_var'], width=20).grid(row=3, column=1, padx=5, sticky="ew")

        ttk.Label(model_frame, text="Hop Size (s):").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(model_frame, textvariable=self.config_vars['sub_hop_size_var'], width=20).grid(row=4, column=1, padx=5, sticky="ew")

        model_frame.columnconfigure(1, weight=1)
    
    def _build_filter_settings(self, parent):
        """Build the Bandpass Filter section."""
        filter_frame = ttk.LabelFrame(parent, text="Bandpass Filter (Optional)", padding=12)
        filter_frame.pack(fill="x", padx=10, pady=10)

        # Enable filter checkbox
        filter_check_frame = ttk.Frame(filter_frame)
        filter_check_frame.grid(row=0, column=0, columnspan=2, sticky="w", pady=5)
        filter_check = ttk.Checkbutton(
            filter_check_frame, 
            variable=self.config_vars['use_filter_var'],
            command=lambda: self._toggle_filter_settings(filter_low_label, filter_low_entry, filter_up_label, filter_up_entry)
        )
        filter_check.pack(side="left")
        ttk.Label(filter_check_frame, text="Enable Filter").pack(side="left", padx=(2, 0))

        # Filter parameters
        filter_low_label = ttk.Label(filter_frame, text="Low Cut (Hz):")
        filter_low_label.grid(row=1, column=0, sticky="w", pady=5)
        filter_low_entry = ttk.Entry(filter_frame, textvariable=self.config_vars['low_cut_var'], width=20)
        filter_low_entry.grid(row=1, column=1, padx=5, sticky="ew")

        filter_up_label = ttk.Label(filter_frame, text="Up Cut (Hz):")
        filter_up_label.grid(row=2, column=0, sticky="w", pady=5)
        filter_up_entry = ttk.Entry(filter_frame, textvariable=self.config_vars['up_cut_var'], width=20)
        filter_up_entry.grid(row=2, column=1, padx=5, sticky="ew")

        filter_frame.columnconfigure(1, weight=1)
        
        # Initialize filter settings state
        self._toggle_filter_settings(filter_low_label, filter_low_entry, filter_up_label, filter_up_entry)
    
    def _build_sample_rate_settings(self, parent):
        """Build the Sample Rate & Normalization section."""
        sr_frame = ttk.LabelFrame(parent, text="Sample Rate & Normalization", padding=12)
        sr_frame.pack(fill="x", padx=10, pady=10)

        # Downsample checkbox
        downsample_check_frame = ttk.Frame(sr_frame)
        downsample_check_frame.grid(row=0, column=0, columnspan=2, sticky="w", pady=5)
        downsample_check = ttk.Checkbutton(
            downsample_check_frame,
            variable=self.config_vars['enable_downsample_var'],
            command=lambda: self._toggle_downsample_settings(downsample_label, downsample_spin)
        )
        downsample_check.pack(side="left")
        ttk.Label(downsample_check_frame, text="Downsample Audio").pack(side="left", padx=(2, 0))

        # Downsample parameters
        downsample_label = ttk.Label(sr_frame, text="Target Sample Rate (Hz):")
        downsample_label.grid(row=1, column=0, sticky="w", pady=5)
        downsample_spin = ttk.Spinbox(
            sr_frame, 
            from_=8000, to=44100, increment=2000,
            textvariable=self.config_vars['downsample_sr_var'], 
            width=20
        )
        downsample_spin.grid(row=1, column=1, padx=5, sticky="ew")
        
        # Initialize downsample settings state
        self._toggle_downsample_settings(downsample_label, downsample_spin)

        # Separator
        ttk.Separator(sr_frame, orient="horizontal").grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)

        # PCEN checkbox
        pcen_check_frame = ttk.Frame(sr_frame)
        pcen_check_frame.grid(row=3, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Checkbutton(pcen_check_frame, variable=self.config_vars['use_pcen_var']).pack(side="left")
        ttk.Label(pcen_check_frame, text="Use PCEN Normalization").pack(side="left", padx=(2, 0))

        ttk.Label(sr_frame, text="(PCEN: bioacoustics, Log: standard)", 
                  font=("TkDefaultFont", 8)).grid(row=4, column=0, columnspan=2, sticky="w", pady=0)

        sr_frame.columnconfigure(1, weight=1)
    
    def _build_buttons(self, parent):
        """Build the button section."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill="x", padx=10, pady=15)
        ttk.Button(button_frame, text="Close", command=self.window.destroy).pack(fill="x")
    
    def _toggle_filter_settings(self, low_label, low_entry, up_label, up_entry):
        """Enable or disable filter settings based on checkbox state."""
        state = "normal" if self.config_vars['use_filter_var'].get() else "disabled"
        low_label.configure(state=state)
        low_entry.configure(state=state)
        up_label.configure(state=state)
        up_entry.configure(state=state)

    def _toggle_downsample_settings(self, label, spinbox):
        """Enable or disable downsample settings based on checkbox state."""
        state = "normal" if self.config_vars['enable_downsample_var'].get() else "disabled"
        label.configure(state=state)
        spinbox.configure(state=state)
