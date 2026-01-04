import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
import time
import os
import json
import subprocess
import platform
from datetime import datetime
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from app.audio.processor import AudioProcessor
from app.audio.features import FeatureExtractor
from app.audio.scaler import Scaler
from app.model.inference import ModelInference
from app.ui.settings_dialog import SettingsDialog

class ModelsTesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Palmear Audio Testing Tool")
        self.root.geometry("1600x1000")
        self.root.minsize(1200, 800)
        
        self.processor = AudioProcessor()
        self.feature_extractor = FeatureExtractor()
        self.scaler = Scaler()
        self.model = ModelInference()
        self.is_running = False
        self.audio_thread = None
        self.audio_queue = queue.Queue()
        
        # Stats
        self.positive_count = 0
        self.negative_count = 0
        self.total_processed = 0
        
        # Preprocessing Config
        self.use_filter_var = tk.BooleanVar(value=True)
        self.low_cut_var = tk.DoubleVar(value=500.0)
        self.up_cut_var = tk.DoubleVar(value=8000.0)
        self.sub_win_size_var = tk.DoubleVar(value=0.05)
        self.sub_hop_size_var = tk.DoubleVar(value=0.025)
        self.n_mels_var = tk.IntVar(value=32)
        self.seq_len_var = tk.IntVar(value=98)
        self.duration_var = tk.DoubleVar(value=2.5)
        self.time_limit_var = tk.IntVar(value=20)  # Recording time limit in seconds (minimum 20)
        self.enable_downsample_var = tk.BooleanVar(value=False)
        self.downsample_sr_var = tk.IntVar(value=22050)  # Target sample rate for downsampling
        self.use_pcen_var = tk.BooleanVar(value=False)  # Use PCEN normalization instead of log (dB)
        
        # History Data for Plots
        self.score_history = []
        self.trigger_history = [] # (time, is_trigger)
        self.energy_history = []  # (time, rms)
        self.current_energy = 0.0
        self.start_time = 0
        self.raw_audio_snapshot = None # For waveform
        self.current_spectrogram = None # For heatmap
        self.current_frequency_spectrum = None # For frequency diagram
        self.frequency_bins = None # Frequency bins for x-axis
        
        # Scaler Data
        self.scaler_mean = None
        self.scaler_var = None

        # Default paths
        self.default_base_dir = os.path.join(os.getcwd(), "models", "9_1_2")
        self.default_model_path = os.path.join(self.default_base_dir, "model.tflite")
        self.default_scaler_path = os.path.join(self.default_base_dir, "scaler.npz")
        self.default_output_dir = os.path.join(self.default_base_dir, "output")
        os.makedirs(self.default_output_dir, exist_ok=True)

        # Saving options
        self.save_results_var = tk.BooleanVar(value=False) # single toggle controls both wav and json
        self.save_audio_var = self.save_results_var
        self.user_label_var = tk.StringVar(value="infested") # User expectation
        self.output_dir_var = tk.StringVar(value=self.default_output_dir)
        
        # Audio output (monitoring)
        self.enable_output_var = tk.BooleanVar(value=False)
        self.output_device_var = tk.StringVar()

        self._setup_ui()
        
    def _setup_ui(self):
        # Styles
        style = ttk.Style()
        # Prefer native theme when available to respect system light/dark (e.g., 'aqua' on macOS)
        preferred_theme = "aqua" if "aqua" in style.theme_names() else "clam"
        try:
            style.theme_use(preferred_theme)
        except Exception:
            pass

        # Menus
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Settings", command=self.open_prep_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Close", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        dark_mode = self._is_dark_mode()
        if dark_mode:
            base_bg = "#0b1221"
            text_color = "#e2e8f0"
            accent_color = "#3b82f6"
            border_color = "#1f2937"
        else:
            base_bg = "#f5f7fb"
            text_color = "#0f172a"
            accent_color = "#2563eb"
            border_color = "#e5e7eb"

        self.root.configure(bg=base_bg)

        style.configure("TFrame", background=base_bg)
        style.configure("TLabel", background=base_bg, foreground=text_color, font=("Helvetica", 11))
        style.configure("Header.TLabel", background=base_bg, foreground=text_color, font=("Helvetica", 12, "bold"))
        style.configure("Card.TLabelframe", background=base_bg, bordercolor=border_color, relief="solid")
        style.configure("Card.TLabelframe.Label", background=base_bg, foreground=text_color, font=("Helvetica", 11, "bold"))
        style.configure("Primary.TButton", font=("Helvetica", 10, "bold"), foreground="white", background=accent_color)
        style.map("Primary.TButton", background=[("active", "#1d4ed8")])
        style.configure("TCheckbutton", background=base_bg, foreground=text_color)
        style.configure("TRadiobutton", background=base_bg, foreground=text_color)
        
        # Main Layout: Left (Controls) | Right (Plots)
        main_pane = ttk.PanedWindow(self.root, orient="horizontal")
        main_pane.pack(fill="both", expand=True)
        
        left_frame = ttk.Frame(main_pane, padding=10, width=520) # give left pane a starting size
        right_frame = ttk.Frame(main_pane, padding=10) # For Plots
        main_pane.add(left_frame, weight=2)
        main_pane.add(right_frame, weight=3)
        # Set an initial sash position so the left panel is not squeezed on macOS HiDPI
        self.root.update_idletasks()
        main_pane.sashpos(0, 520)
        
        # --- Configuration Frame (Left) ---
        config_frame = ttk.LabelFrame(left_frame, text="Configuration", padding=12, style="Card.TLabelframe")
        config_frame.pack(fill="x", padx=10, pady=5)
        # Allow entries to expand horizontally so browse buttons stay visible on narrower panes
        config_frame.columnconfigure(0, minsize=150)  # Set minimum width for label column
        config_frame.columnconfigure(1, weight=1)
        
        # Model Path
        ttk.Label(config_frame, text="Model Path (.tflite):").grid(row=0, column=0, sticky="w")
        self.model_path_var = tk.StringVar(value=self.default_model_path)
        ttk.Entry(config_frame, textvariable=self.model_path_var).grid(row=0, column=1, padx=(2,2), sticky="ew")
        ttk.Button(config_frame, text="Browse", command=self.load_model_dialog).grid(row=0, column=2)

        # Scaler Path
        ttk.Label(config_frame, text="Scaler Path (.npz):").grid(row=1, column=0, sticky="w", pady=5)
        self.scaler_path_var = tk.StringVar(value=self.default_scaler_path)
        ttk.Entry(config_frame, textvariable=self.scaler_path_var).grid(row=1, column=1, padx=(2,2), sticky="ew")
        ttk.Button(config_frame, text="Browse", command=self.load_scaler_dialog).grid(row=1, column=2)

        # Test Duration (microphone auto-stop)
        ttk.Label(config_frame, text="Test Duration (sec):").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Spinbox(config_frame, from_=20, to=300, increment=5, textvariable=self.time_limit_var, width=10).grid(row=2, column=1, sticky="w", padx=(2,2))
        ttk.Label(config_frame, text="Minimum 20 seconds", font=("Helvetica", 10)).grid(row=2, column=2, sticky="w")

        # Input Check
        ttk.Label(config_frame, text="Input Source:").grid(row=3, column=0, sticky="w", pady=5)
        self.input_type_var = tk.StringVar(value="mic")

        input_frame = ttk.Frame(config_frame)
        input_frame.grid(row=3, column=1, sticky="w", padx=(2,2))
        ttk.Radiobutton(input_frame, text="Microphone", variable=self.input_type_var, value="mic", command=self.refresh_devices).pack(side="left", padx=5)
        ttk.Radiobutton(input_frame, text="Wav File", variable=self.input_type_var, value="file", command=self.refresh_devices).pack(side="left", padx=5)

        # Audio Device / File Path (moved down)
        self.device_label = ttk.Label(config_frame, text="Device:")
        self.device_label.grid(row=4, column=0, sticky="w", pady=5)
        
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(config_frame, textvariable=self.device_var)
        self.device_combo.grid(row=4, column=1, padx=(2,2), sticky="ew")
        
        self.file_btn = ttk.Button(config_frame, text="Browse", command=self.browse_wav_file)
        self.refresh_btn = ttk.Button(config_frame, text="⟳", command=self.refresh_devices, width=3)
        
        # Audio Output (Monitor) - only for microphone (create widgets before refresh_devices)
        self.output_check = ttk.Checkbutton(config_frame, text="Monitor Audio", variable=self.enable_output_var, command=self.toggle_output_device)
        self.output_device_label = ttk.Label(config_frame, text="Output Device:")
        self.output_device_combo = ttk.Combobox(config_frame, textvariable=self.output_device_var)
        
        # Save results (WAV + JSON) - create before refresh_devices
        self.save_check = ttk.Checkbutton(config_frame, text="Save results and audio", variable=self.save_results_var, command=self.toggle_output_dir)
        
        # Output directory for saved audio (create widgets, will be shown/hidden)
        self.output_dir_label = ttk.Label(config_frame, text="Results Path:")
        self.output_dir_entry = ttk.Entry(config_frame, textvariable=self.output_dir_var)
        self.output_dir_btn = ttk.Button(config_frame, text="Browse", command=self.browse_output_dir)
        
        self.refresh_devices()

        # Limits Config (outside main configuration block)
        limits_frame = ttk.LabelFrame(left_frame, text="Thresholds", padding=8, style="Card.TLabelframe")
        limits_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(limits_frame, text="Score Thresh:").pack(side="left", padx=2)
        self.score_thresh_var = tk.DoubleVar(value=0.5)
        ttk.Entry(limits_frame, textvariable=self.score_thresh_var, width=5).pack(side="left", padx=2)

        ttk.Label(limits_frame, text="Suspicious >=").pack(side="left", padx=2)
        self.susp_limit_var = tk.IntVar(value=17)
        ttk.Entry(limits_frame, textvariable=self.susp_limit_var, width=5).pack(side="left", padx=2)

        ttk.Label(limits_frame, text="Infested >=").pack(side="left", padx=2)
        self.inf_limit_var = tk.IntVar(value=27)
        ttk.Entry(limits_frame, textvariable=self.inf_limit_var, width=5).pack(side="left", padx=2)

        # --- Dashboard Frame (Left) ---
        dash_frame = ttk.LabelFrame(left_frame, text="Dashboard", padding=12, style="Card.TLabelframe")
        dash_frame.pack(fill="x", padx=10, pady=5)
        
        # Counters
        count_frame = ttk.Frame(dash_frame)
        count_frame.pack(fill="x", pady=5)
        
        # Positive
        pos_frame = ttk.Frame(count_frame, padding=5, relief="solid", borderwidth=1)
        pos_frame.pack(side="left", fill="both", expand=True, padx=5)
        ttk.Label(pos_frame, text="POSITIVE", font=("Helvetica", 14, "bold"), foreground="red").pack()
        self.pos_label = ttk.Label(pos_frame, text="0", font=("Helvetica", 24, "bold"))
        self.pos_label.pack()
        
        # Negative
        neg_frame = ttk.Frame(count_frame, padding=5, relief="solid", borderwidth=1)
        neg_frame.pack(side="left", fill="both", expand=True, padx=5)
        ttk.Label(neg_frame, text="NEGATIVE", font=("Helvetica", 14, "bold"), foreground="green").pack()
        self.neg_label = ttk.Label(neg_frame, text="0", font=("Helvetica", 24, "bold"))
        self.neg_label.pack()

        # Diagnosis
        self.diag_label = ttk.Label(dash_frame, text="No Result Yet", font=("Helvetica", 16, "bold"), foreground="gray")
        self.diag_label.pack(pady=10)

        # Energy Indicator
        energy_frame = ttk.Frame(dash_frame)
        energy_frame.pack(fill="x", pady=5)
        ttk.Label(energy_frame, text="Current Energy (RMS):").pack(side="left")
        self.energy_bar = ttk.Progressbar(energy_frame, orient="horizontal", length=200, mode="determinate", maximum=100)
        self.energy_bar.pack(side="left", padx=10, fill="x", expand=True)
        self.energy_value_label = ttk.Label(energy_frame, text="0.000")
        self.energy_value_label.pack(side="left", padx=5)
        
        # --- Control Area (Left) ---
        ctrl_frame = ttk.Frame(left_frame, padding=10)
        ctrl_frame.pack(fill="x")
        
        self.start_btn = ttk.Button(ctrl_frame, text="START TEST", command=self.toggle_test, width=20, style="Primary.TButton")
        self.start_btn.pack(pady=5)

        # Status Log
        self.status_log = tk.Listbox(left_frame, height=15)
        self.status_log.pack(fill="both", expand=True, padx=10, pady=5)
        
        # --- Plots Area (Right) ---
        # Scrollable container for individual interactive plots
        plot_container = ttk.Frame(right_frame)
        plot_container.pack(fill="both", expand=True)

        plot_canvas = tk.Canvas(plot_container, highlightthickness=0, background=base_bg)
        vscroll = ttk.Scrollbar(plot_container, orient="vertical", command=plot_canvas.yview)
        plot_canvas.configure(yscrollcommand=vscroll.set)
        plot_canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        plot_frame = ttk.Frame(plot_canvas)
        frame_window = plot_canvas.create_window((0, 0), window=plot_frame, anchor="nw")

        def _on_frame_configure(event):
            plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))

        def _on_canvas_configure(event):
            plot_canvas.itemconfigure(frame_window, width=event.width)

        plot_frame.bind("<Configure>", _on_frame_configure)
        plot_canvas.bind("<Configure>", _on_canvas_configure)

        def add_plot_section(title):
            section = ttk.Frame(plot_frame, padding=(0,4))
            section.pack(fill="x", expand=True)
            fig = Figure(figsize=(8, 2.6), dpi=100)
            ax = fig.add_subplot(111)
            canvas = FigureCanvasTkAgg(fig, master=section)
            toolbar = NavigationToolbar2Tk(canvas, section, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side="top", fill="x", pady=(0,2))
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            return ax, canvas

        self.ax_wave, self.canvas_wave = add_plot_section("Waveform")
        self.ax_freq, self.canvas_freq = add_plot_section("Frequency Spectrum")
        self.ax_hist, self.canvas_hist = add_plot_section("Score Distribution")
        self.ax_spec, self.canvas_spec = add_plot_section("Mel Spectrogram")
        self.ax_energy, self.canvas_energy = add_plot_section("Energy Timeline")
        self.ax_time, self.canvas_time = add_plot_section("Trigger Timeline")

    def _is_dark_mode(self):
        if platform.system() == "Darwin":
            try:
                result = subprocess.run([
                    "defaults", "read", "-g", "AppleInterfaceStyle"
                ], capture_output=True, text=True)
                return result.returncode == 0 and "Dark" in result.stdout
            except Exception:
                return False
        return False
        
    def open_prep_settings(self):
        """Open the preprocessing settings dialog."""
        config_vars = {
            'duration_var': self.duration_var,
            'seq_len_var': self.seq_len_var,
            'n_mels_var': self.n_mels_var,
            'sub_win_size_var': self.sub_win_size_var,
            'sub_hop_size_var': self.sub_hop_size_var,
            'use_filter_var': self.use_filter_var,
            'low_cut_var': self.low_cut_var,
            'up_cut_var': self.up_cut_var,
            'enable_downsample_var': self.enable_downsample_var,
            'downsample_sr_var': self.downsample_sr_var,
            'use_pcen_var': self.use_pcen_var
        }
        dialog = SettingsDialog(self.root, config_vars)
        dialog.show()

    def refresh_devices(self):
        input_type = self.input_type_var.get()
        if input_type == 'mic':
            self.file_btn.grid_remove()
            self.device_combo.grid(row=4, column=1, padx=(2,2), sticky="ew")
            self.refresh_btn.grid(row=4, column=2, padx=2)
            self.device_label.configure(text="Device:")
            
            # Show monitor controls for microphone
            self.output_check.grid(row=6, column=0, columnspan=2, sticky="w", pady=5)
            
            try:
                devices = sd.query_devices()
                input_devices = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]
                self.device_combo['values'] = input_devices
                if input_devices:
                    self.device_combo.current(0)
                
                # Populate output devices
                output_devices = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_output_channels'] > 0]
                self.output_device_combo['values'] = output_devices
                if output_devices:
                    self.output_device_combo.current(0)
                
                self.log("Audio devices refreshed")
            except Exception as e:
                self.log(f"Error listing devices: {e}")
            
            # Show monitor checkbox for microphone mode
            self.output_check.grid(row=6, column=0, columnspan=2, sticky="w", pady=5)
            
            # Show/hide output device selector and position save controls
            self.toggle_output_device()
        else:
            self.device_combo.grid(row=4, column=1, padx=(2,2), sticky="ew")
            self.refresh_btn.grid_remove()
            self.file_btn.grid(row=4, column=2)
            self.device_label.configure(text="Wav File:")
            
            # Hide monitor controls for file mode
            self.output_check.grid_remove()
            self.output_device_label.grid_remove()
            self.output_device_combo.grid_remove()
            self.save_check.grid_remove()
            self.output_dir_label.grid_remove()
            self.output_dir_entry.grid_remove()
            self.output_dir_btn.grid_remove()
            # reuse combo as entry for file path display
    
    def toggle_output_device(self):
        """Show or hide output device selector based on Monitor Audio checkbox"""
        if self.enable_output_var.get():
            self.output_device_label.grid(row=7, column=0, sticky="w", pady=5)
            self.output_device_combo.grid(row=7, column=1, padx=(2,2), sticky="ew")
            # Save checkbox moves to row 8
            self.save_check.grid(row=8, column=0, columnspan=3, sticky="w", pady=5)
        else:
            self.output_device_label.grid_remove()
            self.output_device_combo.grid_remove()
            # Save checkbox moves to row 7 (no output device showing, monitor checkbox is at row 6)
            self.save_check.grid(row=7, column=0, columnspan=3, sticky="w", pady=5)
        
        # Update output dir visibility
        self.toggle_output_dir()
    
    def toggle_output_dir(self):
        """Show or hide output directory based on Save checkbox"""
        if self.save_results_var.get():
            # Position depends on whether monitor audio is enabled
            if self.enable_output_var.get():
                # After save checkbox at row 8, so output dir at row 9
                row = 9
            else:
                # After save checkbox at row 7, so output dir at row 8
                row = 8
            self.output_dir_label.grid(row=row, column=0, sticky="w", pady=5)
            self.output_dir_entry.grid(row=row, column=1, padx=(2,2), sticky="ew")
            self.output_dir_btn.grid(row=row, column=2)
        else:
            self.output_dir_label.grid_remove()
            self.output_dir_entry.grid_remove()
            self.output_dir_btn.grid_remove()
            
    def display_file_entry(self):
        # Helper to swtich UI for file mode
        pass

    def load_model_dialog(self):
        path = filedialog.askopenfilename(filetypes=[("TFLite Models", "*.tflite")])
        if path:
            self.model_path_var.set(path)
            
    def load_scaler_dialog(self):
        path = filedialog.askopenfilename(filetypes=[("NPZ Files", "*.npz")])
        if path:
            self.scaler_path_var.set(path)
            
    def browse_wav_file(self):
        path = filedialog.askopenfilename(filetypes=[("Wav Files", "*.wav")])
        if path:
            self.device_var.set(path)

    def browse_output_dir(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir_var.set(path)
            
    def log(self, message):
        if hasattr(self, 'status_log'):
            self.status_log.insert(tk.END, message)
            self.status_log.see(tk.END)
        
    def load_resources(self):
        # Load Model
        model_path = self.model_path_var.get()
        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Model file not found!")
            return False
            
        if not ModelInference.is_tflite_available():
            messagebox.showerror("Error", "TFLite runtime not installed!")
            return False
            
        try:
            self.model.load_model(model_path)
            self.log(f"Loaded Model: {os.path.basename(model_path)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return False
            
        # Load Scaler
        scaler_path = self.scaler_path_var.get()
        if not os.path.exists(scaler_path):
            messagebox.showerror("Error", "Scaler file not found!")
            return False
            
        self.scaler_mean, self.scaler_var = self.scaler.load(scaler_path)
        if self.scaler_mean is None:
            messagebox.showerror("Error", "Failed to load scaler data.")
            return False
        self.log(f"Loaded Scaler: {os.path.basename(scaler_path)}")
        
        return True

    def toggle_test(self):
        if self.is_running:
            self.stop_test()
        else:
            if self.load_resources():
                self.start_test()
                
    def start_test(self):
        self.is_running = True
        self.positive_count = 0
        self.negative_count = 0
        self.pos_label.configure(text="0")
        self.neg_label.configure(text="0")
        self.diag_label.configure(text="Testing...", foreground="blue")
        
        # Reset History
        self.score_history = []
        self.trigger_history = []
        self.energy_history = []
        self.start_time = time.time()
        self.recording_start_time = time.time()  # Track when recording started
        
        self.start_btn.configure(text="STOP TEST")
        self.audio_queue = queue.Queue()
        
        # Start Thread based on input
        if self.input_type_var.get() == "mic":
            device_idx = int(self.device_var.get().split(":")[0])
            self.audio_thread = threading.Thread(target=self.mic_loop, args=(device_idx,))
        else:
            file_path = self.device_var.get()
            self.audio_thread = threading.Thread(target=self.file_loop, args=(file_path,))
            
        self.audio_thread.start()
        self.root.after(100, self.process_queue)
        self.root.after(500, self.update_plots) # Update plots every 500ms
        
    def stop_test(self):
        self.is_running = False
        if self.audio_thread:
            self.audio_thread.join(timeout=1.0)
        
        # Wait a bit for any final callbacks to complete
        time.sleep(0.2)
        
        # Process any remaining chunks in the queue
        remaining_count = 0
        try:
            while True:
                item = self.audio_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == "ERROR":
                    self.log(f"Error: {item[1]}")
                    break
                # Process remaining audio chunks
                self.handle_audio_chunk(item)
                remaining_count += 1
        except queue.Empty:
            pass
        
        if remaining_count > 0:
            self.log(f"Processed {remaining_count} remaining chunks")
        
        self.start_btn.configure(text="START TEST")
        predicted_infested = self.calculate_diagnosis()

        audio_path = None
        if self.save_results_var.get():
            # Ask user for label before saving
            user_label = self.ask_user_label()
            if user_label:  # Only save if user didn't cancel
                self.user_label_var.set(user_label)  # Set the label for use in file naming
                if self.input_type_var.get() == "mic":
                    audio_path = self.save_audio_snapshot(predicted_infested)
                self.save_results(predicted_infested, audio_path)
            else:
                self.log("Save cancelled; results not written")
        self.log("Test Stopped.")
    
    def ask_user_label(self):
        """Show dialog to ask user for label before saving"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Save Results")
        dialog.geometry("350x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        result = {"label": None}
        
        ttk.Label(dialog, text="What is the expected label for this sample?", font=("Helvetica", 11)).pack(pady=15)
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def select_label(label):
            result["label"] = label
            dialog.destroy()
        
        ttk.Button(button_frame, text="Infested", command=lambda: select_label("infested"), width=12).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Healthy", command=lambda: select_label("healthy"), width=12).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Unknown", command=lambda: select_label("unknown"), width=12).pack(side="left", padx=5)
        
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=5)
        
        self.root.wait_window(dialog)
        return result["label"]
        
    def calculate_diagnosis(self):
        pos = self.positive_count
        susp_limit = self.susp_limit_var.get()
        inf_limit = self.inf_limit_var.get()
        predicted_infested = False
        
        if pos < susp_limit:
            self.diag_label.configure(text=f"HEALTHY (Pos: {pos})", foreground="green")
        elif pos < inf_limit:
            self.diag_label.configure(text=f"SUSPICIOUS (Pos: {pos})", foreground="orange")
        else:
            self.diag_label.configure(text=f"INFESTED (Pos: {pos})", foreground="red")
            predicted_infested = True
        return predicted_infested

    def mic_loop(self, device_idx):
        time_limit = self.time_limit_var.get()
        if time_limit > 0:
            self.log(f"Starting Mic Stream on Device {device_idx} (Time Limit: {time_limit}s)...")
            # Calculate expected number of chunks (0.5s per chunk)
            expected_chunks = int(time_limit / 0.5)
            self.log(f"Expected chunks: {expected_chunks}")
        else:
            self.log(f"Starting Mic Stream on Device {device_idx} (No Time Limit)...")
            expected_chunks = 0
        
        sample_rate = 44100
        buffer_duration = self.duration_var.get()
        buffer_len = int(sample_rate * buffer_duration)
        
        # Initialize buffer with zeros
        audio_buffer = np.zeros(buffer_len, dtype=np.float32)
        
        block_size = int(sample_rate * 0.5) # 0.5 sec blocks
        
        # Chunk counter
        chunk_count = [0]  # Use list to allow modification in callback
        
        # Monitoring buffer for output (store latest block)
        monitor_buffer = {
            "data": np.zeros((block_size, 1), dtype=np.float32)
        }
        
        # Prepare output device if monitoring is enabled
        output_device_idx = None
        if self.enable_output_var.get():
            try:
                output_device_idx = int(self.output_device_var.get().split(":")[0])
                self.log(f"Audio monitoring enabled on device {output_device_idx}")
            except:
                self.log("Warning: Could not parse output device, monitoring disabled")
        
        def callback(indata, frames, time, status):
            chunk_count[0] += 1
            self.audio_queue.put(indata.copy())
            # Update monitor buffer with latest block
            monitor_buffer["data"] = indata.copy()
            
        try:
            if output_device_idx is not None:
                # Single output stream with callback fed from latest monitored buffer
                def output_callback(outdata, frames, time, status):
                    buf = monitor_buffer.get("data")
                    if buf is None or buf.size == 0:
                        outdata.fill(0)
                        return
                    # Ensure shape match and pad/trim as needed
                    if buf.shape[0] < frames:
                        padded = np.zeros((frames, 1), dtype=np.float32)
                        padded[:buf.shape[0], :] = buf
                        outdata[:] = padded
                    else:
                        outdata[:] = buf[:frames]

                with sd.InputStream(
                    device=device_idx,
                    channels=1,
                    samplerate=sample_rate,
                    blocksize=block_size,
                    callback=callback,
                ) as _input_stream, sd.OutputStream(
                    device=output_device_idx,
                    channels=1,
                    samplerate=sample_rate,
                    blocksize=block_size,
                    callback=output_callback,
                ) as _output_stream:
                    while self.is_running:
                        # Check chunk limit if set (0 means no limit)
                        if expected_chunks > 0 and chunk_count[0] >= expected_chunks:
                            self.log(f"Chunk limit reached ({chunk_count[0]} chunks). Stopping...")
                            self.is_running = False
                            break
                        time.sleep(0.05)
            else:
                # No output, just input
                with sd.InputStream(device=device_idx, channels=1, samplerate=sample_rate, 
                                    blocksize=block_size, callback=callback):
                    while self.is_running:
                        # Check chunk limit if set (0 means no limit)
                        if expected_chunks > 0 and chunk_count[0] >= expected_chunks:
                            self.log(f"Chunk limit reached ({chunk_count[0]} chunks). Stopping...")
                            self.is_running = False
                            break
                        time.sleep(0.05)
        except Exception as e:
            self.audio_queue.put(("ERROR", str(e)))
            
    def file_loop(self, file_path):
        self.log(f"Processing File: {file_path}")
        try:
            data, fs = sf.read(file_path, always_2d=True)
            # Mix to mono
            if data.shape[1] > 1:
                data = np.mean(data, axis=1)
            else:
                data = data[:, 0]
                
            # Resample if needed
            if fs != 44100:
                self.log(f"Resampling from {fs} to 44100...")
                data = librosa.resample(data, orig_sr=fs, target_sr=44100)
                fs = 44100
                
            block_size_samples = int(fs * 0.5)
            total_samples = len(data)
            
            idx = 0
            while self.is_running and idx < total_samples:
                end = min(idx + block_size_samples, total_samples)
                chunk = data[idx:end]
                
                # If last chunk is small, pad?
                if len(chunk) < block_size_samples:
                    chunk = np.pad(chunk, (0, block_size_samples - len(chunk)))
                    
                self.audio_queue.put(chunk.reshape(-1, 1))
                idx += block_size_samples
                time.sleep(0.5) # Simulate real-time
                
            self.log("File finished.")
            self.is_running = False
            
        except Exception as e:
            self.audio_queue.put(("ERROR", str(e)))
            
    def process_queue(self):
        if not self.is_running:
            if self.start_btn.cget("text") == "STOP TEST":
                 self.stop_test()
            return
            
        try:
            while True:
                item = self.audio_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == "ERROR":
                    self.log(f"Error: {item[1]}")
                    self.stop_test()
                    return
                
                # Assume item is audio chunk
                self.handle_audio_chunk(item)
        except queue.Empty:
            pass
            
        self.root.after(100, self.process_queue)
        
    def handle_audio_chunk(self, chunk):
        # We need a persistent buffer for the session
        if not hasattr(self, 'session_buffer'):
            buffer_len = int(44100 * self.duration_var.get())
            self.session_buffer = np.zeros(buffer_len, dtype=np.float32)
            
        # Shift and append
        chunk_flat = chunk.flatten()
        chunk_len = len(chunk_flat)
        
        self.session_buffer = np.roll(self.session_buffer, -chunk_len)
        self.session_buffer[-chunk_len:] = chunk_flat
        
        # Save snapshot for plotting
        self.raw_audio_snapshot = self.session_buffer.copy()
        
        # Run Inference
        self.run_inference(self.session_buffer)

    def _derive_output_base(self, predicted_infested):
        user_choice = self.user_label_var.get()
        predicted_infested = bool(predicted_infested)
        if user_choice == "infested":
            tag = "TP" if predicted_infested else "FN"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        elif user_choice == "healthy":
            tag = "TN" if not predicted_infested else "FP"
            timestamp = datetime.now().strftime("%Y%m%d_%I%M%p") if tag == "TN" else datetime.now().strftime("%Y%m%d_%H%M%S")
        else:  # unknown
            tag = "U"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        out_dir = self.output_dir_var.get().strip() or "recordings"
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, f"{tag}_{timestamp}")
        return base

    def save_audio_snapshot(self, predicted_positive):
        if not hasattr(self, 'session_buffer'):
            return None
        try:
            base = self._derive_output_base(predicted_positive)
            wav_path = f"{base}.wav"
            sf.write(wav_path, self.session_buffer, 44100)
            self.log(f"Saved audio: {os.path.basename(wav_path)}")
            return wav_path
        except Exception as e:
            self.log(f"Save audio error: {e}")
            return None

    def save_results(self, predicted_infested, audio_path=None):
        try:
            base = os.path.splitext(audio_path)[0] if audio_path else self._derive_output_base(predicted_infested)

            data = {
                "timestamp": datetime.now().isoformat(),
                "duration_sec": self.duration_var.get(),
                "model_path": self.model_path_var.get(),
                "scaler_path": self.scaler_path_var.get(),
                "user_label": self.user_label_var.get(),
                "predicted_infested": bool(predicted_infested),
                    "positive_count": self.positive_count,
                    "negative_count": self.negative_count,
                    "total_processed": self.positive_count + self.negative_count,
                "score_thresh": self.score_thresh_var.get(),
                "susp_limit": self.susp_limit_var.get(),
                "inf_limit": self.inf_limit_var.get(),
                "n_mels": self.n_mels_var.get(),
                "seq_len": self.seq_len_var.get(),
                "low_cut": self.low_cut_var.get(),
                "up_cut": self.up_cut_var.get(),
                "sub_win_size": self.sub_win_size_var.get(),
                "sub_hop_size": self.sub_hop_size_var.get(),
                "save_results_enabled": self.save_results_var.get(),
                "input_type": self.input_type_var.get(),
                "score_history": [float(s) for s in self.score_history],
                "trigger_history": [(float(t), int(v)) for t, v in self.trigger_history],
                "energy_history": [(float(t), float(v)) for t, v in self.energy_history],
            }

            results_path = f"{base}.json"
            results_dir = os.path.dirname(results_path)
            if results_dir:
                os.makedirs(results_dir, exist_ok=True)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.log(f"Saved results: {results_path}")
        except Exception as e:
            self.log(f"Save results error: {e}")
        
    def run_inference(self, audio_data):
        try:
            # 0. Calculate Energy (RMS)
            rms = np.sqrt(np.mean(audio_data**2))
            self.current_energy = rms
            scaled_rms = min(rms * 100, 100)
            self.energy_bar["value"] = scaled_rms  # Progressbar scaled to 0-100
            if hasattr(self, "energy_value_label"):
                self.energy_value_label.config(text=f"{rms:.3f}")
            
            # 1. Extract Features with Config
            specs = self.feature_extractor.extract_features(
                audio_data, 
                sr=44100,
                n_mels=self.n_mels_var.get(),
                seq_len=self.seq_len_var.get(),
                low_cut=self.low_cut_var.get(),
                up_cut=self.up_cut_var.get(),
                sub_win_size_sec=self.sub_win_size_var.get(),
                sub_hop_size_sec=self.sub_hop_size_var.get(),
                use_filter=self.use_filter_var.get(),
                enable_downsample=self.enable_downsample_var.get(),
                downsample_sr=self.downsample_sr_var.get(),
                use_pcen=self.use_pcen_var.get()
            )
            
            # Save for plotting (Mel Spectrogram in dB)
            self.current_spectrogram = specs.copy()
            
            # Calculate Frequency Spectrum using FFT
            fft_size = 4096
            if len(audio_data) >= fft_size:
                # Use the last fft_size samples
                fft_data = audio_data[-fft_size:]
                # Apply window to reduce spectral leakage
                window = np.hanning(fft_size)
                windowed_data = fft_data * window
                # Compute FFT
                fft_vals = np.fft.rfft(windowed_data)
                fft_magnitude = np.abs(fft_vals)
                # Convert to dB scale
                fft_magnitude_db = 20 * np.log10(fft_magnitude + 1e-10)
                # Create frequency bins (Hz)
                self.frequency_bins = np.fft.rfftfreq(fft_size, 1/44100)
                self.current_frequency_spectrum = fft_magnitude_db
            
            # 2. Scale
            specs_scaled = self.scaler.apply(specs, self.scaler_mean, self.scaler_var)
            
            # 3. Reshape for Model
            # Model Input: (1, seq_len, n_mels, 1)
            seq_len = self.seq_len_var.get()
            n_mels = self.n_mels_var.get()
            input_data = specs_scaled.reshape(1, seq_len, n_mels, 1).astype(np.float32)
            
            # 4. Invoke Model
            score_thresh = self.score_thresh_var.get()
            predicted, score = self.model.predict_with_score(input_data, threshold=score_thresh)

            # Store stats for plot
            self.score_history.append(score)
            curr_time = time.time() - self.start_time
            self.trigger_history.append((curr_time, predicted))
            self.energy_history.append((curr_time, rms))

            # Update Stats
            if predicted == 1:
                self.positive_count += 1
                self.pos_label.configure(text=str(self.positive_count))
                res_str = f"POS ({score:.2f})"
            else:
                self.negative_count += 1
                self.neg_label.configure(text=str(self.negative_count))
                res_str = f"NEG ({score:.2f})"
                
            self.log(f"Processed: {res_str}")
                
        except Exception as e:
            self.log(f"Inference Error: {e}")

    def update_plots(self):
        if not self.is_running:
            return
            
        try:
            # Update Waveform
            if self.raw_audio_snapshot is not None:
                self.ax_wave.clear()
                self.ax_wave.plot(self.raw_audio_snapshot[::10]) # Decimate for speed
                self.ax_wave.set_title("Audio Waveform")
                self.ax_wave.set_ylim(-1, 1)
                self.ax_wave.grid(True)
                self.canvas_wave.draw()
            
            # Update Frequency Spectrum
            if self.current_frequency_spectrum is not None and self.frequency_bins is not None:
                self.ax_freq.clear()
                # Plot frequency spectrum, limit to 0-10kHz for better visualization
                freq_limit = 10000  # Hz
                freq_mask = self.frequency_bins <= freq_limit
                self.ax_freq.plot(self.frequency_bins[freq_mask], 
                                 self.current_frequency_spectrum[freq_mask], 
                                 'b-', linewidth=0.8)
                self.ax_freq.set_title("Frequency Spectrum (FFT)")
                self.ax_freq.set_xlabel("Frequency (Hz)")
                self.ax_freq.set_ylabel("Magnitude (dB)")
                self.ax_freq.set_xlim(0, freq_limit)
                self.ax_freq.grid(True, alpha=0.3)
                self.canvas_freq.draw()
            
            # Update Histogram (Score)
            if self.score_history:
                self.ax_hist.clear()
                self.ax_hist.hist(self.score_history, bins=10, range=(0,1), color='skyblue', edgecolor='black')
                self.ax_hist.set_title("Score Distribution")
                self.ax_hist.set_xlim(0, 1)
                self.ax_hist.grid(True)
                self.canvas_hist.draw()

            # Update Spectrogram (Heatmap)
            if self.current_spectrogram is not None:
                self.ax_spec.clear()
                # Transpose to have Freq on Y, Time on X
                # specs is (98, 32) -> .T is (32, 98)
                self.ax_spec.imshow(self.current_spectrogram.T, aspect='auto', origin='lower', cmap='inferno')
                self.ax_spec.set_title("Mel Spectrogram")
                self.ax_spec.set_ylabel("Mel Bands")
                self.ax_spec.set_xlabel("Time Frames")
                self.canvas_spec.draw()

            # Update Energy
            if self.energy_history:
                times, energy = zip(*self.energy_history)
                self.ax_energy.clear()
                self.ax_energy.plot(times, energy, 'g-')
                self.ax_energy.set_title("Energy Timeline (RMS)")
                self.ax_energy.set_ylim(0, max(0.1, max(energy) * 1.2)) # Dynamic limit
                self.ax_energy.grid(True)
                self.canvas_energy.draw()
            
            # Update Timeline
            if self.trigger_history:
                times, triggers = zip(*self.trigger_history)
                self.ax_time.clear()
                self.ax_time.plot(times, triggers, 'r-o', markersize=4)
                self.ax_time.set_title("Trigger Timeline (1=Pos, 0=Neg)")
                self.ax_time.set_ylim(-0.1, 1.1)
                self.ax_time.set_xlabel("Time (s)")
                self.ax_time.grid(True)
                self.canvas_time.draw()
            
        except Exception as e:
            self.log(f"Plot Error: {e}")

        if self.is_running:
            self.root.after(1000, self.update_plots)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelsTesterApp(root)
    root.mainloop()
