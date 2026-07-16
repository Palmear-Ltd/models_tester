import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from inference_utils import AudioProcessor
from app.health.config import DEFAULT_CHECK_THRESHOLDS_PATH, PROFILES, pipeline_for_profile
from app.health.models import AudioWindow, HealthState
from app.health.reporting import report_rows, root_cause_row
from app.health.monitoring import RuntimeMonitor
from app.health.startup import StartupDecision, run_validation
from app.health.serialization import (
    startup_result_to_dict,
    anomaly_event_to_dict,
    root_cause_to_dict,
)
from app.health import rootcause
from app.ui import SettingsDialog
from app.decision.threshold import EwmaPeakDecision, default_config as default_decision_config

SAMPLE_RATE = 44100  # acquisition sample rate (Hz); the whole pipeline runs at this rate
VALIDATION_WINDOWS = 40  # ~20 s of validation at 0.5 s per window
USER_SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_settings.json")


def mic_capture_duration_sec(inference_mode, single_shot_duration_sec, sliding_test_duration_sec):
    """Max mic capture duration (seconds) before a test auto-stops, by inference mode."""
    if inference_mode == "single":
        return single_shot_duration_sec
    return sliding_test_duration_sec

# Try to import TFLite Interpreter
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None

class ModelsTesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Palmear Audio Testing Tool")
        self.root.geometry("1600x1000")
        self.root.minsize(1200, 800)
        
        self.processor = AudioProcessor()
        self.profile_var = tk.StringVar(value="development")
        self.calibration_profile = None
        self.calibration_profile_path_var = tk.StringVar(value="")
        self.health_pipeline = pipeline_for_profile(
            self.profile_var.get(), calibration_profile=self.calibration_profile
        )
        self.latest_health_report = None
        self._last_health_state = None
        self.runtime_monitor = RuntimeMonitor()
        self._validating = False
        self._validation_reports = []
        self._last_anomalous = False
        self._last_root_cause = None
        self.is_running = False
        self.audio_thread = None
        self.audio_queue = queue.Queue()
        
        # Stats
        self.total_processed = 0
        
        # Preprocessing Config
        # Default off: the mobile app ships with the bandpass filter disabled
        # (opt-in via app settings), and most users never enable it.
        self.use_filter_var = tk.BooleanVar(value=False)
        self.low_cut_var = tk.DoubleVar(value=500.0)
        self.up_cut_var = tk.DoubleVar(value=8000.0)
        self.sub_win_size_var = tk.DoubleVar(value=0.05)
        self.fmin_var = tk.DoubleVar(value=50.0)
        self.fmax_var = tk.DoubleVar(value=10000.0)
        self.sub_hop_size_var = tk.DoubleVar(value=0.025)
        self.n_mels_var = tk.IntVar(value=32)
        self.seq_len_var = tk.IntVar(value=98)
        self.inference_mode_var = tk.StringVar(value="sliding")
        self.single_shot_duration_sec = 20
        # Sliding-window mic tests auto-stop after this many seconds (like single-shot
        # already did), so a tester isn't stuck manually clicking STOP TEST every run.
        self.sliding_test_duration_var = tk.DoubleVar(value=20.0)
        self.collected_audio_chunks = []
        self.final_audio_clip = None
        self.single_shot_processed = False
        
        # History Data for Plots
        self.score_history = []
        self.trigger_history = [] # (time, is_trigger)
        self.energy_history = []  # (time, rms)
        self.health_state_history = []  # (time, level: 0=OK, 1=WARNING, 2=FAULT)
        self.current_energy = 0.0
        self.start_time = 0
        self.raw_audio_snapshot = None # For waveform
        self.current_spectrogram = None # For heatmap
        
        # Scaler Data
        self.scaler_mean = None
        self.scaler_var = None

        # Default paths
        self.default_base_dir = os.path.join(os.getcwd(), "models", "9_1_2")
        self.default_model_path = os.path.join(self.default_base_dir, "model.tflite")
        self.default_scaler_path = os.path.join(self.default_base_dir, "scaler.json")
        self.default_output_dir = os.path.join(self.default_base_dir, "output")
        os.makedirs(self.default_output_dir, exist_ok=True)

        # Preloaded models: models/<name>/ folders that contain both a model.tflite
        # and a scaler file, so the Settings dialog can offer a dropdown instead of
        # requiring manual Browse for every model swap.
        self.available_models = self._discover_models()
        default_model_name = "9_1_2" if "9_1_2" in self.available_models else next(iter(self.available_models), "")
        self.model_choice_var = tk.StringVar(value=default_model_name)

        # Saving options
        self.save_results_var = tk.BooleanVar(value=False) # single toggle controls both wav and json
        self.save_audio_var = self.save_results_var
        self.user_label_var = tk.StringVar(value="infested") # User expectation
        self.output_dir_var = tk.StringVar(value=self.default_output_dir)
        self.model_path_var = tk.StringVar(value=self.default_model_path)
        self.scaler_path_var = tk.StringVar(value=self.default_scaler_path)
        self.duration_var = tk.DoubleVar(value=2.5)
        # Single-shot mode's per-window threshold. The sliding-window (multi-window)
        # decision no longer uses a hand-set threshold — see decision_config above.
        self.score_thresh_var = tk.DoubleVar(value=0.5)
        
        # Model
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_seq_len = self.seq_len_var.get()
        self.model_n_mels = self.n_mels_var.get()
        self.is_one_shot_model = False

        # Session decision (EWMA-peak score vs a data-driven cutoff; see
        # app/decision/threshold.py and evaluate_decision_rules.py for how the cutoff was
        # chosen). Reloaded per model directory in load_resources().
        self.decision_config = default_decision_config()
        self.decision_accumulator = None

        self._setup_ui()
        self._load_settings()

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
        file_menu.add_command(label="Close", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

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
        style.configure("TLabel", background=base_bg, foreground=text_color, font=("Helvetica", 10))
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
        
        # --- Run Bar (Left): input source + device/file + controls ---
        run_frame = ttk.LabelFrame(left_frame, text="Run", padding=12, style="Card.TLabelframe")
        run_frame.pack(fill="x", padx=10, pady=5)
        run_frame.columnconfigure(1, weight=1)

        ttk.Label(run_frame, text="Input Source:").grid(row=0, column=0, sticky="w", pady=5)
        self.input_type_var = tk.StringVar(value="mic")
        input_frame = ttk.Frame(run_frame)
        input_frame.grid(row=0, column=1, columnspan=2, sticky="w", padx=5)
        ttk.Radiobutton(input_frame, text="Microphone", variable=self.input_type_var, value="mic", command=self.refresh_devices).pack(side="left", padx=5)
        ttk.Radiobutton(input_frame, text="Wav File", variable=self.input_type_var, value="file", command=self.refresh_devices).pack(side="left", padx=5)

        self.device_label = ttk.Label(run_frame, text="Device:")
        self.device_label.grid(row=1, column=0, sticky="w", pady=5)
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(run_frame, textvariable=self.device_var)
        self.device_combo.grid(row=1, column=1, padx=5, sticky="ew")
        self.file_btn = ttk.Button(run_frame, text="Browse", command=self.browse_wav_file)
        self.refresh_devices_btn = ttk.Button(run_frame, text="⟳ Refresh", command=self.refresh_devices)
        self.refresh_devices_btn.grid(row=1, column=3, padx=5)
        self.refresh_devices()

        controls_frame = ttk.Frame(run_frame)
        controls_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.start_btn = ttk.Button(controls_frame, text="START TEST", command=self.toggle_test, width=18, style="Primary.TButton")
        self.start_btn.pack(side="left")
        ttk.Button(controls_frame, text="⚙ Settings", command=self.open_prep_settings).pack(side="left", padx=8)
        ttk.Button(controls_frame, text="Validate Acquisition", command=self.validate_acquisition).pack(side="left", padx=8)

        # --- Decision strip (Left) ---
        # The multi-window (sliding) decision is EWMA-peak score vs a cutoff fit offline
        # from labeled data (see app/decision/threshold.py) — no thresholds to hand-tune
        # here; the active cutoff is shown read-only for transparency and refreshed by
        # load_resources() whenever a model is (re)loaded.
        limits_frame = ttk.LabelFrame(left_frame, text="Decision", padding=8, style="Card.TLabelframe")
        limits_frame.pack(fill="x", padx=10, pady=5)
        self.decision_cutoff_label = ttk.Label(limits_frame, text="Cutoff (auto-calibrated): —")
        self.decision_cutoff_label.pack(side="left", padx=2)

        # --- Dashboard Frame (Left) ---
        dash_frame = ttk.LabelFrame(left_frame, text="Dashboard", padding=12, style="Card.TLabelframe")
        dash_frame.pack(fill="x", padx=10, pady=5)

        # Counters
        count_frame = ttk.Frame(dash_frame)
        count_frame.pack(fill="x", pady=5)

        # EWMA peak
        pos_frame = ttk.Frame(count_frame, padding=5, relief="solid", borderwidth=1)
        pos_frame.pack(side="left", fill="both", expand=True, padx=5)
        ttk.Label(pos_frame, text="EWMA PEAK", font=("Helvetica", 14, "bold"), foreground="red").pack()
        self.pos_label = ttk.Label(pos_frame, text="0.00", font=("Helvetica", 24, "bold"))
        self.pos_label.pack()

        # Live state
        neg_frame = ttk.Frame(count_frame, padding=5, relief="solid", borderwidth=1)
        neg_frame.pack(side="left", fill="both", expand=True, padx=5)
        ttk.Label(neg_frame, text="STATE", font=("Helvetica", 14, "bold"), foreground="green").pack()
        self.neg_label = ttk.Label(neg_frame, text="—", font=("Helvetica", 24, "bold"))
        self.neg_label.pack()

        # Diagnosis
        self.diag_label = ttk.Label(dash_frame, text="No Result Yet", font=("Helvetica", 16, "bold"), foreground="gray")
        self.diag_label.pack(pady=10)

        # Signal Health Indicator (Audio Signal Health Monitoring subsystem)
        self.health_label = ttk.Label(
            dash_frame,
            text="Signal Health: UNKNOWN",
            font=("Helvetica", 12, "bold"),
            foreground="gray",
        )
        self.health_label.pack(pady=4)
        self.health_cause_label = ttk.Label(
            dash_frame,
            text="",
            font=("Helvetica", 10),
            foreground="gray",
        )
        self.health_cause_label.pack(pady=(0, 4))

        # Energy Indicator
        energy_frame = ttk.Frame(dash_frame)
        energy_frame.pack(fill="x", pady=5)
        ttk.Label(energy_frame, text="Current Energy (RMS):").pack(side="left")
        self.energy_bar = ttk.Progressbar(energy_frame, orient="horizontal", length=200, mode="determinate", maximum=1.0)
        self.energy_bar.pack(side="left", padx=10, fill="x", expand=True)

        # Health monitoring profile selector
        profile_frame = ttk.Frame(left_frame)
        profile_frame.pack(fill="x", padx=10, pady=(5, 0))
        ttk.Label(profile_frame, text="Health Profile:").pack(side="left")
        self.profile_combo = ttk.Combobox(
            profile_frame,
            textvariable=self.profile_var,
            values=list(PROFILES),
            state="readonly",
            width=14,
        )
        self.profile_combo.pack(side="left", padx=5)
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_profile_change)
        self.cal_label = ttk.Label(profile_frame, text="Calibration: none")
        self.cal_label.pack(side="left", padx=12)

        # --- Signal Health Detail (per-check breakdown) ---
        health_frame = ttk.LabelFrame(left_frame, text="Signal Health Detail", padding=8, style="Card.TLabelframe")
        health_frame.pack(fill="x", padx=10, pady=5)
        self.health_tree = ttk.Treeview(
            health_frame, columns=("status", "cal", "detail"), show="tree headings", height=11
        )
        self.health_tree.heading("#0", text="Check")
        self.health_tree.heading("status", text="Status")
        self.health_tree.heading("cal", text="Cal")
        self.health_tree.heading("detail", text="Detail")
        self.health_tree.column("#0", width=170, stretch=False)
        self.health_tree.column("status", width=70, anchor="center", stretch=False)
        self.health_tree.column("cal", width=60, anchor="center", stretch=False)
        self.health_tree.column("detail", width=210)
        self.health_tree.tag_configure("PASS", foreground="green")
        self.health_tree.tag_configure("WARNING", foreground="orange")
        self.health_tree.tag_configure("FAIL", foreground="red")
        self.health_tree.tag_configure("NOT_EXECUTED", foreground="gray")
        self.health_tree.tag_configure("CAUSE", font=("Helvetica", 9, "bold"))
        self.health_tree.pack(fill="x")

        # Status Log
        status_frame = ttk.Frame(left_frame)
        status_frame.pack(fill="both", expand=True, padx=10, pady=5)

        status_scrollbar = ttk.Scrollbar(status_frame, orient="vertical")
        status_scrollbar.pack(side="right", fill="y")

        self.status_log = tk.Text(
            status_frame,
            height=15,
            wrap="word",
            yscrollcommand=status_scrollbar.set,
        )
        self.status_log.pack(side="left", fill="both", expand=True)
        self.status_log.configure(state="disabled")
        status_scrollbar.configure(command=self.status_log.yview)
        
        # --- Plots Area (Right): one responsive grid figure (no scroll) ---
        self.plot_fig = Figure(figsize=(7, 6), dpi=100, constrained_layout=True)
        gs = self.plot_fig.add_gridspec(4, 2, height_ratios=[2, 2, 2, 1.3])
        self.ax_wave = self.plot_fig.add_subplot(gs[0, :])      # waveform: full-width top
        self.ax_spec = self.plot_fig.add_subplot(gs[1, 0])      # spectrogram
        self.ax_energy = self.plot_fig.add_subplot(gs[1, 1])    # energy timeline
        self.ax_time = self.plot_fig.add_subplot(gs[2, 0])      # trigger timeline
        self.ax_hist = self.plot_fig.add_subplot(gs[2, 1])      # score distribution
        self.ax_health = self.plot_fig.add_subplot(gs[3, :])    # health timeline: full-width strip
        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig, master=right_frame)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)

        # status_log now exists, so this is the first point where the initial
        # health_pipeline built above (before any UI existed) can be logged.
        self._rebuild_health_pipeline()

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
        SettingsDialog(self).show()

    def _update_model_specific_ui(self):
        # Duration & inference-mode controls now live in the Settings dialog, so we
        # only set the appropriate mode here (a one-shot model forces single-shot).
        if getattr(self, "is_one_shot_model", False):
            self.inference_mode_var.set("single")
        else:
            self.inference_mode_var.set("sliding")

    def refresh_devices(self):
        input_type = self.input_type_var.get()
        if input_type == 'mic':
            self.file_btn.grid_remove()
            self.device_label.configure(text="Device:")
            self.device_combo.grid(row=1, column=1, padx=5, sticky="ew")
            try:
                devices = sd.query_devices()
                input_devices = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]
                self.device_combo['values'] = input_devices
                if input_devices:
                    self.device_combo.current(0)
            except Exception as e:
                self.log(f"Error listing devices: {e}")
        else:
            self.device_label.configure(text="Wav File:")
            self.device_combo.grid(row=1, column=1, padx=5, sticky="ew")
            self.file_btn.grid(row=1, column=2, padx=5)
            
    def display_file_entry(self):
        # Helper to swtich UI for file mode
        pass

    def _discover_models(self):
        """Scan models/<name>/ for a model.tflite + a scaler file.

        Returns {name: (model_path, scaler_path)}, sorted by name. A folder
        without both files is skipped (e.g. an in-progress model drop).
        """
        models_dir = os.path.join(os.getcwd(), "models")
        discovered = {}
        if not os.path.isdir(models_dir):
            return discovered
        for name in sorted(os.listdir(models_dir)):
            folder = os.path.join(models_dir, name)
            model_path = os.path.join(folder, "model.tflite")
            if not os.path.isfile(model_path):
                continue
            scaler_path = None
            for candidate in ("scaler.json", "scaler.npz", "scalar.json"):
                p = os.path.join(folder, candidate)
                if os.path.isfile(p):
                    scaler_path = p
                    break
            if scaler_path is None:
                continue
            discovered[name] = (model_path, scaler_path)
        return discovered

    def on_model_choice_selected(self, event=None):
        name = self.model_choice_var.get()
        entry = self.available_models.get(name)
        if entry is None:
            return
        model_path, scaler_path = entry
        self.model_path_var.set(model_path)
        self.scaler_path_var.set(scaler_path)
        self.log(f"Selected model: {name}")

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
        self.status_log.configure(state="normal")
        self.status_log.insert(tk.END, message + "\n")
        self.status_log.see(tk.END)
        self.status_log.configure(state="disabled")
        
    def load_resources(self):
        # Load Model
        model_path = self.model_path_var.get()
        if not os.path.exists(model_path):
            messagebox.showerror("Error", "Model file not found!")
            return False
            
        if tflite is None:
            messagebox.showerror("Error", "TFLite runtime not installed!")
            return False
            
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            input_shape = self.input_details[0].get("shape", [])
            if len(input_shape) >= 4:
                self.model_seq_len = int(input_shape[1])
                self.model_n_mels = int(input_shape[2])
                self.seq_len_var.set(self.model_seq_len)
                self.n_mels_var.set(self.model_n_mels)
                self.is_one_shot_model = self.model_seq_len >= 784 or "one_shot" in model_path.lower()
            else:
                self.is_one_shot_model = False
            self.log(f"Loaded Model: {os.path.basename(model_path)}")
            self.log(f"Model input shape: {tuple(input_shape)}")
            self._update_model_specific_ui()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            return False
            
        # Load Scaler
        scaler_path = self.scaler_path_var.get()
        if not os.path.exists(scaler_path):
            messagebox.showerror("Error", "Scaler file not found!")
            return False
            
        self.scaler_mean, self.scaler_var = self.processor.load_scaler(scaler_path)
        if self.scaler_mean is None:
            reason = self.processor.last_scaler_error or "unknown error"
            messagebox.showerror("Error", f"Failed to load scaler data:\n{reason}")
            return False
        self.log(f"Loaded Scaler: {os.path.basename(scaler_path)}")

        # Load decision calibration (EWMA-peak cutoff) from next to the model, if present;
        # falls back to the shipped models/9_1_2 default otherwise (a fitted cutoff is
        # specific to the model/scaler pair it was calibrated against).
        threshold_path = os.path.join(os.path.dirname(model_path), "decision_threshold.json")
        self.decision_config = default_decision_config(threshold_path=threshold_path)
        self.log(f"Decision cutoff: {self.decision_config.cutoff:.4f} (span={self.decision_config.span})")

        return True

    def toggle_test(self):
        if self.is_running:
            self.stop_test()
        else:
            if self.load_resources():
                self.start_test()
                
    def start_test(self):
        self.is_running = True
        self.decision_accumulator = EwmaPeakDecision(self.decision_config)
        self.pos_label.configure(text="0.00")
        self.neg_label.configure(text="—")
        self.decision_cutoff_label.configure(text=f"Cutoff (auto-calibrated): {self.decision_config.cutoff:.4f}")
        self.diag_label.configure(text="Testing...", foreground="blue")
        
        # Reset History
        self.score_history = []
        self.trigger_history = []
        self.energy_history = []
        self.health_state_history = []
        self.runtime_monitor = RuntimeMonitor()
        self._last_anomalous = False
        self._last_root_cause = None
        self.start_time = time.time()
        self.session_buffer = np.zeros(int(44100 * self.duration_var.get()), dtype=np.float32)
        self.collected_audio_chunks = []
        self.final_audio_clip = None
        self.single_shot_processed = False
        
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

        # Drain any pending chunks before evaluating final result.
        try:
            while True:
                item = self.audio_queue.get_nowait()
                if isinstance(item, tuple) and item[0] == "ERROR":
                    self.log(f"Error: {item[1]}")
                    break
                self.handle_audio_chunk(item)
        except queue.Empty:
            pass

        # For single-shot mode, run exactly one inference on the full clip.
        if self.inference_mode_var.get() == "single" and not self.single_shot_processed:
            self.run_single_shot_inference()

        self.start_btn.configure(text="START TEST")
        predicted_infested = self.calculate_diagnosis()

        audio_path = None
        if self.save_results_var.get():
            if self.input_type_var.get() == "mic":
                audio_path = self.save_audio_snapshot(predicted_infested)
            self.save_results(predicted_infested, audio_path)
        self.log("Test Stopped.")
        
    def calculate_diagnosis(self):
        if self.inference_mode_var.get() == "single":
            if not self.trigger_history:
                self.diag_label.configure(text="NO RESULT", foreground="gray")
                return False
            predicted = self.trigger_history[-1][1]
            score = self.score_history[-1] if self.score_history else 0.0
            if predicted == 1:
                self.diag_label.configure(text=f"INFESTED (Score: {score:.2f})", foreground="red")
                return True
            self.diag_label.configure(text=f"HEALTHY (Score: {score:.2f})", foreground="green")
            return False

        if self.decision_accumulator is None:
            self.diag_label.configure(text="NO RESULT", foreground="gray")
            return False

        peak = self.decision_accumulator.peak
        state = self.decision_accumulator.state
        predicted_infested = state == "INFESTED"
        color = "red" if predicted_infested else "green"
        self.diag_label.configure(text=f"{state} (EWMA peak: {peak:.3f})", foreground=color)
        return predicted_infested

    def mic_loop(self, device_idx):
        self.log(f"Starting Mic Stream on Device {device_idx}...")
        
        sample_rate = 44100
        
        block_size = int(sample_rate * 0.5) # 0.5 sec blocks

        duration_sec = mic_capture_duration_sec(
            self.inference_mode_var.get(), self.single_shot_duration_sec, self.sliding_test_duration_var.get()
        )
        max_chunks = int(duration_sec / 0.5)
        mode_label = "Single-shot" if self.inference_mode_var.get() == "single" else "Sliding-window"
        self.log(f"{mode_label} mode: capturing {duration_sec}s")
        chunk_count = [0]
        
        def callback(indata, frames, time, status):
            self.audio_queue.put(indata.copy())
            chunk_count[0] += 1
            
        try:
            with sd.InputStream(device=device_idx, channels=1, samplerate=sample_rate, 
                                blocksize=block_size, callback=callback):
                while self.is_running:
                    if max_chunks is not None and chunk_count[0] >= max_chunks:
                        self.is_running = False
                        break
                    time.sleep(0.1)
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
            if self.inference_mode_var.get() == "single":
                total_samples = min(total_samples, int(fs * self.single_shot_duration_sec))
            
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
        
    def _settings_spec(self):
        """(key, tk Variable) pairs that can be persisted with a plain .get()/.set() —
        i.e. no side effects needed on restore. Settings that trigger follow-up work
        (device list, model paths, health pipeline, calibration profile) are handled
        separately in _save_settings/_load_settings."""
        return [
            ("use_filter", self.use_filter_var),
            ("low_cut", self.low_cut_var),
            ("up_cut", self.up_cut_var),
            ("sub_win_size", self.sub_win_size_var),
            ("fmin", self.fmin_var),
            ("fmax", self.fmax_var),
            ("sub_hop_size", self.sub_hop_size_var),
            ("n_mels", self.n_mels_var),
            ("seq_len", self.seq_len_var),
            ("inference_mode", self.inference_mode_var),
            ("save_results", self.save_results_var),
            ("user_label", self.user_label_var),
            ("output_dir", self.output_dir_var),
            ("duration", self.duration_var),
            ("score_thresh", self.score_thresh_var),
            ("sliding_test_duration", self.sliding_test_duration_var),
        ]

    def _save_settings(self):
        data = {key: var.get() for key, var in self._settings_spec()}
        data["input_type"] = self.input_type_var.get()
        data["device"] = self.device_var.get()
        data["model_choice"] = self.model_choice_var.get()
        data["health_profile"] = self.profile_var.get()
        data["calibration_profile_path"] = self.calibration_profile_path_var.get()
        data["window_geometry"] = self.root.geometry()
        try:
            with open(USER_SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.log(f"Failed to save settings: {e}")

    def _load_settings(self):
        if not os.path.isfile(USER_SETTINGS_PATH):
            return
        try:
            with open(USER_SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.log(f"Failed to load saved settings: {e}")
            return

        for key, var in self._settings_spec():
            if key in data:
                try:
                    var.set(data[key])
                except Exception:
                    pass

        try:
            input_type = data.get("input_type")
            if input_type:
                self.input_type_var.set(input_type)
            # refresh_devices() resets device_var to index 0 in mic mode, so the
            # saved device must be applied after it, not before.
            self.refresh_devices()
            device = data.get("device")
            if device:
                self.device_var.set(device)
        except Exception as e:
            self.log(f"Failed to restore saved input device: {e}")

        try:
            model_choice = data.get("model_choice")
            if model_choice and model_choice in self.available_models:
                self.model_choice_var.set(model_choice)
                self.on_model_choice_selected()
        except Exception as e:
            self.log(f"Failed to restore saved model choice: {e}")

        try:
            health_profile = data.get("health_profile")
            if health_profile and health_profile in PROFILES:
                self.profile_var.set(health_profile)
                self._rebuild_health_pipeline()
        except Exception as e:
            self.log(f"Failed to restore saved health profile: {e}")

        try:
            cal_path = data.get("calibration_profile_path")
            if cal_path and os.path.isfile(cal_path):
                self._load_calibration_profile(cal_path)
        except Exception as e:
            self.log(f"Failed to restore saved calibration profile: {e}")

        geometry = data.get("window_geometry")
        if geometry:
            try:
                self.root.geometry(geometry)
            except Exception:
                pass

    def _on_close(self):
        self._save_settings()
        self.root.destroy()

    def _rebuild_health_pipeline(self):
        self.health_pipeline = pipeline_for_profile(
            self.profile_var.get(), calibration_profile=self.calibration_profile
        )
        self._last_health_state = None
        if os.path.isfile(DEFAULT_CHECK_THRESHOLDS_PATH):
            self.log(f"Health check thresholds: loaded {DEFAULT_CHECK_THRESHOLDS_PATH}")
        else:
            self.log("Health check thresholds: shipped defaults (no check_thresholds.json found)")

    def _load_calibration_profile(self, path):
        from app.health.calibration import load_profile

        try:
            self.calibration_profile = load_profile(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration profile:\n{e}")
            return
        self.calibration_profile_path_var.set(path)
        self._rebuild_health_pipeline()
        p = self.calibration_profile
        self.cal_label.configure(text=f"Calibration: {p.profile_id} ({p.window_count} win)")
        self.log(f"Loaded calibration profile '{p.profile_id}' ({p.window_count} windows)")

    def generate_calibration_profile(self):
        import calibrate

        folder = filedialog.askdirectory(title="Select folder of HEALTHY recordings")
        if not folder:
            return
        out = filedialog.asksaveasfilename(
            title="Save calibration profile", defaultextension=".json",
            filetypes=[("Calibration profile", "*.json")],
        )
        if not out:
            return
        profile_id = simpledialog.askstring(
            "Calibration profile id", "Profile id:", initialvalue="piezo"
        )
        if not profile_id:
            return
        self.log(f"Generating calibration profile from {folder} ...")
        self.root.update_idletasks()
        try:
            profile = calibrate.run(folder, out, profile_id=profile_id)
        except Exception as e:
            messagebox.showerror("Error", f"Calibration failed:\n{e}")
            return
        self.log(f"Calibration profile saved: {out} ({profile.window_count} windows)")
        self._load_calibration_profile(out)

    def browse_calibration_profile(self):
        path = filedialog.askopenfilename(
            title="Select calibration profile", filetypes=[("Calibration profile", "*.json")]
        )
        if path:
            self._load_calibration_profile(path)

    def _on_profile_change(self, event=None):
        self._rebuild_health_pipeline()
        profile = self.profile_var.get()
        count = len(self.health_pipeline.manager.checks)
        self.log(f"Health profile: {profile} ({count} checks active)")

    def _update_health_panel(self, report, assessment=None):
        tree = self.health_tree
        tree.delete(*tree.get_children())
        if assessment is not None and assessment.primary_cause is not rootcause.RootCause.NONE:
            check_id, name, status, detail, cal = root_cause_row(assessment)
            tree.insert(
                "", "end", text=f"{check_id}  {name}",
                values=(status, cal, detail), tags=("CAUSE",)
            )
        for check_id, name, status, detail, cal in report_rows(report):
            tree.insert(
                "", "end", text=f"{check_id}  {name}",
                values=(status, cal, detail), tags=(status,)
            )

    def validate_acquisition(self):
        if not self.is_running:
            messagebox.showinfo(
                "Validate Acquisition",
                "Start a test first, then click Validate to assess ~20 s of the live signal.",
            )
            return
        self._validation_reports = []
        self._validating = True
        self.log("Validating acquisition over the next ~20 s ...")

    def _write_report(self, kind, payload):
        """Persist a health report dict to reports/<kind>_<timestamp>.json.

        The actual file I/O runs on a daemon thread so it never blocks the audio
        callback thread that triggers it (best-effort; failures are logged only).
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        def _do_write():
            try:
                os.makedirs("reports", exist_ok=True)
                path = os.path.join("reports", f"{kind}_{ts}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                self.log(f"[report] wrote {path}")
            except Exception as e:
                self.log(f"[report] write error: {e}")

        threading.Thread(target=_do_write, daemon=True).start()

    def _show_validation_result(self):
        result = run_validation(
            self._validation_reports,
            sample_rate=SAMPLE_RATE,
            input_ready=True,
            calibration_loaded=self.calibration_profile is not None,
        )
        self.log(f"[validation] {result.summary}")
        assessment = rootcause.assess_many(self._validation_reports)
        payload = startup_result_to_dict(result)
        payload["root_cause"] = root_cause_to_dict(assessment)
        self._write_report("startup", payload)
        show = {
            StartupDecision.PASS: messagebox.showinfo,
            StartupDecision.WARNING: messagebox.showwarning,
            StartupDecision.FAIL: messagebox.showerror,
        }[result.decision]
        show(
            "Acquisition Validation",
            f"{result.decision.value}\n\n{result.summary}\n\n"
            f"Likely cause: {assessment.primary_cause.value}\n{assessment.explanation}",
        )

    def _update_health_indicator(self, report):
        colors = {
            HealthState.OK: "green",
            HealthState.WARNING: "orange",
            HealthState.FAULT: "red",
            HealthState.UNKNOWN: "gray",
        }
        # The monitor debounces the raw per-window verdict into a stable state.
        events = self.runtime_monitor.update(report)
        state = self.runtime_monitor.runtime_state
        level = {HealthState.OK: 0, HealthState.WARNING: 1, HealthState.FAULT: 2}.get(state, 0)
        self.health_state_history.append((time.time() - self.start_time, level))
        self.health_label.configure(
            text=f"Signal Health: {state.value} · conf {report.confidence:.2f}",
            foreground=colors.get(state, "gray"),
        )

        # Root-cause attribution — only computed while unhealthy (state != OK);
        # cheap rule-based scoring over already-computed check results, never
        # blocks or alters inference.
        assessment = None
        if state is not HealthState.OK:
            assessment = rootcause.assess(report)
        if assessment is not None and assessment.primary_cause is not rootcause.RootCause.NONE:
            self.health_cause_label.configure(
                text=f"Likely cause: {assessment.primary_cause.value} — {assessment.explanation}",
                foreground=colors.get(state, "gray"),
            )
        else:
            # Also covers the debounce-lag window where runtime_state is still
            # WARNING/FAULT (RuntimeMonitor holding a recovery streak) but the
            # raw report has already recovered to NONE -- avoid showing a
            # "Likely cause: NONE" line that contradicts health_label above.
            self.health_cause_label.configure(text="", foreground="gray")

        for event in events:
            self.log(f"[monitor] {event.message}: {report.diagnostic_summary}")
        # Log the likely cause only on change (rising-edge-on-change), mirroring
        # the anomaly rising-edge pattern below — avoids spamming the log every
        # 0.5s while a persistent fault is showing.
        if assessment is not None and assessment.primary_cause is not self._last_root_cause:
            self.log(f"[likely-cause] {assessment.primary_cause.value}: {assessment.explanation}")
        self._last_root_cause = assessment.primary_cause if assessment is not None else None

        # Log an anomaly only on its rising edge (anomaly_result is None without a profile).
        anomaly = report.anomaly_result
        anomalous = anomaly is not None and anomaly.is_anomalous
        if anomalous and not self._last_anomalous:
            top_label, top_c = anomaly.contributors[0] if anomaly.contributors else ("", 0.0)
            self.log(f"[anomaly] distance {anomaly.distance:.1f} — {top_label} contrib={top_c:.1f}")
            source = "wav" if self.input_type_var.get() == "file" else "mic"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._write_report("anomaly", anomaly_event_to_dict(anomaly, source=source, timestamp=ts))
        self._last_anomalous = anomalous
        self._update_health_panel(report, assessment)

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

        # Audio signal health monitoring — additive; never blocks or alters inference.
        try:
            window = AudioWindow(samples=self.session_buffer, sample_rate=SAMPLE_RATE)
            self.latest_health_report = self.health_pipeline.analyze(window)
            self._update_health_indicator(self.latest_health_report)
            if self._validating and self.latest_health_report is not None:
                self._validation_reports.append(self.latest_health_report)
                if len(self._validation_reports) >= VALIDATION_WINDOWS:
                    self._validating = False
                    self._show_validation_result()
        except Exception as e:
            self.log(f"Health monitoring error: {e}")

        if self.inference_mode_var.get() == "single":
            self.collected_audio_chunks.append(chunk_flat.copy())
            # Keep energy timeline live while recording
            rms = np.sqrt(np.mean(chunk_flat**2))
            self.current_energy = rms
            self.energy_bar["value"] = min(rms * 100, 100)
            curr_time = time.time() - self.start_time
            self.energy_history.append((curr_time, rms))
        else:
            # Sliding-window behavior
            self.run_inference(self.session_buffer)

    def run_single_shot_inference(self):
        if not self.collected_audio_chunks:
            self.log("Single-shot inference skipped: no audio captured")
            self.single_shot_processed = True
            return

        full_audio = np.concatenate(self.collected_audio_chunks).astype(np.float32)
        target_samples = int(44100 * self.single_shot_duration_sec)
        if len(full_audio) < target_samples:
            full_audio = np.pad(full_audio, (0, target_samples - len(full_audio)))
        else:
            full_audio = full_audio[:target_samples]

        self.final_audio_clip = full_audio.copy()
        self.raw_audio_snapshot = self.final_audio_clip.copy()
        self.run_inference(self.final_audio_clip)
        self.single_shot_processed = True

    def _derive_output_base(self, predicted_infested):
        user_choice = self.user_label_var.get()
        predicted_infested = bool(predicted_infested)
        if user_choice == "infested":
            tag = "TP" if predicted_infested else "FN"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else: # healthy
            tag = "TN" if not predicted_infested else "FP"
            timestamp = datetime.now().strftime("%Y%m%d_%I%M%p") if tag == "TN" else datetime.now().strftime("%Y%m%d_%H%M%S")

        out_dir = self.output_dir_var.get().strip() or "recordings"
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.join(out_dir, f"{tag}_{timestamp}")
        return base

    def save_audio_snapshot(self, predicted_positive):
        if not hasattr(self, 'session_buffer') and self.final_audio_clip is None:
            return None
        try:
            base = self._derive_output_base(predicted_positive)
            wav_path = f"{base}.wav"
            audio_to_save = self.final_audio_clip if self.inference_mode_var.get() == "single" and self.final_audio_clip is not None else self.session_buffer
            sf.write(wav_path, audio_to_save, 44100)
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
                "score_thresh": self.score_thresh_var.get(),  # single-shot mode only
                "decision_method": "ewma_peak",
                "decision_cutoff": self.decision_config.cutoff,
                "decision_span": self.decision_config.span,
                "decision_final_state": self.decision_accumulator.state if self.decision_accumulator else None,
                "decision_ewma_peak": self.decision_accumulator.peak if self.decision_accumulator else None,
                "n_mels": self.n_mels_var.get(),
                "seq_len": self.seq_len_var.get(),
                "low_cut": self.low_cut_var.get(),
                "up_cut": self.up_cut_var.get(),
                "fmin": self.fmin_var.get(),
                "fmax": self.fmax_var.get(),
                "sub_win_size": self.sub_win_size_var.get(),
                "sub_hop_size": self.sub_hop_size_var.get(),
                "save_results_enabled": self.save_results_var.get(),
                "input_type": self.input_type_var.get(),
                "inference_mode": self.inference_mode_var.get(),
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
            self.energy_bar["value"] = min(rms * 100, 100) # Scaling for visibility, assuming max ~1.0
            
            # 1. Extract Features with Config
            specs = self.processor.extract_features(
                audio_data,
                sr=44100,
                n_mels=self.model_n_mels,
                seq_len=self.model_seq_len,
                low_cut=self.low_cut_var.get(),
                up_cut=self.up_cut_var.get(),
                fmin=self.fmin_var.get(),
                fmax=self.fmax_var.get(),
                sub_win_size_sec=self.sub_win_size_var.get(),
                sub_hop_size_sec=self.sub_hop_size_var.get(),
                use_filter=self.use_filter_var.get()
            )
            
            # Save for plotting (Mel Spectrogram in dB)
            self.current_spectrogram = specs.copy()
            
            # 2. Scale
            specs_scaled = self.processor.apply_scaler(specs, self.scaler_mean, self.scaler_var)
            
            # 3. Reshape for Model
            # Model Input: (1, seq_len, n_mels, 1)
            seq_len = self.model_seq_len
            n_mels = self.model_n_mels
            input_data = specs_scaled.reshape(1, seq_len, n_mels, 1).astype(np.float32)
            
            # 4. Invoke
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # 5. Logic
            # score_thresh only drives the display-only per-window marker below (and
            # single-shot mode's final call) — the sliding-window session decision is the
            # EWMA-peak accumulator updated further down, not a per-window threshold.
            score_thresh = self.score_thresh_var.get()

            score = 0
            if output_data.shape[-1] == 1:
                score = output_data[0][0]
            else:
                # Softmax - usually binary still, but if index 1 is positive
                score = output_data[0][1] # P(Positive)
            predicted = 1 if score > score_thresh else 0

            # Store stats for plot
            self.score_history.append(score)
            curr_time = time.time() - self.start_time
            self.trigger_history.append((curr_time, predicted))
            self.energy_history.append((curr_time, rms))

            # Update session decision (EWMA-peak vs the auto-calibrated cutoff)
            if self.decision_accumulator is not None:
                self.decision_accumulator.update(float(score))
                self.pos_label.configure(text=f"{self.decision_accumulator.peak:.2f}")
                self.neg_label.configure(text=self.decision_accumulator.state)

            res_str = f"POS ({score:.2f})" if predicted == 1 else f"NEG ({score:.2f})"
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
                self.ax_wave.set_title("Audio Waveform", fontsize=9)
                self.ax_wave.set_ylim(-1, 1)
                self.ax_wave.tick_params(labelsize=7)
                self.ax_wave.grid(True)

            # Update Histogram (Score)
            if self.score_history:
                self.ax_hist.clear()
                self.ax_hist.hist(self.score_history, bins=10, range=(0,1), color='skyblue', edgecolor='black')
                self.ax_hist.set_title("Score Distribution", fontsize=9)
                self.ax_hist.set_xlim(0, 1)
                self.ax_hist.tick_params(labelsize=7)
                self.ax_hist.grid(True)

            # Update Spectrogram (Heatmap)
            if self.current_spectrogram is not None:
                self.ax_spec.clear()
                # Transpose to have Freq on Y, Time on X
                # specs is (98, 32) -> .T is (32, 98)
                self.ax_spec.imshow(self.current_spectrogram.T, aspect='auto', origin='lower', cmap='inferno')
                self.ax_spec.set_title("Mel Spectrogram", fontsize=9)
                self.ax_spec.set_ylabel("Mel Bands", fontsize=8)
                self.ax_spec.set_xlabel("Time Frames", fontsize=8)
                self.ax_spec.tick_params(labelsize=7)

            # Update Energy
            if self.energy_history:
                times, energy = zip(*self.energy_history)
                self.ax_energy.clear()
                self.ax_energy.plot(times, energy, 'g-')
                self.ax_energy.set_title("Energy Timeline (RMS)", fontsize=9)
                self.ax_energy.set_ylim(0, max(0.1, max(energy) * 1.2)) # Dynamic limit
                self.ax_energy.tick_params(labelsize=7)
                self.ax_energy.grid(True)

            # Update Timeline
            if self.trigger_history:
                times, triggers = zip(*self.trigger_history)
                self.ax_time.clear()
                self.ax_time.plot(times, triggers, 'r-o', markersize=4)
                self.ax_time.set_title("Trigger Timeline (1=Pos, 0=Neg)", fontsize=9)
                self.ax_time.set_ylim(-0.1, 1.1)
                self.ax_time.set_xlabel("Time (s)", fontsize=8)
                self.ax_time.tick_params(labelsize=7)
                self.ax_time.grid(True)

            # Update Health Timeline (debounced runtime state over time)
            if self.health_state_history:
                times, levels = zip(*self.health_state_history)
                self.ax_health.clear()
                self.ax_health.step(times, levels, where="post", color="purple")
                self.ax_health.set_title("Health Timeline", fontsize=9)
                self.ax_health.set_ylim(-0.2, 2.2)
                self.ax_health.set_yticks([0, 1, 2])
                self.ax_health.set_yticklabels(["OK", "WARN", "FAULT"], fontsize=7)
                self.ax_health.set_xlabel("Time (s)", fontsize=8)
                self.ax_health.tick_params(labelsize=7)
                self.ax_health.grid(True)

            self.plot_canvas.draw_idle()
        except Exception as e:
            self.log(f"Plot Error: {e}")

        if self.is_running:
            self.root.after(1000, self.update_plots)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelsTesterApp(root)
    root.mainloop()
