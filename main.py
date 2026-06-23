import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from inference_utils import AudioProcessor
from app.health.pipeline import HealthAnalysisPipeline
from app.health.models import AudioWindow, HealthState

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
        self.health_pipeline = HealthAnalysisPipeline()
        self.latest_health_report = None
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
        self.inference_mode_var = tk.StringVar(value="sliding")
        self.single_shot_duration_sec = 20
        self.collected_audio_chunks = []
        self.final_audio_clip = None
        self.single_shot_processed = False
        
        # History Data for Plots
        self.score_history = []
        self.trigger_history = [] # (time, is_trigger)
        self.energy_history = []  # (time, rms)
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

        # Saving options
        self.save_results_var = tk.BooleanVar(value=False) # single toggle controls both wav and json
        self.save_audio_var = self.save_results_var
        self.user_label_var = tk.StringVar(value="infested") # User expectation
        self.output_dir_var = tk.StringVar(value=self.default_output_dir)
        
        # Model
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_seq_len = self.seq_len_var.get()
        self.model_n_mels = self.n_mels_var.get()
        self.is_one_shot_model = False

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
        
        # --- Configuration Frame (Left) ---
        config_frame = ttk.LabelFrame(left_frame, text="Configuration", padding=12, style="Card.TLabelframe")
        config_frame.pack(fill="x", padx=10, pady=5)
        # Allow entries to expand horizontally so browse buttons stay visible on narrower panes
        config_frame.columnconfigure(1, weight=1)
        
        # Model Path
        ttk.Label(config_frame, text="Model Path (.tflite):").grid(row=0, column=0, sticky="w")
        self.model_path_var = tk.StringVar(value=self.default_model_path)
        ttk.Entry(config_frame, textvariable=self.model_path_var).grid(row=0, column=1, padx=5, sticky="ew")
        ttk.Button(config_frame, text="Browse", command=self.load_model_dialog).grid(row=0, column=2)

        # Scaler Path
        ttk.Label(config_frame, text="Scaler Path (.json/.npz):").grid(row=1, column=0, sticky="w", pady=5)
        self.scaler_path_var = tk.StringVar(value=self.default_scaler_path)
        ttk.Entry(config_frame, textvariable=self.scaler_path_var).grid(row=1, column=1, padx=5, sticky="ew")
        ttk.Button(config_frame, text="Browse", command=self.load_scaler_dialog).grid(row=1, column=2)

        # Duration (used for sliding-window mode)
        self.duration_row = ttk.Frame(config_frame)
        self.duration_row.grid(row=2, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Label(self.duration_row, text="Window Duration (sec):").pack(side="left")
        self.duration_var = tk.DoubleVar(value=2.5)
        ttk.Spinbox(self.duration_row, from_=0.5, to=10.0, increment=0.5, textvariable=self.duration_var, width=10).pack(side="left", padx=5)

        # Input Check
        ttk.Label(config_frame, text="Input Source:").grid(row=3, column=0, sticky="w", pady=5)
        self.input_type_var = tk.StringVar(value="mic")

        input_frame = ttk.Frame(config_frame)
        input_frame.grid(row=3, column=1, sticky="w", padx=5)
        ttk.Radiobutton(input_frame, text="Microphone", variable=self.input_type_var, value="mic", command=self.refresh_devices).pack(side="left", padx=5)
        ttk.Radiobutton(input_frame, text="Wav File", variable=self.input_type_var, value="file", command=self.refresh_devices).pack(side="left", padx=5)

        # Audio Device / File Path (moved down)
        self.device_label = ttk.Label(config_frame, text="Device:")
        self.device_label.grid(row=4, column=0, sticky="w", pady=5)
        
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(config_frame, textvariable=self.device_var)
        self.device_combo.grid(row=4, column=1, padx=5, sticky="ew")
        
        self.file_btn = ttk.Button(config_frame, text="Browse", command=self.browse_wav_file)
        
        self.refresh_devices()

        # Output directory for saved audio
        ttk.Label(config_frame, text="Output Dir:").grid(row=5, column=0, sticky="w", pady=5)
        ttk.Entry(config_frame, textvariable=self.output_dir_var).grid(row=5, column=1, padx=5, sticky="ew")
        ttk.Button(config_frame, text="Browse", command=self.browse_output_dir).grid(row=5, column=2)

        # Limits Config
        limits_frame = ttk.LabelFrame(config_frame, text="Thresholds", padding=8, style="Card.TLabelframe")
        limits_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=5)

        ttk.Label(limits_frame, text="Score Thresh:").pack(side="left", padx=2)
        self.score_thresh_var = tk.DoubleVar(value=0.5)
        ttk.Entry(limits_frame, textvariable=self.score_thresh_var, width=5).pack(side="left", padx=2)

        ttk.Label(limits_frame, text="Suspicious >=").pack(side="left", padx=2)
        self.susp_limit_var = tk.IntVar(value=17)
        ttk.Entry(limits_frame, textvariable=self.susp_limit_var, width=5).pack(side="left", padx=2)

        ttk.Label(limits_frame, text="Infested >").pack(side="left", padx=2)
        self.inf_limit_var = tk.IntVar(value=27)
        ttk.Entry(limits_frame, textvariable=self.inf_limit_var, width=5).pack(side="left", padx=2)
        
        # User label (expected outcome)
        ttk.Label(config_frame, text="User Label:").grid(row=7, column=0, sticky="w", pady=5)
        label_frame = ttk.Frame(config_frame)
        label_frame.grid(row=7, column=1, sticky="w", padx=5)
        ttk.Radiobutton(label_frame, text="Infested", variable=self.user_label_var, value="infested").pack(side="left", padx=5)
        ttk.Radiobutton(label_frame, text="Healthy", variable=self.user_label_var, value="healthy").pack(side="left", padx=5)

        # Save results (WAV + JSON)
        ttk.Checkbutton(config_frame, text="Save results and audio", variable=self.save_results_var).grid(row=8, column=0, columnspan=3, sticky="w", pady=5)

        # Inference behavior selection
        self.mode_row = ttk.Frame(config_frame)
        self.mode_row.grid(row=9, column=0, columnspan=3, sticky="w", pady=5)
        ttk.Label(self.mode_row, text="Inference Mode:").pack(side="left")
        mode_frame = ttk.Frame(self.mode_row)
        mode_frame.pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Sliding Window", variable=self.inference_mode_var, value="sliding").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text=f"Single Shot ({self.single_shot_duration_sec}s)", variable=self.inference_mode_var, value="single").pack(side="left", padx=5)
        
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

        # Signal Health Indicator (Audio Signal Health Monitoring subsystem)
        self.health_label = ttk.Label(
            dash_frame,
            text="Signal Health: UNKNOWN",
            font=("Helvetica", 12, "bold"),
            foreground="gray",
        )
        self.health_label.pack(pady=4)

        # Energy Indicator
        energy_frame = ttk.Frame(dash_frame)
        energy_frame.pack(fill="x", pady=5)
        ttk.Label(energy_frame, text="Current Energy (RMS):").pack(side="left")
        self.energy_bar = ttk.Progressbar(energy_frame, orient="horizontal", length=200, mode="determinate", maximum=1.0)
        self.energy_bar.pack(side="left", padx=10, fill="x", expand=True)
        
        # --- Control Area (Left) ---
        ctrl_frame = ttk.Frame(left_frame, padding=10)
        ctrl_frame.pack(fill="x")
        
        self.start_btn = ttk.Button(ctrl_frame, text="START TEST", command=self.toggle_test, width=20, style="Primary.TButton")
        self.start_btn.pack(pady=5)

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
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Preprocessing Pipeline Configuration")
        settings_win.geometry("400x300")
        settings_win.transient(self.root)
        settings_win.grab_set()

        frame = ttk.Frame(settings_win, padding=20)
        frame.pack(fill="both", expand=True)

        ttk.Checkbutton(frame, text="Use Bandpass Filter", variable=self.use_filter_var).grid(row=0, column=0, columnspan=2, sticky="w", pady=5)

        row = 1
        if not self.is_one_shot_model:
            ttk.Label(frame, text="Duration (sec):").grid(row=row, column=0, sticky="w", pady=5)
            ttk.Spinbox(frame, from_=0.5, to=10.0, increment=0.5, textvariable=self.duration_var, width=15).grid(row=row, column=1, padx=5)
            row += 1

        ttk.Label(frame, text="Low Cut (Hz):").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.low_cut_var, width=15).grid(row=row, column=1, padx=5)
        row += 1

        ttk.Label(frame, text="Up Cut (Hz):").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.up_cut_var, width=15).grid(row=row, column=1, padx=5)
        row += 1

        ttk.Label(frame, text="FFT Win Size (s):").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.sub_win_size_var, width=15).grid(row=row, column=1, padx=5)
        row += 1

        ttk.Label(frame, text="Hop Size (s):").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.sub_hop_size_var, width=15).grid(row=row, column=1, padx=5)
        row += 1

        ttk.Label(frame, text="Mel Bands:").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.n_mels_var, width=15).grid(row=row, column=1, padx=5)
        row += 1

        if not self.is_one_shot_model:
            ttk.Label(frame, text="Seq Len (frames):").grid(row=row, column=0, sticky="w", pady=5)
            ttk.Entry(frame, textvariable=self.seq_len_var, width=15).grid(row=row, column=1, padx=5)
            row += 1

        ttk.Button(frame, text="Close", command=settings_win.destroy).grid(row=row, column=0, columnspan=2, pady=20)

    def _update_model_specific_ui(self):
        if getattr(self, "is_one_shot_model", False):
            self.inference_mode_var.set("single")
            self.duration_row.grid_remove()
            self.mode_row.grid_remove()
        else:
            self.inference_mode_var.set("sliding")
            self.duration_row.grid()
            self.mode_row.grid()

    def refresh_devices(self):
        input_type = self.input_type_var.get()
        if input_type == 'mic':
            self.file_btn.grid_remove()
            self.device_combo.grid(row=4, column=1, padx=5, sticky="w")
            try:
                devices = sd.query_devices()
                input_devices = [f"{i}: {d['name']}" for i, d in enumerate(devices) if d['max_input_channels'] > 0]
                self.device_combo['values'] = input_devices
                if input_devices:
                    self.device_combo.current(0)
            except Exception as e:
                self.log(f"Error listing devices: {e}")
        else:
            self.device_combo.grid_remove()
            self.file_btn.grid(row=4, column=2)
            self.device_label.configure(text="Wav File:")
            self.device_combo.grid(row=4, column=1, padx=5, sticky="w") # reuse combo as entry for file
            
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

        pos = self.positive_count
        susp_limit = self.susp_limit_var.get()
        inf_limit = self.inf_limit_var.get()
        predicted_infested = False
        
        if pos < susp_limit:
            self.diag_label.configure(text=f"HEALTHY (Pos: {pos})", foreground="green")
        elif pos <= inf_limit:
            self.diag_label.configure(text=f"SUSPICIOUS (Pos: {pos})", foreground="orange")
        else:
            self.diag_label.configure(text=f"INFESTED (Pos: {pos})", foreground="red")
            predicted_infested = True
        return predicted_infested

    def mic_loop(self, device_idx):
        self.log(f"Starting Mic Stream on Device {device_idx}...")
        
        sample_rate = 44100
        
        block_size = int(sample_rate * 0.5) # 0.5 sec blocks

        max_chunks = None
        if self.inference_mode_var.get() == "single":
            max_chunks = int(self.single_shot_duration_sec / 0.5)
            self.log(f"Single-shot mode: capturing {self.single_shot_duration_sec}s")
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
        
    def _update_health_indicator(self, report):
        colors = {
            HealthState.OK: "green",
            HealthState.WARNING: "orange",
            HealthState.FAULT: "red",
            HealthState.UNKNOWN: "gray",
        }
        state = report.final_state
        self.health_label.configure(
            text=f"Signal Health: {state.value}",
            foreground=colors.get(state, "gray"),
        )

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
            window = AudioWindow(samples=self.session_buffer, sample_rate=44100)
            self.latest_health_report = self.health_pipeline.analyze(window)
            self._update_health_indicator(self.latest_health_report)
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
            score_thresh = self.score_thresh_var.get()
            
            predicted = 0
            score = 0
            if output_data.shape[-1] == 1:
                score = output_data[0][0]
                predicted = 1 if score > score_thresh else 0
            else:
                # Softmax - usually binary still, but if index 1 is positive
                score = output_data[0][1] # P(Positive)
                predicted = 1 if score > score_thresh else 0

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
