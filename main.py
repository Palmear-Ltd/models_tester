import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
import time
import os
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from inference_utils import AudioProcessor

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
        self.root.geometry("2000x1200")
        
        self.processor = AudioProcessor()
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
        
        # Model
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        self._setup_ui()
        
    def _setup_ui(self):
        # Styles
        style = ttk.Style()
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("TButton", font=("Helvetica", 10))
        style.configure("Header.TLabel", font=("Helvetica", 12, "bold"))
        
        # Main Layout: Left (Controls) | Right (Plots)
        main_pane = ttk.PanedWindow(self.root, orient="horizontal")
        main_pane.pack(fill="both", expand=True)
        
        left_frame = ttk.Frame(main_pane, padding=10)
        right_frame = ttk.Frame(main_pane, padding=10) # For Plots
        main_pane.add(left_frame, weight=1)
        main_pane.add(right_frame, weight=3)
        
        # --- Configuration Frame (Left) ---
        config_frame = ttk.LabelFrame(left_frame, text="Configuration", padding=10)
        config_frame.pack(fill="x", padx=10, pady=5)
        
        # Model Path
        ttk.Label(config_frame, text="Model Path (.tflite):").grid(row=0, column=0, sticky="w")
        self.model_path_var = tk.StringVar()
        ttk.Entry(config_frame, textvariable=self.model_path_var, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(config_frame, text="Browse", command=self.load_model_dialog).grid(row=0, column=2)
        
        # Scaler Path
        ttk.Label(config_frame, text="Scaler Path (.npz):").grid(row=1, column=0, sticky="w", pady=5)
        self.scaler_path_var = tk.StringVar(value="scaler.npz")
        ttk.Entry(config_frame, textvariable=self.scaler_path_var, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(config_frame, text="Browse", command=self.load_scaler_dialog).grid(row=1, column=2)
        
        # Duration
        ttk.Label(config_frame, text="Duration (sec):").grid(row=2, column=0, sticky="w", pady=5)
        self.duration_var = tk.DoubleVar(value=2.5)
        ttk.Spinbox(config_frame, from_=0.5, to=10.0, increment=0.5, textvariable=self.duration_var, width=10).grid(row=2, column=1, sticky="w", padx=5)
        
        # Input Check
        ttk.Label(config_frame, text="Input Source:").grid(row=3, column=0, sticky="w", pady=5)
        self.input_type_var = tk.StringVar(value="mic")
        
        input_frame = ttk.Frame(config_frame)
        input_frame.grid(row=3, column=1, sticky="w", padx=5)
        ttk.Radiobutton(input_frame, text="Microphone", variable=self.input_type_var, value="mic", command=self.refresh_devices).pack(side="left", padx=5)
        ttk.Radiobutton(input_frame, text="Wav File", variable=self.input_type_var, value="file", command=self.refresh_devices).pack(side="left", padx=5)

        # Limits Config
        limits_frame = ttk.LabelFrame(config_frame, text="Thresholds", padding=5)
        limits_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=5)

        ttk.Label(limits_frame, text="Score Thresh:").pack(side="left", padx=2)
        self.score_thresh_var = tk.DoubleVar(value=0.5)
        ttk.Entry(limits_frame, textvariable=self.score_thresh_var, width=5).pack(side="left", padx=2)

        ttk.Label(limits_frame, text="Suspicious >=").pack(side="left", padx=2)
        self.susp_limit_var = tk.IntVar(value=17)
        ttk.Entry(limits_frame, textvariable=self.susp_limit_var, width=5).pack(side="left", padx=2)

        ttk.Label(limits_frame, text="Infested >").pack(side="left", padx=2)
        self.inf_limit_var = tk.IntVar(value=27)
        ttk.Entry(limits_frame, textvariable=self.inf_limit_var, width=5).pack(side="left", padx=2)
        
        # Audio Device / File Path (moved down)
        self.device_label = ttk.Label(config_frame, text="Device:")
        self.device_label.grid(row=4, column=0, sticky="w", pady=5)
        
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(config_frame, textvariable=self.device_var, width=38)
        self.device_combo.grid(row=4, column=1, padx=5, sticky="w")
        
        self.file_btn = ttk.Button(config_frame, text="Browse", command=self.browse_wav_file)
        
        self.refresh_devices()
        
        # --- Dashboard Frame (Left) ---
        dash_frame = ttk.LabelFrame(left_frame, text="Dashboard", padding=10)
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
        self.energy_bar = ttk.Progressbar(energy_frame, orient="horizontal", length=200, mode="determinate", maximum=1.0)
        self.energy_bar.pack(side="left", padx=10, fill="x", expand=True)
        
        # Status Log
        self.status_log = tk.Listbox(left_frame, height=15)
        self.status_log.pack(fill="both", expand=True, padx=10, pady=5)
        
        # --- Control Area (Left) ---
        ctrl_frame = ttk.Frame(left_frame, padding=10)
        ctrl_frame.pack(fill="x")
        
        self.start_btn = ttk.Button(ctrl_frame, text="START TEST", command=self.toggle_test, width=20)
        self.start_btn.pack(pady=5)

        self.prep_btn = ttk.Button(ctrl_frame, text="Preprocessing Settings", command=self.open_prep_settings)
        self.prep_btn.pack(pady=5)
        
        # --- Plots Area (Right) ---
        # We will have 5 subplots: Waveform, Score Hist, Spectrogram, Energy, Timeline
        self.fig = Figure(figsize=(8, 14), dpi=100)
        self.ax_wave = self.fig.add_subplot(511)
        self.ax_hist = self.fig.add_subplot(512)
        self.ax_spec = self.fig.add_subplot(513)
        self.ax_energy = self.fig.add_subplot(514)
        self.ax_time = self.fig.add_subplot(515)
        
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
    def open_prep_settings(self):
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Preprocessing Pipeline Configuration")
        settings_win.geometry("400x300")
        settings_win.transient(self.root)
        settings_win.grab_set()

        frame = ttk.Frame(settings_win, padding=20)
        frame.pack(fill="both", expand=True)

        ttk.Checkbutton(frame, text="Use Bandpass Filter", variable=self.use_filter_var).grid(row=0, column=0, columnspan=2, sticky="w", pady=5)
        
        ttk.Label(frame, text="Low Cut (Hz):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.low_cut_var, width=15).grid(row=1, column=1, padx=5)

        ttk.Label(frame, text="Up Cut (Hz):").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.up_cut_var, width=15).grid(row=2, column=1, padx=5)

        ttk.Label(frame, text="FFT Win Size (s):").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.sub_win_size_var, width=15).grid(row=3, column=1, padx=5)

        ttk.Label(frame, text="Hop Size (s):").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.sub_hop_size_var, width=15).grid(row=4, column=1, padx=5)

        ttk.Label(frame, text="Mel Bands:").grid(row=5, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.n_mels_var, width=15).grid(row=5, column=1, padx=5)

        ttk.Label(frame, text="Seq Len (frames):").grid(row=6, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.seq_len_var, width=15).grid(row=6, column=1, padx=5)

        ttk.Button(frame, text="Close", command=settings_win.destroy).grid(row=7, column=0, columnspan=2, pady=20)

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
            
    def log(self, message):
        self.status_log.insert(tk.END, message)
        self.status_log.see(tk.END)
        
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
            self.log(f"Loaded Model: {os.path.basename(model_path)}")
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
        self.start_btn.configure(text="START TEST")
        self.calculate_diagnosis()
        self.log("Test Stopped.")
        
    def calculate_diagnosis(self):
        pos = self.positive_count
        susp_limit = self.susp_limit_var.get()
        inf_limit = self.inf_limit_var.get()
        
        if pos < susp_limit:
            self.diag_label.configure(text=f"HEALTHY (Pos: {pos})", foreground="green")
        elif pos <= inf_limit:
            self.diag_label.configure(text=f"SUSPICIOUS (Pos: {pos})", foreground="orange")
        else:
            self.diag_label.configure(text=f"INFESTED (Pos: {pos})", foreground="red")

    def mic_loop(self, device_idx):
        self.log(f"Starting Mic Stream on Device {device_idx}...")
        
        sample_rate = 44100
        buffer_duration = self.duration_var.get()
        buffer_len = int(sample_rate * buffer_duration)
        
        # Initialize buffer with zeros
        audio_buffer = np.zeros(buffer_len, dtype=np.float32)
        
        block_size = int(sample_rate * 0.5) # 0.5 sec blocks
        
        def callback(indata, frames, time, status):
            self.audio_queue.put(indata.copy())
            
        try:
            with sd.InputStream(device=device_idx, channels=1, samplerate=sample_rate, 
                                blocksize=block_size, callback=callback):
                while self.is_running:
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
                n_mels=self.n_mels_var.get(),
                seq_len=self.seq_len_var.get(),
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
            seq_len = self.seq_len_var.get()
            n_mels = self.n_mels_var.get()
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
            
            # Update Histogram (Score)
            if self.score_history:
                self.ax_hist.clear()
                self.ax_hist.hist(self.score_history, bins=10, range=(0,1), color='skyblue', edgecolor='black')
                self.ax_hist.set_title("Score Distribution")
                self.ax_hist.set_xlim(0, 1)
                self.ax_hist.grid(True)

            # Update Spectrogram (Heatmap)
            if self.current_spectrogram is not None:
                self.ax_spec.clear()
                # Transpose to have Freq on Y, Time on X
                # specs is (98, 32) -> .T is (32, 98)
                self.ax_spec.imshow(self.current_spectrogram.T, aspect='auto', origin='lower', cmap='inferno')
                self.ax_spec.set_title("Mel Spectrogram")
                self.ax_spec.set_ylabel("Mel Bands")
                self.ax_spec.set_xlabel("Time Frames")

            # Update Energy
            if self.energy_history:
                times, energy = zip(*self.energy_history)
                self.ax_energy.clear()
                self.ax_energy.plot(times, energy, 'g-')
                self.ax_energy.set_title("Energy Timeline (RMS)")
                self.ax_energy.set_ylim(0, max(0.1, max(energy) * 1.2)) # Dynamic limit
                self.ax_energy.grid(True)
            
            # Update Timeline
            if self.trigger_history:
                times, triggers = zip(*self.trigger_history)
                self.ax_time.clear()
                self.ax_time.plot(times, triggers, 'r-o', markersize=4)
                self.ax_time.set_title("Trigger Timeline (1=Pos, 0=Neg)")
                self.ax_time.set_ylim(-0.1, 1.1)
                self.ax_time.set_xlabel("Time (s)")
                self.ax_time.grid(True)
                
            self.canvas.draw()
            
        except Exception as e:
            self.log(f"Plot Error: {e}")

        if self.is_running:
            self.root.after(1000, self.update_plots)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelsTesterApp(root)
    root.mainloop()
