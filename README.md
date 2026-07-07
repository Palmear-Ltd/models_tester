# Palmear Audio Testing Tool

A professional audio testing application for bioacoustic analysis using TFLite models. Features real-time microphone processing, file analysis, and comprehensive visualization tools.

## 📋 Features

- **Real-time Audio Processing**: Test with microphone input or pre-recorded WAV files
- **Advanced Signal Processing**:
  - Optional downsampling for preprocessing
  - Bandpass filtering (configurable high/low cut frequencies)
  - PCEN normalization for bioacoustics or standard log (dB) normalization
- **TFLite Model Support**: Load and run inference with TensorFlow Lite models
- **Comprehensive Visualization**:
  - Real-time waveform display
  - Frequency spectrum analysis
  - Mel spectrogram heatmaps
  - Score distribution plots
  - Energy and trigger timelines
- **Flexible Configuration**: Adjust mel bands, FFT parameters, sequence length, and more
- **Result Tracking**: Save audio snapshots and detection results to JSON
- **Sensor/Cable Link Health Detection**: Automatically flags a broken or intermittent connection between the piezo sensor and the audio jack, independent of the classifier results

## 🗂️ Project Structure

```
models_tester/
├── app/                          # Application code
│   ├── audio/                    # Audio processing modules
│   │   ├── processor.py          # Filtering, resampling, downsampling
│   │   ├── features.py           # Mel spectrogram extraction
│   │   └── scaler.py             # Feature normalization
│   ├── model/                    # Model operations
│   │   └── inference.py          # TFLite model loading & inference
│   └── ui/                       # User interface components
│       └── settings_dialog.py    # Configuration dialog
│
├── models/                       # Model files (.tflite, scaler.npz)
├── test_data/                    # Test audio files
│
├── main.py                       # Application entry point
├── launcher.py                   # Launcher with dependency checks
├── run.command                   # macOS launcher script
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- macOS, Linux, or Windows
- Microphone (for real-time testing)

### Installation

1. **Clone the repository** (or download the project)

2. **Install uv** (recommended) or use pip:
   ```bash
   # Using uv (recommended)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Or use pip
   pip install --upgrade pip
   ```

3. **Install system dependencies** (Linux only):
   ```bash
   sudo apt-get update
   sudo apt-get install python3-tk libportaudio2 libsndfile1
   ```

4. **Create virtual environment and install dependencies**:
   ```bash
   # Using uv
   uv venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -r requirements.txt
   
   # Or using pip
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Run the application**:
   ```bash
   python launcher.py
   ```
   
   Or on macOS, double-click `run.command`

## 🎯 Usage

### 1. Load Your Model

- Click **Browse** next to "Model Path" and select your `.tflite` file
- Click **Browse** next to "Scaler Path" and select your `scaler.npz` file
- The model and scaler must be compatible (same feature dimensions)

### 2. Configure Settings

Open **Settings** from the menu to configure:

#### Model Settings
- **Duration**: Audio window duration (default: 2.5s)
- **Sequence Length**: Number of frames for model input
- **Mel Bands**: Number of mel frequency bands
- **FFT Window Size**: FFT window size in seconds
- **Hop Size**: Hop size for FFT

#### Bandpass Filter (Optional)
- **Enable Filter**: Toggle bandpass filtering
- **Low Cut**: High-pass filter cutoff frequency (Hz)
- **Up Cut**: Low-pass filter cutoff frequency (Hz)

#### Sample Rate & Normalization
- **Downsample Audio**: Optional downsampling before processing
- **Target Sample Rate**: Target rate for downsampling
- **Use PCEN Normalization**: Use PCEN (bioacoustics) instead of log (dB)

### 3. Configure Detection Thresholds

- **Score Thresh**: Confidence threshold (0.0-1.0) for positive detection
- **Suspicious >=**: Minimum positive count for "Suspicious" classification
- **Infested >=**: Minimum positive count for "Infested" classification

### 4. Select Input Source

- **Microphone**: Real-time testing with your microphone
  - Select device from dropdown
  - Set optional time limit for auto-stop
  - Enable "Monitor Audio" to hear the input
- **Wav File**: Process pre-recorded audio
  - Click **Browse** to select a WAV file

### 5. Run Test

1. Click **START TEST**
2. Monitor the dashboard for results:
   - **Positive/Negative counters** update in real-time
   - **Diagnosis** shows classification (Healthy/Suspicious/Infested)
   - **Plots** show waveform, spectrogram, and analysis
3. Click **STOP TEST** when done (or let it auto-stop with time limit)

### 6. Save Results (Optional)

Enable **"Save results and audio"** to:
- Save audio snapshots as WAV files
- Export detection results as JSON
- Choose output directory for saved files

## 📊 Understanding the Dashboard

- **POSITIVE**: Count of frames classified as positive
- **NEGATIVE**: Count of frames classified as negative
- **Diagnosis**: Overall classification based on thresholds
  - GREEN = Healthy (below suspicious threshold)
  - ORANGE = Suspicious (between thresholds)
  - RED = Infested (above infested threshold)
- **Current Energy (RMS)**: Real-time audio energy level

## 🔌 Sensor/Cable Link Health Detection

The tool continuously monitors the physical link between the piezo sensor (needle) and the audio input jack — the cable/connection that most often gets damaged in the field from rough handling (the needle being forced into a tree, cables getting twisted or pressured, or the joint between the sensor and cable working loose). This runs automatically alongside every test and never blocks, slows, or changes the classifier's detection results.

### What it detects

- **Complete signal loss** — a fully broken or disconnected cable (dead silence)
- **Clicking/crackling** — an intermittent or loose connection cutting in and out
- **Recurring dropouts** — the same problem repeating across several seconds of recording, not just a one-off glitch

When a problem is detected you'll see, in real time:
- The **Signal Health** label turns orange (WARNING) or red (FAULT)
- A new line reading **"Likely cause: SENSOR_LINK — ..."** with a plain-language explanation
- A highlighted **CAUSE** row in the **Signal Health Detail** panel
- A `[likely-cause]` entry in the log

### Testing locally

1. Run the automated test suite:
   ```bash
   .venv/bin/python -m pytest tests/ -q
   ```

2. Confirm detection works on a known-bad recording:
   - Select **Wav File** input, browse to `test_data/audio_signal_health/broken_mic/F/`, and pick any file
   - Click **START TEST**
   - The health indicator should turn orange/red with a "SENSOR_LINK" cause shown

3. Confirm it stays quiet on a known-good recording:
   - Browse to `test_data/audio_signal_health/chinese_needle_1/` or `test_data/audio_signal_health/sanded_needle_1/`
   - The health indicator should stay green, with no cause shown

4. Try **Validate Acquisition** (in the run bar) — it captures ~20 seconds of the loaded/recorded audio and shows a PASS/WARNING/FAIL popup summary, including the likely cause if one was found.

### Testing in the field

1. Select **Microphone** input and start a test with a known-good sensor connection — confirm the indicator stays healthy.
2. Gently wiggle or partially loosen the cable/connector while recording (without permanently damaging it) — the indicator should flag "SENSOR_LINK" within a couple of seconds.
3. Use **Validate Acquisition** before starting a real session as a quick pre-check on the connection.
4. Every flagged issue and validation run is automatically saved as JSON under `reports/`. Periodically collecting these from field devices — especially genuine failures as they happen naturally — helps validate and improve detection accuracy over time.

## 🔧 Advanced Features

### Feature Extraction Options

The tool supports both standard and bioacoustic-optimized feature extraction:

- **Log (dB) Normalization**: Standard mel spectrogram with logarithmic scaling
- **PCEN Normalization**: Per-Channel Energy Normalization optimized for bioacoustics
  - Gain: 0.8
  - Bias: 10
  - Power: 0.25
  - Time constant: 0.06

### Audio Preprocessing Pipeline

1. **Optional Downsampling**: Reduce sample rate before processing
2. **Resampling**: Standardize to target sample rate (default: 44100 Hz)
3. **Bandpass Filtering**: Remove unwanted frequencies
4. **Feature Extraction**: Generate mel spectrogram
5. **Normalization**: Apply PCEN or log scaling
6. **Standardization**: Apply trained scaler for model input

## 🐛 Troubleshooting

### "TFLite runtime not installed"
- Install TensorFlow Lite: `pip install tensorflow` or `pip install tflite-runtime`

### No microphone devices shown
- Check microphone permissions in System Settings (macOS)
- Verify microphone is connected and recognized by OS
- Click **Refresh** button to reload devices

### Audio monitoring not working
- Ensure output device supports the sample rate
- Try a different output device
- Check system audio settings

### Model loading fails
- Verify model file is a valid `.tflite` file
- Check model input shape matches your settings
- Ensure scaler dimensions match model expectations

## 📝 Requirements

Core dependencies:
- Python 3.11+
- TensorFlow Lite (tensorflow or tflite-runtime)
- NumPy
- Librosa (audio processing)
- SciPy (signal processing)
- Sounddevice (audio I/O)
- Soundfile (WAV file handling)
- Matplotlib (visualization)
- scikit-learn (for scaler compatibility)

See `requirements.txt` for complete list.

## 🤝 Contributing

This is a research tool for bioacoustic analysis. For modifications:
1. Code is organized in the `app/` directory
2. Follow the existing module structure
3. Add tests for new features
4. Update documentation

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

Built for bioacoustic research and pest detection applications.
