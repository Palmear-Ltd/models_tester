# Palmear Audio Testing Tool - User Guide

A comprehensive guide to using the Palmear Audio Testing Tool for bioacoustic analysis.

## 🚀 Getting Started

### First Time Setup

1. **Complete installation** following the instructions in `INSTALL.md`
2. **Prepare your model files**:
   - `.tflite` model file (TensorFlow Lite model)
   - `scaler.npz` file (StandardScaler from training)
3. **Optional**: Prepare test audio files in WAV format

### Quick Start

1. **Launch the application**:
   ```bash
   python launcher.py
   # Or double-click run.command (macOS)
   ```

2. **Load your model**:
   - Click **Browse** next to "Model Path (.tflite)"
   - Select your TFLite model file
   - Click **Browse** next to "Scaler Path (.npz)"
   - Select your scaler file

3. **Click START TEST** to begin testing

## 📖 Main Interface

### Configuration Panel

#### Model Files
- **Model Path**: Your trained TensorFlow Lite model (`.tflite`)
- **Scaler Path**: StandardScaler used during training (`.npz`)

#### Input Source
- **Microphone**: Real-time audio from your microphone
  - Select device from dropdown menu
  - Use **⟳ Refresh** if your device doesn't appear
- **Wav File**: Pre-recorded audio file
  - Click **Browse** to select a `.wav` file

#### Detection Thresholds
- **Score Thresh**: Confidence threshold (0.0-1.0) for classification
  - Higher = stricter (fewer false positives)
  - Lower = more sensitive (may increase false positives)
- **Suspicious >=**: Minimum positive detections for "Suspicious" label
- **Infested >=**: Minimum positive detections for "Infested" label

### Dashboard

Real-time statistics and classification:
- **POSITIVE**: Count of frames classified as positive
- **NEGATIVE**: Count of frames classified as negative  
- **Diagnosis**: Overall assessment based on thresholds
  - 🟢 **HEALTHY**: Below suspicious threshold
  - 🟠 **SUSPICIOUS**: Between thresholds
  - 🔴 **INFESTED**: Above infested threshold
- **Current Energy (RMS)**: Audio input level indicator

### Visualization Plots

The right panel shows real-time visualizations:

1. **Waveform**: Raw audio signal over time
2. **Frequency Spectrum**: Frequency content analysis (FFT)
3. **Score Distribution**: Histogram of confidence scores
4. **Mel Spectrogram**: Time-frequency representation (heatmap)
5. **Energy Timeline**: RMS energy over time
6. **Trigger Timeline**: Detection events over time

Use the toolbar above each plot to:
- 🏠 Reset view
- ⬅️ ➡️ Pan
- 🔍 Zoom
- 💾 Save plot as image

## ⚙️ Settings

Open **Settings** from the File menu to access advanced configuration.

### Model Settings

Core parameters that should match your training configuration:

- **Duration (sec)**: Audio window duration
  - Default: 2.5 seconds
  - Should match training window size
  
- **Sequence Length (frames)**: Number of time frames
  - Default: 98 frames
  - Must match model input shape
  
- **Mel Bands**: Number of mel frequency bands
  - Default: 32 bands
  - Must match model input shape
  
- **FFT Window Size (s)**: Short-time FFT window
  - Default: 0.05 seconds (50ms)
  - Affects frequency resolution
  
- **Hop Size (s)**: Overlap between windows
  - Default: 0.025 seconds (25ms)
  - Affects time resolution

### Bandpass Filter (Optional)

Remove unwanted frequencies from audio:

- **Enable Filter**: Toggle bandpass filtering on/off
- **Low Cut (Hz)**: High-pass filter cutoff
  - Default: 500 Hz
  - Removes low-frequency noise
- **Up Cut (Hz)**: Low-pass filter cutoff
  - Default: 8000 Hz
  - Removes high-frequency noise

**When to use**: Enable if your training data was filtered, or to reduce environmental noise.

### Sample Rate & Normalization

Advanced preprocessing options:

- **Downsample Audio**: Optional downsampling before processing
  - Enable to reduce computational load
  - Useful for high sample rate recordings
  
- **Target Sample Rate (Hz)**: Downsample target rate
  - Range: 8000-44100 Hz
  - Lower = faster processing, less detail
  
- **Use PCEN Normalization**: Specialized normalization
  - ✅ **PCEN**: Optimized for bioacoustics
    - Better for varying background noise
    - Bioacoustic-optimized parameters
  - ❌ **Log (dB)**: Standard log scaling
    - Traditional approach
    - Good for controlled environments

**Important**: Use the same normalization method as training!

## 🎤 Microphone Testing

### Setup

1. **Select Input Source**: Choose "Microphone"
2. **Select Device**: Pick your microphone from dropdown
3. **Set Time Limit** (optional):
   - 0 = No limit (manual stop only)
   - >20 = Auto-stop after N seconds
4. **Monitor Audio** (optional):
   - Enable to hear what the mic captures
   - Select output device for playback

### Running a Test

1. Click **START TEST**
2. Speak/play audio near the microphone
3. Watch the dashboard update in real-time
4. Click **STOP TEST** or wait for auto-stop

### Saving Results

Enable **"Save results and audio"**:
- Short audio clips are saved automatically
- JSON files contain detection statistics
- Choose output directory with **Browse**

After stopping:
1. Dialog asks for expected label:
   - **Infested**: Sample should be classified as infested
   - **Healthy**: Sample should be clean
   - **Unknown**: Unsure or mixed
2. Files are saved with labels:
   - `TP_` = True Positive (correctly detected infested)
   - `TN_` = True Negative (correctly detected healthy)
   - `FP_` = False Positive (falsely detected infested)
   - `FN_` = False Negative (missed detection)

## 📁 File Testing

### Setup

1. **Select Input Source**: Choose "Wav File"
2. Click **Browse** to select a WAV file
3. Configure thresholds and settings as needed

### Running a Test

1. Click **START TEST**
2. The file is processed in simulated real-time
   - Mimics live microphone processing
   - Allows monitoring the progression
3. Wait for completion or click **STOP TEST**

**Note**: File testing does NOT save audio (original file already exists)

## 📊 Understanding Results

### Score Interpretation

- **Score = 0.0**: Very confident NEGATIVE
- **Score ≈ 0.5**: Uncertain (near decision boundary)
- **Score = 1.0**: Very confident POSITIVE

### Classification Logic

The tool uses a two-level threshold system:

```
Positive Count < Suspicious Threshold  → HEALTHY
Suspicious ≤ Count < Infested          → SUSPICIOUS  
Infested ≤ Count                       → INFESTED
```

Example (default: Suspicious=17, Infested=27):
- 10 positives → HEALTHY
- 20 positives → SUSPICIOUS
- 30 positives → INFESTED

### JSON Output Format

Saved JSON files contain:
```json
{
  "timestamp": "2026-01-04_15:30:45",
  "model": "model.tflite",
  "total_processed": 50,
  "positive_count": 28,
  "negative_count": 22,
  "predicted": "infested",
  "user_label": "infested",
  "classification": "TP",
  "settings": {
    "score_threshold": 0.5,
    "duration": 2.5,
    ...
  }
}
```

## 🔧 Advanced Tips

### Optimizing Performance

- **Enable downsampling** if processing is slow
- **Reduce duration** for faster response (but less context)
- **Lower mel bands** if your model supports it

### Improving Accuracy

- **Tune thresholds** based on your validation data
- **Enable filtering** to reduce noise
- **Use PCEN** for challenging acoustic environments
- **Adjust duration** to capture complete vocalizations

### Troubleshooting

**Low detection count:**
- Check if audio is reaching the microphone (energy bar)
- Verify score threshold isn't too high
- Ensure filter settings match training

**Too many false positives:**
- Increase score threshold
- Check for environmental noise
- Verify model quality with test data

**Choppy playback (monitoring):**
- Select a different output device
- Reduce processing complexity (downsample, fewer mel bands)

**Application freezes:**
- Reduce visualization frequency
- Close other audio applications
- Check system resources

## 🎯 Best Practices

1. **Always test with known samples first** to validate setup
2. **Document your threshold settings** for reproducibility
3. **Save representative samples** for future analysis
4. **Use consistent settings** across testing sessions
5. **Monitor system performance** during long recordings

## 📞 Support

For issues or questions:
1. Check `INSTALL.md` for setup problems
2. Verify model and scaler compatibility
3. Test with provided sample data (if available)
4. Review error messages in the status log
