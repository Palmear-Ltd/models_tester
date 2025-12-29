# Audio Testing Tool - User Guide

This tool allows you to test audio models using your microphone or recorded wav files.

## 🚀 First Time Setup

Before running the tool for the first time, you need to set up the environment.

1.  **Install Python**: Ensure Python 3.9+ is installed.
2.  **Run Setup Script**:
    - Open a terminal or command prompt in this folder.
    - Run the following command:
      ```bash
      python setup_env.py
      ```
    - This will install all necessary libraries and check your system compatibility.

### ⚠️ Common Issues (Linux Users)
If you are on Linux (Ubuntu/Debian), you may need to install system libraries manually if the setup script complains:
```bash
sudo apt-get update
sudo apt-get install python3-tk libportaudio2
```

---

## ▶️ How to Run

To start the application, simply run the launcher:

```bash
python launcher.py
```
*(You can also double-click `launcher.py` if your system is configured to run Python scripts)*

---

## 🛠️ Using the Tool

### 1. Configuration
- **Model Path**: Click "Browse" and select your `.tflite` model file.
- **Scaler Path**: Click "Browse" and select your `scaler.npz` file.
    - *Note: The scaler must match the one used during training.*
- **Duration**: Set the sliding window duration (default is **2.5 seconds**).
- **Thresholds**:
    - **Score Thresh**: Minimum confidence score (0.0 - 1.0) to count as Positive (default 0.5).
    - **Suspicious >=**: Minimum positive samples to classify result as "Suspicious" (default 17).
    - **Infested >**: Minimum positive samples to classify result as "Infested" (default 27).
- **Input Source**:
    - **Microphone**: Uses your computer's mic for real-time testing.
    - **Wav File**: Processes a pre-recorded `.wav` file.
- **Device**: Select your microphone from the dropdown list.

### 2. Running a Test
1.  Click the **START TEST** button.
2.  **If using Mic**: Speak into the microphone. The tool will continuously update the "Positive" and "Negative" counters based on the model's detection.
3.  **If using File**: The tool will simulate playback of the file and process it.
4.  Watch the **Dashboard** for results.
    - **RED** = Positive Detection
    - **GREEN** = Negative Detection

5.  Click **STOP TEST** to finish.

---

## 📁 Files in this Folder
- `launcher.py`: The main script to run the app.
- `setup_env.py`: Script to install dependencies.
- `main.py`: The application code.
- `requirements.txt`: List of required Python libraries.
