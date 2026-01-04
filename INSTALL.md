# Installation Guide

## Quick Installation (Recommended)

### macOS

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install uv package manager**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install audio libraries**:
   ```bash
   brew install portaudio libsndfile
   ```

4. **Install Python 3.11**:
   ```bash
   uv python install 3.11
   ```

5. **Create virtual environment and install dependencies**:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

6. **Run the application**:
   ```bash
   python launcher.py
   # Or double-click run.command
   ```

7. **Grant microphone permission** when prompted:
   - System Settings → Privacy & Security → Microphone

### Linux (Ubuntu/Debian)

1. **Install system dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install python3.11 python3.11-venv python3-tk libportaudio2 libsndfile1
   ```

2. **Install uv** (optional, can use pip instead):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create virtual environment**:
   ```bash
   # Using uv
   uv venv .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   
   # Or using pip
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python launcher.py
   ```

### Windows

1. **Install Python 3.11+** from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

2. **Open Command Prompt or PowerShell** in the project directory

3. **Create virtual environment**:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```cmd
   python launcher.py
   ```

## Alternative: Using setup_env.py

For automated setup, run:
```bash
python setup_env.py
```

This will:
- Check Python version
- Install required packages
- Verify system dependencies
- Test audio functionality

## Verification

After installation, verify everything works:

1. **Check Python version**:
   ```bash
   python --version  # Should be 3.11 or higher
   ```

2. **Verify installation**:
   ```bash
   python verify_headless.py
   ```

3. **Test audio devices**:
   ```bash
   python -c "import sounddevice as sd; print(sd.query_devices())"
   ```

## Troubleshooting

### "No module named 'tkinter'"
- **Linux**: `sudo apt-get install python3-tk`
- **macOS**: Reinstall Python with tkinter support
- **Windows**: Reinstall Python, ensure "tcl/tk" is selected

### "PortAudio not found"
- **macOS**: `brew install portaudio`
- **Linux**: `sudo apt-get install libportaudio2`
- **Windows**: Should be included with sounddevice

### "soundfile requires libsndfile"
- **macOS**: `brew install libsndfile`
- **Linux**: `sudo apt-get install libsndfile1`
- **Windows**: Included with soundfile pip package

### Permission issues (macOS)
- Grant microphone access: System Settings → Privacy & Security → Microphone
- Allow the Terminal/Python app when prompted
