import sys
import subprocess
import importlib.util
import platform
import os

REQUIRED_PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "sounddevice",
    "soundfile",
    "librosa",
    "sklearn",  # scikit-learn
    "PIL",      # Pillow
    "rich",
    "matplotlib"
]

# TFLite runtime check
if platform.system() == "Linux" and platform.machine() != "x86_64":
    REQUIRED_PACKAGES.append("tflite_runtime")
else:
    REQUIRED_PACKAGES.append("tensorflow")

def install(package):
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_dependencies():
    print("Verifying dependencies...")
    missing = []
    
    # Check specific imports
    mapping = {
        "sklearn": "scikit-learn",
        "PIL": "Pillow",
        "tensorflow": "tensorflow",
        "tflite_runtime": "tflite-runtime"
    }

    for pkg in REQUIRED_PACKAGES:
        try:
            # Handle special import names
            import_name = pkg
            if pkg == "scikit-learn": import_name = "sklearn"
            if pkg == "Pillow": import_name = "PIL"
            if pkg == "tflite-runtime": import_name = "tflite_runtime"
            
            importlib.import_module(import_name)
        except ImportError:
            # Map back to pip package name
            pip_name = mapping.get(pkg, pkg)
            missing.append(pip_name)
        except OSError:
            # This happens if the package is installed but a system lib is missing (e.g. sounddevice -> portaudio)
            # We treat this as "installed" for pip purposes, and let check_system_libraries warn about it.
            pass

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        try:
            print("Attempting to install missing packages via pip...")
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            print("Please try running: pip install -r requirements.txt manually.")
            input("Press Enter to exit...")
            sys.exit(1)
    else:
        print("All Python dependencies are satisfied.")

def check_system_libraries():
    # Check for Tkinter
    try:
        import tkinter
    except ImportError:
        print("WARNING: Tkinter (GUI library) is missing.")
        if platform.system() == "Linux":
             print("  sudo apt-get install python3-tk")
        else:
             print("Please install Python with Tkinter support (tcl/tk).")
        print("The application will likely crash without it.\n")

    # Only for Linux mostly, check for PortAudio
    if platform.system() == "Linux":
        try:
            import sounddevice
            # Try to query devices to see if PortAudio loads
            sounddevice.query_devices()
        except Exception as e:
            print("WARNING: Audio system seems to have issues.")
            print(f"Error: {e}")
            print("\nIf you are on Linux, you might need to install 'libportaudio2':")
            print("  sudo apt-get install libportaudio2")
            print("\nProceeding anyway...")

def run_app():
    print("Starting Application...")
    try:
        # Run main.py in the same directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_script = os.path.join(script_dir, "main.py")
        subprocess.call([sys.executable, main_script])
    except Exception as e:
        print(f"Failed to launch application: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    print(f"Platform: {platform.system()} {platform.release()}")
    check_dependencies()
    check_system_libraries()
    run_app()
