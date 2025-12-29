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

if platform.system() == "Linux" and platform.machine() != "x86_64":
    REQUIRED_PACKAGES.append("tflite_runtime")
else:
    REQUIRED_PACKAGES.append("tensorflow")

def check_and_install():
    print("=== Environment Setup ===")
    print(f"Platform: {platform.system()} {platform.release()}")
    
    missing = []
    mapping = {
        "sklearn": "scikit-learn",
        "PIL": "Pillow",
        "tensorflow": "tensorflow",
        "tflite_runtime": "tflite-runtime"
    }

    print("Checking Python packages...")
    for pkg in REQUIRED_PACKAGES:
        try:
            import_name = pkg
            if pkg == "scikit-learn": import_name = "sklearn"
            if pkg == "Pillow": import_name = "PIL"
            if pkg == "tflite-runtime": import_name = "tflite_runtime"
            importlib.import_module(import_name)
        except ImportError:
            pip_name = mapping.get(pkg, pkg)
            missing.append(pip_name)
        except OSError:
            pass # Installed but system lib missing

    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("✅ Python dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            return
    else:
        print("✅ All Python dependencies are already installed.")

    print("\nChecking System Libraries...")
    issues = False
    
    # Tkinter Check
    try:
        import tkinter
        print("✅ GUI Library (Tkinter) found.")
    except ImportError:
        print("❌ WARNING: Tkinter is missing.")
        if platform.system() == "Linux":
            print("   Run: sudo apt-get install python3-tk")
        else:
            print("   Please reinstall Python and ensure 'tcl/tk' is selected.")
        issues = True

    # Audio details
    if platform.system() == "Linux":
        try:
            import sounddevice
            sounddevice.query_devices()
            print("✅ Audio System (PortAudio) found.")
        except Exception as e:
            print("❌ WARNING: Audio system issue detected.")
            print(f"   Error: {e}")
            print("   Run: sudo apt-get install libportaudio2")
            issues = True
            
    if issues:
        print("\n⚠️  Setup completed with warnings. Please fix the issues above.")
    else:
        print("\n🎉 Setup verified successfully! You can now run 'python launcher.py'.")

    input("\nPress Enter to exit...")

if __name__ == "__main__":
    check_and_install()
