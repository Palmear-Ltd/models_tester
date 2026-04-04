import numpy as np
import threading
import time
from inference_utils import AudioProcessor
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None

def verify_pipeline():
    print("Verifying Pipeline...")
    
    # 1. Load Resources
    processor = AudioProcessor()
    mean, var = processor.load_scaler("scaler.json")  # prefer
    if mean is None:
        mean, var = processor.load_scaler("scaler.npz")  # fallback    
    assert mean is not None, "Scaler load failed"
    
    interpreter = tflite.Interpreter(model_path="dummy_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("Resources Loaded.")
    
    # 2. Simulate Audio (2.5s of random noise)
    # Using 44100 Hz
    fs = 44100
    duration = 2.5
    raw_audio = np.random.uniform(-1, 1, int(fs * duration)).astype(np.float32)
    
    # 3. Extract Features
    start = time.time()
    specs = processor.extract_features(raw_audio, sr=fs)
    print(f"Feature Extraction Time: {time.time() - start:.4f}s")
    print(f"Features Shape: {specs.shape}")
    
    assert specs.shape == (98, 32), f"Expected (98, 32) but got {specs.shape}"
    
    # 4. Scale
    specs_scaled = processor.apply_scaler(specs, mean, var)
    
    # 5. Predict
    input_data = specs_scaled.reshape(input_details[0]['shape']).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"Prediction: {output_data[0][0]}")
    print("Pipeline Verified Successfully.")

if __name__ == "__main__":
    verify_pipeline()
