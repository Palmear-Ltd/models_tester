"""
TFLite model inference module.

Handles loading TFLite models and running inference.
"""

import numpy as np

# Try to import TFLite Interpreter
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None


class ModelInference:
    """
    Handles TFLite model loading and inference operations.
    
    Manages model lifecycle including loading, tensor allocation,
    and running inference on input data.
    """
    
    def __init__(self):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_path = None
    
    @staticmethod
    def is_tflite_available():
        """Check if TFLite runtime is available."""
        return tflite is not None
    
    def load_model(self, model_path):
        """
        Load a TFLite model from file.
        
        Args:
            model_path: Path to .tflite model file
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            RuntimeError: If TFLite is not available
            Exception: If model loading fails
        """
        if not self.is_tflite_available():
            raise RuntimeError("TFLite runtime not installed!")
        
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model_path = model_path
            return True
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    def get_input_shape(self):
        """Get the expected input shape for the model."""
        if self.input_details is None:
            return None
        return self.input_details[0]['shape']
    
    def get_output_shape(self):
        """Get the output shape of the model."""
        if self.output_details is None:
            return None
        return self.output_details[0]['shape']
    
    def predict(self, input_data):
        """
        Run inference on input data.
        
        Args:
            input_data: Preprocessed input data as numpy array
            
        Returns:
            Model output as numpy array
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.interpreter is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Ensure input is float32
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        # Set input tensor and run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get output tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data
    
    def predict_with_score(self, input_data, threshold=0.5):
        """
        Run inference and return prediction with score.
        
        Args:
            input_data: Preprocessed input data as numpy array
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            Tuple of (predicted_class, score)
            - predicted_class: 0 or 1
            - score: Confidence score for positive class
        """
        output_data = self.predict(input_data)
        
        # Handle different output formats
        if output_data.shape[-1] == 1:
            # Single output (sigmoid)
            score = float(output_data[0][0])
            predicted = 1 if score >= threshold else 0
        else:
            # Multiple outputs (softmax) - take probability of class 1
            score = float(output_data[0][1])
            predicted = 1 if score >= threshold else 0
        
        return predicted, score
    
    def is_loaded(self):
        """Check if a model is currently loaded."""
        return self.interpreter is not None
