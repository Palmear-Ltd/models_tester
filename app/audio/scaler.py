"""
Scaler operations for feature normalization.

Handles loading and applying StandardScaler transformations
for model input normalization.
"""

import json
import os
import numpy as np


class Scaler:
    """
    Handles scaler operations for feature normalization.
    
    Loads and applies StandardScaler transformations (mean and variance)
    to normalize features before feeding them to the model.
    """
    
    def __init__(self):
        self.mean = None
        self.var = None
    
    def load(self, path: str):
        """
        Load scaler parameters from .json or .npz.
        JSON supported keys (any of these):
          
          - mean / mean_
          - var / var_
          - scale / scale_   (if var is missing, var = scale**2)

        Args:
            path: Path to scaler file (.json or .npz)
            
        Returns:
            Tuple of (mean, var) or (None, None) if loading fails
        """
        try:
            ext = os.path.splitext(path)[1].lower()

            # -------- JSON --------
            if ext == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)

                mean = obj.get("mean_", obj.get("mean", None))
                var = obj.get("var_", obj.get("var", None))
                scale = obj.get("scale_", obj.get("scale", None))

                if mean is None:
                    return None, None

                mean = np.asarray(mean, dtype=np.float32)

                # Prefer explicit var; otherwise compute from scale
                if var is not None:
                    var = np.asarray(var, dtype=np.float32)
                elif scale is not None:
                    scale = np.asarray(scale, dtype=np.float32)
                    var = scale * scale
                else:
                    return None, None

                self.mean = mean
                self.var = var
                return self.mean, self.var

            # -------- NPZ (fallback / default) --------
            data = np.load(path)
            mean = data["mean_"] if "mean_" in data else data["mean"]
            var = data["var_"] if "var_" in data else data["var"]

            self.mean = np.asarray(mean, dtype=np.float32)
            self.var = np.asarray(var, dtype=np.float32)
            return self.mean, self.var
        except Exception:
            return None, None
    
    def apply(self, features, mean=None, var=None):
        """
        Apply scaler transformation to features.
        
        Args:
            features: Feature array to normalize
            mean: Mean values (uses loaded mean if None)
            var: Variance values (uses loaded variance if None)
            
        Returns:
            Normalized features
        """
        if mean is None:
            mean = self.mean
        if var is None:
            var = self.var

        if mean is None or var is None:
            raise ValueError("Scaler not loaded: mean/var are None")

        mean = np.asarray(mean, dtype=np.float32)
        var = np.asarray(var, dtype=np.float32)

        scale = np.sqrt(var)
        # Avoid divide by zero
        scale[scale == 0] = 1.0
        return (features - mean) / scale
