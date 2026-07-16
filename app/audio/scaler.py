"""
Scaler operations for feature normalization.

Handles loading and applying StandardScaler transformations
for model input normalization.
"""

import json
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


class Scaler:
    """
    Handles scaler operations for feature normalization.

    Loads and applies StandardScaler transformations (mean and variance)
    to normalize features before feeding them to the model.
    """

    def __init__(self):
        self.mean = None
        self.var = None
        # Reason the most recent load() returned (None, None); None on success.
        self.last_error = None

    def _fail(self, reason: str):
        """Record why a load failed (logged + retained), return (None, None)."""
        self.last_error = reason
        logger.warning("Scaler load failed: %s", reason)
        return None, None
    
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
        self.last_error = None
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
                    return self._fail(f"{path}: JSON missing 'mean'/'mean_'")

                mean = np.asarray(mean, dtype=np.float64)

                # Prefer explicit var; otherwise compute from scale
                if var is not None:
                    var = np.asarray(var, dtype=np.float64)
                elif scale is not None:
                    scale = np.asarray(scale, dtype=np.float64)
                    var = scale * scale
                else:
                    return self._fail(f"{path}: JSON missing 'var'/'scale'")

                self.mean = mean
                self.var = var
                return self.mean, self.var

            # -------- NPZ (fallback / default) --------
            data = np.load(path)
            if "mean_" in data:
                mean = data["mean_"]
            elif "mean" in data:
                mean = data["mean"]
            else:
                return self._fail(
                    f"{path}: .npz missing 'mean'/'mean_' (keys: {list(data.keys())})"
                )
            if "var_" in data:
                var = data["var_"]
            elif "var" in data:
                var = data["var"]
            else:
                return self._fail(
                    f"{path}: .npz missing 'var'/'var_' (keys: {list(data.keys())})"
                )

            self.mean = np.asarray(mean, dtype=np.float64)
            self.var = np.asarray(var, dtype=np.float64)
            return self.mean, self.var
        except Exception as e:
            return self._fail(f"{path}: {type(e).__name__}: {e}")
    
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

        mean = np.asarray(mean, dtype=np.float64)
        var = np.asarray(var, dtype=np.float64)

        scale = np.sqrt(var)
        # Avoid divide by zero
        scale[scale == 0] = 1.0
        return (features - mean) / scale
