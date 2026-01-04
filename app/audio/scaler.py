"""
Scaler operations for feature normalization.

Handles loading and applying StandardScaler transformations
for model input normalization.
"""

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
    
    def load(self, path):
        """
        Load scaler parameters from .npz file.
        
        Args:
            path: Path to .npz file containing mean and variance
            
        Returns:
            Tuple of (mean, var) or (None, None) if loading fails
        """
        try:
            data = np.load(path)
            self.mean = data['mean_'] if 'mean_' in data else data['mean']
            self.var = data['var_'] if 'var_' in data else data['var']
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
        
        scale = np.sqrt(var)
        # Avoid divide by zero
        scale[scale == 0] = 1.0
        return (features - mean) / scale
