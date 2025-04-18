#   analysis/preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Union, Any

def clean_and_normalize_data(feature_matrix: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Handle missing values and normalize features.
    
    Args:
        feature_matrix: Input data matrix or DataFrame
        
    Returns:
        Normalized feature matrix as numpy array
    """
    # Handle empty matrix
    if isinstance(feature_matrix, np.ndarray) and feature_matrix.size == 0:
        return feature_matrix
        
    # Convert to DataFrame for robust type handling
    df = pd.DataFrame(feature_matrix)
    
    # Convert all columns to numeric, non-convertible values become NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Fill NaN with 0.0
    df = df.fillna(0.0)
    
    # Convert back to numpy array of float
    clean_features = df.values.astype(float)

    # Normalize features
    if len(clean_features) > 1:  # Only normalize if multiple samples
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(clean_features)
    else:
        normalized_features = clean_features

    return normalized_features