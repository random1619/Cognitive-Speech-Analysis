# analysis/risk_analysis.py
import numpy as np
import scipy.stats
from sklearn.ensemble import IsolationForest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from io import BytesIO
from typing import List, Dict, Any, Union

def run_risk_analysis(
    normalized_features: np.ndarray, 
    feature_names: List[str], 
    audio_features: Dict[str, Dict[str, Any]]
) -> List[int]:
    """
    Perform risk analysis on speech samples to identify abnormal patterns.
    Robust implementation that handles single sample cases.
    
    Args:
        normalized_features: Normalized feature matrix
        feature_names: List of feature names
        audio_features: Dictionary of audio features by filename
        
    Returns:
        List of indices for samples identified as potential risks
    """
    print("\n--- Risk Analysis Results ---")

    if normalized_features is None or len(normalized_features) < 3:
        print("Insufficient data for meaningful risk analysis")
        return []

    # Replace any NaN/Inf values to prevent errors
    clean_features = np.nan_to_num(normalized_features, nan=0.0)
    
    # Method 1: Isolation Forest for anomaly detection
    print("Performing anomaly detection...")
    
    # Only run isolation forest if we have enough samples
    if len(clean_features) >= 3:  # IsolationForest needs at least 3 samples
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = iso_forest.fit_predict(clean_features)
        anomalies = np.where(anomaly_scores == -1)[0]
    else:
        anomalies = []

    # Method 2: Statistical outlier detection using z-scores
    # Only calculate z-scores if we have enough samples
    if len(clean_features) >= 2:  # Need at least 2 samples for meaningful z-scores
        z_scores = np.abs(scipy.stats.zscore(clean_features, nan_policy='omit'))
        outliers = np.where(np.any(z_scores > 3, axis=1))[0]
    else:
        outliers = []

    potential_risks = list(set(anomalies) | set(outliers))
    
    feature_df = pd.DataFrame(clean_features, columns=feature_names)
    feature_df.index = list(audio_features.keys())
    
    # Display risk results
    if potential_risks:
        print(f"\nFound {len(potential_risks)} potentially abnormal speech samples:")
        for idx in potential_risks:
            sample_id = feature_df.index[idx]
            print(f"- Sample {sample_id}")
            
    return potential_risks

def plot_risk_visualization(
    features: np.ndarray, 
    risk_indices: List[int], 
    feature_names: List[str], 
    audio_features: Dict[str, Dict[str, Any]]
) -> BytesIO:
    """
    Create visualizations to highlight potentially risky samples.
    Handles both single and multiple sample cases.
    
    Args:
        features: Feature matrix
        risk_indices: List of indices for samples identified as risks
        feature_names: List of feature names
        audio_features: Dictionary of audio features by filename
        
    Returns:
        BytesIO object containing the plot image
    """
    # Clean features to avoid NaN/Inf issues
    clean_features = np.nan_to_num(features, nan=0.0)
    
    # Check if we have enough samples for 2D PCA
    n_samples = clean_features.shape[0]
    max_components = min(n_samples, clean_features.shape[1])
    
    # Adjust number of components based on available data
    n_components = min(2, max_components)
    
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Perform PCA with adjusted components
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(clean_features)
    
    # Plot based on dimensions available
    if n_components == 1 or pca_results.shape[1] == 1:
        # For 1D results, plot on X axis with zeros for Y
        normal_indices = [i for i in range(len(features)) if i not in risk_indices]
        
        # Plot normal samples
        if normal_indices:
            plt.scatter(
                pca_results[normal_indices, 0],
                np.zeros(len(normal_indices)),
                c='blue',
                label='Normal',
                alpha=0.7
            )
        
        # Plot risky samples
        if risk_indices:
            plt.scatter(
                pca_results[risk_indices, 0],
                np.zeros(len(risk_indices)),
                c='red',
                marker='X',
                s=100,
                label='Potential Risk',
                alpha=0.9
            )
            
            # Add sample labels for risky samples
            for idx in risk_indices:
                sample_id = list(audio_features.keys())[idx]
                plt.annotate(
                    sample_id,
                    (pca_results[idx, 0], 0), 
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9
                )
                
        plt.xlabel("Principal Component 1")
        plt.ylabel("Value")
        plt.title("Speech Analysis Risk Assessment (1D Projection)")
    else:
        # 2D visualization
        normal_indices = [i for i in range(len(features)) if i not in risk_indices]
        
        # Plot normal samples
        if normal_indices:
            plt.scatter(
                pca_results[normal_indices, 0],
                pca_results[normal_indices, 1],
                c='blue',
                label='Normal',
                alpha=0.7
            )

        # Plot risky samples
        if risk_indices:
            plt.scatter(
                pca_results[risk_indices, 0],
                pca_results[risk_indices, 1],
                c='red',
                marker='X',
                s=100,
                label='Potential Risk',
                alpha=0.9
            )

            # Add sample labels for risky samples
            for idx in risk_indices:
                sample_id = list(audio_features.keys())[idx]
                plt.annotate(
                    sample_id,
                    (pca_results[idx, 0], pca_results[idx, 1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9
                )

        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("Speech Analysis Risk Assessment")
    
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Return the plot as a binary stream
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf