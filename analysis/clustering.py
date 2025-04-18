#   analysis/clustering.py
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from typing import Tuple, List, Optional, Union

def calculate_optimal_clusters(normalized_features: np.ndarray, max_clusters: int = 5) -> int:
    """Calculate the optimal number of clusters using silhouette analysis."""
    best_score = -1
    best_n_clusters = 2  # Default to 2 clusters
    
    for k in range(2, max_clusters + 1):
        if k >= len(normalized_features):
            continue
            
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(normalized_features)
        
        if len(np.unique(labels)) > 1:  # Ensure we have at least 2 clusters
            score = silhouette_score(normalized_features, labels)
            
            if score > best_score:
                best_score = score
                best_n_clusters = k
                
    return best_n_clusters

def run_clustering(normalized_features: np.ndarray, n_clusters: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster the normalized features to identify speech patterns.
    Automatically determines optimal number of clusters based on sample size.
    
    Args:
        normalized_features: Array of normalized feature values
        n_clusters: Optional number of clusters (auto-determined if None)
        
    Returns:
        Tuple of (cluster assignments, PCA results)
    """
    # Determine appropriate number of clusters based on sample size
    n_samples = len(normalized_features)
    
    # Handle special case with insufficient samples
    if n_samples < 2:
        print("Warning: Only one sample available - comparative analysis not possible")
        clusters = np.zeros(n_samples, dtype=int)
        
        # For PCA, adjust n_components to be valid
        if n_samples > 0:
            n_features = normalized_features.shape[1]
            n_components = min(1, min(n_samples, n_features))
            
            # Handle potential NaN values
            clean_features = np.nan_to_num(normalized_features, nan=0.0)
            
            # Create PCA with appropriate number of components
            pca = PCA(n_components=n_components)
            pca_results = pca.fit_transform(clean_features)
        else:
            pca_results = np.array([])
            
        return clusters, pca_results
        
    # For multiple samples, determine optimal number of clusters
    if n_clusters is None:
        max_clusters = min(n_samples - 1, 5)  # Maximum number of clusters to try
        n_clusters = calculate_optimal_clusters(normalized_features, max_clusters)
        
    # Perform clustering with optimal number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(normalized_features)
    
    # Dimensionality reduction for visualization
    n_components = min(2, min(n_samples, normalized_features.shape[1]))
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(normalized_features)
    
    return clusters, pca_results


def plot_feature_clusters(
    pca_results: np.ndarray, 
    labels: np.ndarray, 
    sample_names: Optional[List[str]] = None
) -> BytesIO:
    """
    Visualize clusters in 2D PCA space with sample labels.
    
    Args:
        pca_results: PCA-transformed feature data
        labels: Cluster assignments
        sample_names: Optional list of sample names for annotations
        
    Returns:
        BytesIO object containing the plot image
    """
    plt.figure(figsize=(12, 8))
    
    # Check dimensions of PCA results
    if len(pca_results.shape) == 1 or pca_results.shape[1] == 1:
        # For 1D results, plot on X axis with zeros for Y
        if len(pca_results.shape) == 1:
            x_values = pca_results
        else:
            x_values = pca_results[:, 0]
            
        scatter = plt.scatter(x_values, np.zeros_like(x_values), c=labels, cmap='viridis')
        
        # Add sample labels if provided
        if sample_names is not None:
            for i, name in enumerate(sample_names):
                plt.annotate(name, (x_values[i], 0), fontsize=9)
                
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel("Principal Component 1")
        plt.ylabel("Zero Axis")
        plt.title("Speech Feature Clusters (1D Projection)")
    else:
        # Normal 2D scatter plot
        scatter = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=labels, cmap='viridis')
        
        # Add sample labels if provided
        if sample_names is not None:
            for i, name in enumerate(sample_names):
                plt.annotate(name, (pca_results[i, 0], pca_results[i, 1]), fontsize=9)
                
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("Speech Feature Clusters")
    
    plt.grid(True)
    
    # Return the plot as a binary stream
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf