#   analysis/insights.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from analysis.report_generator import generate_insights_report

def get_speech_insights(
    features: pd.DataFrame, 
    clusters: np.ndarray, 
    risk_samples: Optional[List[int]] = None,
    generate_report: bool = False,
    file_details: Optional[Dict[str, Any]] = None
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], str]]:
    """
    Analyzes speech data to identify patterns and potential concerns.
    Handles single sample cases gracefully.
    
    Args:
        features: DataFrame with speech features for each sample
        clusters: Group assignments for each speech sample
        risk_samples: List of samples flagged as potential risks
        generate_report: Whether to generate a markdown report
        file_details: Dictionary with individual file analysis details
        
    Returns:
        A dictionary with organized insights about the speech patterns,
        and optionally a markdown report if generate_report is True
    """
    results = {
        "summary": {},
        "clusters": {},
        "risk_patterns": {},
        "key_features": {},
        "recommendations": []
    }

    # Handle empty or None inputs
    if features is None or len(features) == 0:
        results["summary"] = {
            "total_samples": 0,
            "clusters_found": 0,
            "risk_samples": 0,
            "risk_percentage": 0
        }
        results["recommendations"].append("No samples provided for analysis")
        return (results, "") if generate_report else results

    # Statistics
    total_samples = len(features)
    unique_clusters = len(np.unique(clusters))
    high_risk_count = len(risk_samples) if risk_samples is not None else 0
    risk_percentage = round(high_risk_count / total_samples * 100, 1) if total_samples > 0 else 0

    # Stats summary
    results["summary"] = {
        "total_samples": total_samples,
        "clusters_found": unique_clusters,
        "risk_samples": high_risk_count,
        "risk_percentage": risk_percentage
    }

    # Single sample case - simplified insights
    if total_samples == 1:
        results["clusters"]["cluster_0"] = {
            "size": 1,
            "percentage": 100.0,
            "silhouette_score": "N/A",
            "distinctive_features": {},
            "interpretation": "Only one sample available - comparative analysis not possible"
        }
        
        # Get the top 5 most extreme feature values
        sample_values = features.iloc[0]
        sorted_features = sample_values.abs().sort_values(ascending=False).head(5)
        
        for feature in sorted_features.index:
            results["clusters"]["cluster_0"]["distinctive_features"][feature] = {
                "value": float(sample_values[feature]),
                "diff_from_mean": 0,
                "std_diff": 0,
                "direction": "N/A"
            }
        
        results["key_features"] = {
            feature: 1.0 for feature in sorted_features.index
        }
        
        results["recommendations"].append("Collect more samples for comparative analysis")
        
        if generate_report:
            report = generate_insights_report(results, file_details)
            return results, report
        return results

    # Multiple samples case - full analysis
    for cluster_id in range(unique_clusters):
        cluster_samples = features.iloc[clusters == cluster_id]
        cluster_size = len(cluster_samples)
        
        if cluster_size == 0:
            continue

        cluster_average = cluster_samples.mean()
        overall_average = features.mean()

        # Handle potential NaN values
        cluster_average = cluster_average.fillna(0)
        overall_average = overall_average.fillna(0)

        feature_variation = features.std().fillna(0.001)  # Avoid division by zero
        standardized_differences = (cluster_average - overall_average) / feature_variation

        standout_features = standardized_differences.abs().sort_values(ascending=False).head(5)

        cluster_cohesion = "N/A"
        if len(cluster_samples) > 1 and unique_clusters > 1:
            try:
                from sklearn.metrics import silhouette_samples
                cluster_indices = np.where(clusters == cluster_id)[0]
                cluster_cohesion = silhouette_samples(features.values, clusters)[cluster_indices].mean()
                cluster_cohesion = round(float(cluster_cohesion), 3)
            except Exception:
                pass

        results["clusters"][f"cluster_{cluster_id}"] = {
            "size": int(cluster_size),
            "percentage": round(cluster_size / total_samples * 100, 1),
            "silhouette_score": cluster_cohesion,
            "distinctive_features": {
                feature: {
                    "value": float(cluster_average[feature]),
                    "diff_from_mean": float(cluster_average[feature] - overall_average[feature]),
                    "std_diff": float(standardized_differences[feature]),
                    "direction": "higher" if standardized_differences[feature] > 0 else "lower"
                }
                for feature in standout_features.index
            }
        }

        description = "This cluster is characterized by "
        feature_descriptions = []

        for feature in standout_features.index[:3]:
            direction = "higher" if standardized_differences[feature] > 0 else "lower"
            strength = "significantly " if abs(standardized_differences[feature]) > 2 else ""
            feature_descriptions.append(f"{strength}{direction} {feature}")

        if feature_descriptions:
            results["clusters"][f"cluster_{cluster_id}"]["interpretation"] = description + ", ".join(feature_descriptions)
        else:
            results["clusters"][f"cluster_{cluster_id}"]["interpretation"] = "No distinctive features identified"

    if risk_samples is not None and len(risk_samples) > 0:
        risk_data = features.iloc[risk_samples]

        risk_by_cluster = {}
        for idx in risk_samples:
            cluster = int(clusters[idx])
            risk_by_cluster[cluster] = risk_by_cluster.get(cluster, 0) + 1

        results["risk_patterns"]["cluster_distribution"] = risk_by_cluster

        # Handle potential NaN values in z-score calculation
        features_mean = features.mean().fillna(0)
        features_std = features.std().fillna(0.001)  # Avoid division by zero
        risk_z_scores = (risk_data - features_mean) / features_std
        mean_deviations = risk_z_scores.abs().mean().sort_values(ascending=False)

        results["risk_patterns"]["common_deviations"] = {
            feature: round(float(score), 2) for feature, score in mean_deviations.head(5).items()
        }

        results["risk_patterns"]["individual_samples"] = {}
        for idx in risk_samples:
            sample_id = features.index[idx]
            sample = features.iloc[idx]
            sample_z = (sample - features_mean) / features_std
            extreme_features = sample_z.abs().sort_values(ascending=False).head(3)

            results["risk_patterns"]["individual_samples"][sample_id] = {
                "cluster": int(clusters[idx]),
                "extreme_features": {
                    feature: {
                        "value": float(sample[feature]),
                        "z_score": float(sample_z[feature])
                    }
                    for feature in extreme_features.index
                }
            }

    # Identify which features are most important overall
    feature_importance = {}
    for feature in features.columns:
        # Importance based on variation
        feature_std = features[feature].std()
        
        if unique_clusters > 1:
            between_cluster_variance = np.var([
                features.loc[clusters == c, feature].mean()
                for c in range(unique_clusters)
            ]) 
        else:
            between_cluster_variance = 0

        importance = (feature_std * between_cluster_variance) if between_cluster_variance > 0 else feature_std
        feature_importance[feature] = float(importance)

    max_importance = max(feature_importance.values()) if feature_importance else 1
    for feature in feature_importance:
        feature_importance[feature] /= max_importance

    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    results["key_features"] = {
        feature: round(float(score), 3) for feature, score in sorted_features[:10]
    }

    # Generate recommendations
    if high_risk_count > 0:
        top_risk_features = list(results["risk_patterns"]["common_deviations"].keys())[:3]
        if top_risk_features:
            results["recommendations"].append(
                f"Review {high_risk_count} samples identified as potential risks, particularly focusing on "
                f"abnormal patterns in {', '.join(top_risk_features)}"
            )

    if unique_clusters > 1:
        for cluster_id in range(unique_clusters):
            if f"cluster_{cluster_id}" in results["clusters"]:
                cluster_info = results["clusters"][f"cluster_{cluster_id}"]
                if cluster_info["size"] < total_samples * 0.2:  # If cluster is small (< 20%)
                    top_features = list(cluster_info["distinctive_features"].keys())
                    if top_features:
                        top_feature = top_features[0]
                        results["recommendations"].append(
                            f"Investigate Cluster {cluster_id} as a minority group ({cluster_info['percentage']}% of samples) "
                            f"with distinctive {top_feature} patterns"
                        )

    if total_samples == 1:
        results["recommendations"].append("Collect more samples for comparative analysis")
    else:
        top_features = list(results["key_features"].keys())[:3]
        if top_features:
            results["recommendations"].append(
                f"Focus future analysis on the top identified features: {', '.join(top_features)}"
            )

    if generate_report:
        report = generate_insights_report(results, file_details)
        return results, report
        
    return results