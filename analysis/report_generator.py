#   analysis/report_generator.py
from datetime import datetime
import pandas as pd

def generate_insights_report(insights, file_details=None):
    """
    Generates a formatted markdown report from speech analysis insights.
    
    Args:
        insights: Dictionary containing speech analysis insights
        file_details: Optional dictionary containing individual file analysis details
        
    Returns:
        str: Formatted markdown report
    """
    report = []
    
    # Header
    report.append("# Speech Analysis Insights Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n---\n")
    
    # Summary Section
    report.append("## Summary")
    summary = insights["summary"]
    report.append(f"- Total Samples Analyzed: {summary['total_samples']}")
    report.append(f"- Number of Clusters: {summary['clusters_found']}")
    report.append(f"- Risk Samples: {summary['risk_samples']} ({summary['risk_percentage']}%)\n")
    
    # Individual File Analysis
    if file_details:
        report.append("## Individual File Analysis")
        for filename, details in file_details.items():
            report.append(f"\n### {filename}")
            report.append("#### Stress Assessment")
            report.append(f"Overall: **{details['stress_assessment'].upper()}**")
            
            report.append("\nStress Indicators:")
            for indicator, value in details['stress_indicators'].items():
                report.append(f"- {indicator.replace('_', ' ').title()}: **{value}**")
            
            if details.get('transcribed_text'):
                report.append("\n#### Transcription")
                report.append(f"```\n{details['transcribed_text']}\n```\n")
    
    # Cluster Analysis
    report.append("## Cluster Analysis")
    for cluster_id, cluster_info in insights["clusters"].items():
        report.append(f"\n### {cluster_id.replace('_', ' ').title()}")
        report.append(f"- Size: {cluster_info['size']} samples ({cluster_info['percentage']}%)")
        report.append(f"- Cohesion Score: {cluster_info['silhouette_score']}")
        
        report.append("\n#### Distinctive Features:")
        for feature, details in cluster_info["distinctive_features"].items():
            report.append(f"- {feature}: {details['direction']} "
                        f"(diff: {details['std_diff']:.2f} std)")
        
        report.append(f"\n**Interpretation**: {cluster_info['interpretation']}")
    
    # Risk Patterns
    if insights.get("risk_patterns"):
        report.append("\n## Risk Pattern Analysis")
        if insights["risk_patterns"].get("common_deviations"):
            report.append("\n### Most Common Deviations")
            for feature, score in insights["risk_patterns"]["common_deviations"].items():
                report.append(f"- {feature}: {score:.2f} std")
        
        if insights["risk_patterns"].get("cluster_distribution"):
            report.append("\n### Risk Distribution Across Clusters")
            for cluster, count in insights["risk_patterns"]["cluster_distribution"].items():
                report.append(f"- Cluster {cluster}: {count} samples")
    
    # Key Features
    report.append("\n## Key Features")
    for feature, importance in insights["key_features"].items():
        report.append(f"- {feature}: {importance:.3f}")
    
    # Recommendations
    report.append("\n## Recommendations")
    for i, rec in enumerate(insights["recommendations"], 1):
        report.append(f"{i}. {rec}")
    
    return "\n".join(report)