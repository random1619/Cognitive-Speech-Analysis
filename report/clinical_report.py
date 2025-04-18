#   report/clinical_report.py
def generate_clinical_report(analysis_results):
    """Generate a comprehensive clinical report from analysis results."""
    report = {
        "insightful_features": extract_insightful_features(analysis_results),
        "ml_methods": describe_ml_methods(),
        "clinical_robustness": recommend_next_steps()
    }
    return report

def extract_insightful_features(analysis_results):
    """Identify and explain the most insightful features from the analysis."""
    
    # Extract feature information from all files
    features_by_file = {}
    if analysis_results.get("file_details"):
        for filename, file_data in analysis_results["file_details"].items():
            features_by_file[filename] = file_data.get("features", {})
    
    # Identify the most informative features across all files
    insights = {
        "acoustic_features": {
            "pitch_variability": {
                "description": "Variation in vocal pitch (frequency)",
                "insight": "High pitch variability can indicate cognitive load or emotional state changes. Research shows this correlates with working memory demands and executive function.",
                "clinical_relevance": "Reduced pitch variability may indicate cognitive impairment, while excessive variability can suggest emotional distress or difficulty with speech planning."
            },
            "speech_rate": {
                "description": "Speed of speech production",
                "insight": "Speech rate reflects cognitive processing speed and planning abilities. Reduced or variable speech rate may indicate increased cognitive load.",
                "clinical_relevance": "Slowed speech rate is associated with cognitive decline, particularly in executive function and processing speed domains."
            },
            "pause_patterns": {
                "description": "Frequency and duration of pauses during speech",
                "insight": "Pauses reflect cognitive processing, word retrieval, and planning abilities.",
                "clinical_relevance": "Increased pause frequency and duration have been linked to memory impairments and word-finding difficulties in MCI and early dementia."
            },
            "jitter_shimmer": {
                "description": "Voice quality measures (frequency and amplitude perturbation)",
                "insight": "Jitter and shimmer reflect vocal stability and control.",
                "clinical_relevance": "Increased jitter and shimmer have been associated with neurological conditions affecting speech motor control."
            }
        },
        "linguistic_features": {
            "hesitation_markers": {
                "description": "Fillers like 'um', 'uh', 'er'",
                "insight": "Hesitation markers indicate word-finding difficulties and speech planning issues.",
                "clinical_relevance": "Increased use of fillers is associated with cognitive load and may indicate early cognitive decline."
            },
            "word_recall_issues": {
                "description": "Word substitutions and semantic errors",
                "insight": "Word substitutions may indicate lexical access problems.",
                "clinical_relevance": "Increased word substitution rate is associated with semantic memory impairment."
            },
            "sentence_completion": {
                "description": "Ability to complete grammatical sentences",
                "insight": "Incomplete sentences may indicate difficulties with working memory and language planning.",
                "clinical_relevance": "Higher rates of incomplete sentences are associated with executive function deficits."
            },
            "lexical_diversity": {
                "description": "Variety of words used (unique word ratio)",
                "insight": "Measures vocabulary richness and access to semantic networks.",
                "clinical_relevance": "Reduced lexical diversity can be an early indicator of semantic memory decline."
            }
        },
        "comparative_insights": {
            "insight": "Analysis across multiple samples provides more reliable assessment than single recordings.",
            "recommendation": "Regular longitudinal monitoring can establish personal baselines and detect subtle changes over time."
        }
    }
    
    # Add file-specific insights if multiple files were analyzed
    if analysis_results.get("files_analyzed", 0) > 1:
        insights["multiple_sample_analysis"] = {
            "insight": "Multiple samples enable pattern recognition across different speech contexts.",
            "clinical_relevance": "Variability across samples may indicate context-dependent cognitive challenges."
        }
        
        # Add cluster analysis insights if available
        if analysis_results.get("comparative_analysis", {}).get("insights", {}).get("clusters"):
            cluster_insights = analysis_results["comparative_analysis"]["insights"]["clusters"]
            insights["cluster_analysis"] = {
                "description": "Patterns identified across multiple speech samples",
                "insight": "Clustering reveals natural groupings in speech patterns.",
                "findings": {f"Cluster {k.split('_')[1]}": v.get("interpretation", "") 
                            for k, v in cluster_insights.items() if "interpretation" in v}
            }
    
    return insights

def describe_ml_methods():
    """Describe the ML methods used in the analysis and their rationale."""
    
    methods = {
        "feature_extraction": {
            "description": "Acoustic and linguistic feature extraction from speech",
            "rationale": "Transforms raw audio and text into quantifiable metrics that can be analyzed for patterns associated with cognitive function.",
            "implementation": "Using signal processing (librosa) for acoustic features and NLP techniques for linguistic analysis."
        },
        "pause_analysis": {
            "description": "Detection and analysis of pauses in speech",
            "rationale": "Pauses are important indicators of cognitive processing time and word-finding difficulties.",
            "implementation": "Energy-based thresholding to identify silent periods in speech."
        },
        "hesitation_detection": {
            "description": "Identification of filler words and hesitation markers",
            "rationale": "Hesitations often indicate word retrieval problems and cognitive load.",
            "implementation": "NLP-based detection of common hesitation markers with contextual analysis."
        },
        "word_recall_assessment": {
            "description": "Detection of word substitution and semantic errors",
            "rationale": "Word substitutions may indicate lexical access problems common in cognitive decline.",
            "implementation": "Semantic similarity analysis using WordNet to identify unusual word combinations."
        },
        "dimensionality_reduction": {
            "description": "Principal Component Analysis (PCA)",
            "rationale": "Makes complex multidimensional data interpretable while preserving important variance relationships.",
            "implementation": "Scikit-learn PCA implementation with automatic component selection."
        },
        "unsupervised_clustering": {
            "description": "K-means clustering",
            "rationale": "Identifies natural groupings in speech patterns without predefined labels.",
            "implementation": "K-means with silhouette analysis for optimal cluster determination."
        },
        "anomaly_detection": {
            "description": "Statistical outlier detection",
            "rationale": "Identifies unusual speech patterns that deviate from established norms.",
            "implementation": "Combination of Isolation Forest and z-score based outlier detection."
        }
    }
    
    return methods

def recommend_next_steps():
    """Provide recommendations for improving clinical robustness."""
    
    recommendations = {
        "validation_studies": {
            "description": "Conduct validation studies with clinical populations",
            "rationale": "Establish correlations between speech features and standardized clinical assessments.",
            "implementation": "Partner with clinical institutions to collect data with standardized protocols."
        },
        "longitudinal_tracking": {
            "description": "Implement systems for tracking changes in speech patterns over time",
            "rationale": "Intra-individual changes may be more sensitive than single-timepoint comparisons.",
            "implementation": "Develop secure data storage and visualization tools for longitudinal monitoring."
        },
        "task_optimization": {
            "description": "Develop standardized speech elicitation tasks",
            "rationale": "Different tasks (picture description, story recall, sentence completion) may be sensitive to different aspects of cognitive function.",
            "implementation": "Create a battery of tasks targeting memory, executive function, and language domains."
        },
        "multimodal_integration": {
            "description": "Integrate speech analysis with other biomarkers",
            "rationale": "Combined approaches show improved diagnostic accuracy through data integration.",
            "implementation": "Develop APIs for secure integration with other assessment systems."
        },
        "personalized_baselines": {
            "description": "Establish individual baseline speech patterns",
            "rationale": "Individual differences in speech can mask clinically significant changes when using population norms.",
            "implementation": "Collect multiple samples per individual to establish reliable personal reference points."
        },
        "clinical_decision_support": {
            "description": "Develop interpretable clinical decision support tools",
            "rationale": "Clinicians need clear, actionable insights rather than raw data.",
            "implementation": "Create visualization dashboards with clear risk indicators and longitudinal tracking."
        },
        "privacy_protection": {
            "description": "Implement robust privacy protections",
            "rationale": "Speech data contains sensitive personal information.",
            "implementation": "Develop secure storage, anonymization techniques, and clear consent processes."
        }
    }
    
    return recommendations