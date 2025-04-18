import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import base64
import io
import numpy as np
import pandas as pd
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import processing functions using absolute imports
from audio_processing.audio_loader import load_audio_file
from audio_processing.feature_extraction import extract_audio_features, analyze_text_content
from audio_processing.speech_to_text import speech_to_text
from analysis.preprocessing import clean_and_normalize_data
from analysis.clustering import run_clustering, plot_feature_clusters
from analysis.risk_analysis import run_risk_analysis, plot_risk_visualization
from analysis.insights import get_speech_insights
from utils.json_encoder import make_json_serializable, EnhancedJSONResponse

app = FastAPI(
    title="Cognitive Speech Analysis API",
    description="API for analyzing cognitive stress patterns in speech",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Home endpoint that serves the HTML interface."""
    try:
        with open("static/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
            <body>
                <h1>Welcome to the Cognitive Speech Analysis API</h1>
                <p>Use /analyze-audio/ to upload and analyze audio files.</p>
            </body>
        </html>
        """

@app.post("/analyze-audio/")
async def analyze_audio(files: List[UploadFile] = File(...)):
    """
    Analyze one or multiple audio files for cognitive stress patterns.
    Returns extracted features and analysis results.
    """
    try:
        # Input validation
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Validate file types and sizes
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        for file in files:
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in ['.wav', '.mp3', '.ogg', '.flac']:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file format: {file_ext}. Please use WAV, MP3, OGG, or FLAC."
                )
            
            # Check file size (approximate)
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} exceeds the maximum size of 10MB"
                )
            # Reset file position after reading
            await file.seek(0)
        
        # Process each file
        audio_features = {}
        for file in files:
            logger.info(f"Processing file: {file.filename}")
            
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                temp_file_path = temp_file.name
                content = await file.read()
                temp_file.write(content)
            
            try:
                # Process the audio file
                audio, sample_rate = load_audio_file(temp_file_path)
                if audio is None:
                    logger.warning(f"Failed to load audio from {file.filename}")
                    continue
                
                # Extract audio features
                acoustic_features = extract_audio_features(audio, sample_rate, file.filename)
                if not acoustic_features:
                    acoustic_features = {}
                audio_features[file.filename] = acoustic_features
                
                # Extract text and linguistic features
                try:
                    text = speech_to_text(temp_file_path)
                    linguistic_features = analyze_text_content(text, file.filename)
                    audio_features[file.filename].update(linguistic_features)
                    audio_features[file.filename]["transcribed_text"] = text
                except Exception as e:
                    logger.error(f"Error with speech-to-text: {str(e)}")
                    audio_features[file.filename]["transcribed_text"] = ""
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                traceback.print_exc()
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.error(f"Error deleting temp file: {str(e)}")
        
        if not audio_features:
            raise HTTPException(status_code=400, detail="Failed to process any audio files")
        
        # Convert features to DataFrame for analysis
        try:
            features_df = pd.DataFrame.from_dict(audio_features, orient='index')
            # Ensure all columns are numeric (except transcribed_text)
            for column in features_df.columns:
                if column != 'transcribed_text':
                    features_df[column] = pd.to_numeric(features_df[column], errors='coerce')
                    
            # Remove the text column for numerical analysis
            analysis_df = features_df.drop('transcribed_text', axis=1, errors='ignore')
            feature_names = analysis_df.columns.tolist()
            feature_matrix = analysis_df.values
            
            # Clean and normalize data
            normalized_features = clean_and_normalize_data(feature_matrix)
            
            # Run clustering
            clusters, pca_results = run_clustering(normalized_features)
            
            # Run risk analysis
            risk_indices = run_risk_analysis(normalized_features, feature_names, audio_features)
            
            # Build file details for response - moved this section earlier
            file_details = {}
            for filename, features in audio_features.items():
                # Calculate stress indicators for each file
                stress_indicators = {
                    "speech_rate_indicator": "elevated" if features.get('speech_rate', 0) > 0.15 else "normal",
                    "pitch_variability": "high" if features.get('pitch_std', 0) > 500 else "normal",
                    "energy_level": "high" if features.get('energy_mean', 0) > 0.1 else "normal",
                    "hesitation_level": "high" if features.get('hesitation_count', 0) >= 3 else "low"
                }
                
                # Estimate overall stress level
                stress_count = sum(1 for v in stress_indicators.values() if v in ["elevated", "high"])
                stress_assessment = "high" if stress_count >= 3 else "moderate" if stress_count >= 1 else "low"
                
                # Get transcription or empty string
                transcription = features.get("transcribed_text", "")
                
                # Remove transcribed_text from features dict for JSON serialization
                features_copy = {k: v for k, v in features.items() if k != "transcribed_text"}
                
                file_details[filename] = {
                    "transcribed_text": transcription,
                    "features": features_copy,
                    "stress_indicators": stress_indicators,
                    "stress_assessment": stress_assessment
                }
            
            # Generate insights with report - now file_details is defined before this call
            insights, report = get_speech_insights(
                analysis_df, 
                clusters, 
                risk_indices,
                generate_report=True,
                file_details=file_details
            )
            
            # Generate visualizations
            cluster_plot = plot_feature_clusters(pca_results, clusters, list(audio_features.keys()))
            risk_plot = plot_risk_visualization(normalized_features, risk_indices, feature_names, audio_features)
            
            # Convert plots to base64 for JSON response
            cluster_plot_base64 = base64.b64encode(cluster_plot.read()).decode('utf-8')
            risk_plot_base64 = base64.b64encode(risk_plot.read()).decode('utf-8')
            
            # Build response with individual file analyses and comparative results
            response = {
                "files_analyzed": len(files),
                "file_details": file_details,
                "comparative_analysis": {
                    "clusters": clusters.tolist() if hasattr(clusters, 'tolist') else clusters,
                    "insights": insights,
                    "visualizations": {
                        "cluster_plot": cluster_plot_base64,
                        "risk_plot": risk_plot_base64
                    }
                },
                "report": report
            }
            
            # Return with enhanced JSON response that handles NaN/Inf
            return EnhancedJSONResponse(content=response)
        except Exception as analysis_error:
            logger.error(f"Error during analysis: {str(analysis_error)}")
            traceback.print_exc()
            return EnhancedJSONResponse(
                status_code=500,
                content={"error": f"Analysis failed: {str(analysis_error)}"}
            )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as ve:
        # Handle validation errors
        logger.error(f"Validation error: {str(ve)}")
        return EnhancedJSONResponse(
            status_code=400,
            content={"error": f"Invalid input: {str(ve)}"}
        )
    except IOError as io_err:
        # Handle file I/O errors
        logger.error(f"I/O error: {str(io_err)}")
        return EnhancedJSONResponse(
            status_code=500,
            content={"error": f"File processing error: {str(io_err)}"}
        )
    except Exception as e:
        # Log the full error for debugging
        logger.error(f"Error processing request: {str(e)}")
        traceback.print_exc()
        return EnhancedJSONResponse(
            status_code=500,
            content={"error": f"Failed to analyze audio: {str(e)}"}
        )   