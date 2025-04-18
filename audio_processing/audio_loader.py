#   audio_processing/audio_loader.py
import os
import librosa
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_audio_file(file_path):
    """Load an audio file using librosa with improved error handling."""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return None, None
            
        # Get file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        supported_formats = ['.wav', '.mp3', '.ogg', '.flac']
        
        if file_ext not in supported_formats:
            logger.error(f"Unsupported audio format: {file_ext}. Supported formats: {', '.join(supported_formats)}")
            return None, None
        
        # Load audio file
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        
        # Validate audio data
        if len(audio_data) == 0:
            logger.error(f"Empty audio file: {file_path}")
            return None, None
            
        # Handle potential NaN values
        audio_data = np.nan_to_num(audio_data)
        
        return audio_data, sample_rate
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return None, None