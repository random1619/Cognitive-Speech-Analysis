#   audio_processing/feature_extraction.py
import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

def extract_audio_features(audio, sample_rate, filename):
    """Extract acoustic features from audio signal."""
    features = {}
    
    try:
        # Basic features
        features['duration'] = librosa.get_duration(y=audio, sr=sample_rate)
        features['energy_mean'] = np.mean(np.abs(audio))
        features['energy_std'] = np.std(np.abs(audio))
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
        pitches = pitches[pitches > 0]
        if len(pitches) > 0:
            features['pitch_mean'] = np.mean(pitches)
            features['pitch_std'] = np.std(pitches)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
        
        # Speech rate approximation using zero crossings
        # Fix: Remove 'frame_length' parameter as it's not supported in current librosa
        try:
            zero_crossings = librosa.zero_crossings(audio, pad=False)
            features['speech_rate'] = sum(zero_crossings) / len(audio)
        except Exception as e:
            logger.warning(f"Error calculating speech rate: {str(e)}")
            features['speech_rate'] = 0
        
        # MFCCs for voice quality
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        for i, mfcc in enumerate(mfccs):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfcc)
            features[f'mfcc_{i+1}_std'] = np.std(mfcc)
        
        # Hesitation detection (estimate based on silence periods)
        # Define a low energy threshold
        threshold = 0.01
        frame_length = 2048  # ~100ms at 22050Hz
        hop_length = 512     # ~25ms hop
        
        # Get energy per frame
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Find silence frames (potential hesitations)
        silent_frames = np.where(energy < threshold)[0]
        
        # Count silence segments (groups of consecutive silent frames)
        if len(silent_frames) > 0:
            # Convert to boolean array of silence (1 = silent)
            silence = np.zeros_like(energy, dtype=bool)
            silence[silent_frames] = True
            
            # Count the number of transitions from speaking to silence
            transitions = np.diff(silence.astype(int))
            hesitation_count = np.sum(transitions == 1)
            
            features['hesitation_count'] = hesitation_count
            features['silence_ratio'] = len(silent_frames) / len(energy)
        else:
            features['hesitation_count'] = 0
            features['silence_ratio'] = 0
        
    except Exception as e:
        logger.error(f"Error extracting audio features: {str(e)}")
    
    return features

def analyze_text_content(text, filename):
    """Analyze transcribed text for linguistic features."""
    features = {}
    
    if not text:
        return features
    
    try:
        # Basic text features
        words = text.split()
        features['word_count'] = len(words)
        features['avg_word_length'] = sum(len(word) for word in words) / max(1, len(words))
        
        # Analyze filler words
        filler_words = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally']
        filler_count = sum(text.lower().count(word) for word in filler_words)
        features['filler_ratio'] = filler_count / max(1, len(words))
        
        # Analyze sentence structure
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            features['sentence_count'] = len(sentences)
            features['avg_sentence_length'] = sum(len(s.split()) for s in sentences) / len(sentences)
        else:
            features['sentence_count'] = 0
            features['avg_sentence_length'] = 0
        
    except Exception as e:
        logger.error(f"Error extracting text features: {str(e)}")
    
    return features