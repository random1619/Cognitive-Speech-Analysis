import speech_recognition as sr
import os
import logging
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def speech_to_text(audio_file, language="en-US"):
    """
    Convert speech in audio file to text with improved format handling.
    """
    speech_recognizer = sr.Recognizer()
    
    # Get file extension
    file_ext = os.path.splitext(audio_file)[1].lower()
    
    # Create a temporary file for format conversion if needed
    temp_file = None
    file_to_process = audio_file
    
    try:
        # Convert non-supported formats to WAV
        if file_ext not in ['.wav', '.aiff', '.aif', '.flac']:
            try:
                logger.info(f"Converting {file_ext} file to WAV format")
                temp_file = audio_file + ".temp.wav"
                
                # Use pydub to convert the file
                audio = AudioSegment.from_file(audio_file)
                audio.export(temp_file, format="wav")
                file_to_process = temp_file
            except Exception as e:
                logger.error(f"Error converting audio format: {str(e)}")
                return ""
        
        # Process the audio file
        with sr.AudioFile(file_to_process) as source:
            # Adjust for ambient noise
            speech_recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = speech_recognizer.record(source)
            
            # Try Google Speech Recognition
            try:
                text = speech_recognizer.recognize_google(audio, language=language)
                return text
            except sr.RequestError:
                logger.warning("Google Speech API failed, falling back to Sphinx")
                try:
                    text = speech_recognizer.recognize_sphinx(audio)
                    return text
                except Exception as sphinx_error:
                    logger.error(f"Sphinx recognition failed: {str(sphinx_error)}")
                    return ""
            except Exception as google_error:
                logger.error(f"Google recognition failed: {str(google_error)}")
                return ""
    except Exception as e:
        logger.error(f"Speech recognition failed: {str(e)}")
        return ""
    finally:
        # Clean up temporary file if created
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
