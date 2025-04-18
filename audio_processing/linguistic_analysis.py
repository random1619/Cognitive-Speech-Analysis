#   audio_processing/linguistic_analysis.py
import re
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet, stopwords

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Common English hesitation markers and filler words
HESITATION_MARKERS = set(['um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean', 'sort of', 'kind of'])

# Common English function words
FUNCTION_WORDS = set(stopwords.words('english'))

def analyze_linguistic_features(text):
    """
    Analyze linguistic features related to cognitive function.
    
    Features include:
    - Pauses per sentence (estimated from punctuation)
    - Hesitation markers
    - Word recall issues (estimated from word substitution patterns)
    - Word-finding difficulties
    - Sentence completion issues
    
    Args:
        text: Transcribed speech text
        
    Returns:
        Dictionary of linguistic features
    """
    if not text or len(text.strip()) == 0:
        # Return default values if no text
        return {
            "pause_count": 0,
            "pause_per_sentence": 0,
            "hesitation_count": 0,
            "hesitation_rate": 0,
            "word_substitution_count": 0,
            "word_substitution_rate": 0,
            "incomplete_sentences": 0,
            "incomplete_sentence_rate": 0,
            "word_finding_difficulty": 0,
            "sentence_complexity": 0
        }
    
    try:
        # Tokenize text
        sentences = sent_tokenize(text.lower())
        words = word_tokenize(text.lower())
        
        # Filter out punctuation
        words = [word for word in words if word.isalpha()]
        
        # Count pauses (estimated from ellipses, commas, etc.)
        pause_markers = re.findall(r'\.{2,}|,|\.\.\.|—|–|-', text)
        pause_count = len(pause_markers)
        pause_per_sentence = pause_count / max(1, len(sentences))
        
        # Count hesitation markers
        hesitation_count = sum(1 for word in words if word in HESITATION_MARKERS)
        hesitation_rate = hesitation_count / max(1, len(words))
        
        # Estimate word substitution (word recall issues)
        # This is a simplified approximation - in a real system, you'd use more sophisticated NLP
        content_words = [w for w in words if w not in FUNCTION_WORDS and len(w) > 2]
        word_substitution_count = 0
        
        # Check for semantically unusual word combinations
        if len(content_words) > 3:
            for i in range(len(content_words) - 1):
                word1 = content_words[i]
                word2 = content_words[i+1]
                
                # Check if words are semantically related using WordNet
                word1_synsets = wordnet.synsets(word1)
                word2_synsets = wordnet.synsets(word2)
                
                if word1_synsets and word2_synsets:
                    # Calculate semantic similarity
                    max_similarity = 0
                    for s1 in word1_synsets:
                        for s2 in word2_synsets:
                            similarity = s1.path_similarity(s2)
                            if similarity and similarity > max_similarity:
                                max_similarity = similarity
                    
                    # Very low similarity might indicate word substitution
                    if max_similarity < 0.1:
                        word_substitution_count += 1
        
        word_substitution_rate = word_substitution_count / max(1, len(content_words) - 1)
        
        # Check for incomplete sentences
        incomplete_sentences = 0
        for sentence in sentences:
            # Sentences ending with ellipsis or without proper ending punctuation
            if sentence.strip().endswith(('...', '..', '..')):
                incomplete_sentences += 1
            # Sentences that are very short might be incomplete
            elif len(word_tokenize(sentence)) < 3:
                incomplete_sentences += 1
        
        incomplete_sentence_rate = incomplete_sentences / max(1, len(sentences))
        
        # Estimate word finding difficulty based on pauses and hesitations
        word_finding_difficulty = (hesitation_rate + pause_per_sentence) / 2
        
        # Calculate sentence complexity (average words per sentence)
        words_per_sentence = len(words) / max(1, len(sentences))
        sentence_complexity = min(1.0, words_per_sentence / 20)  # Normalize to 0-1 range
        
        return {
            "pause_count": pause_count,
            "pause_per_sentence": round(pause_per_sentence, 3),
            "hesitation_count": hesitation_count,
            "hesitation_rate": round(hesitation_rate, 3),
            "word_substitution_count": word_substitution_count,
            "word_substitution_rate": round(word_substitution_rate, 3),
            "incomplete_sentences": incomplete_sentences,
            "incomplete_sentence_rate": round(incomplete_sentence_rate, 3),
            "word_finding_difficulty": round(word_finding_difficulty, 3),
            "sentence_complexity": round(sentence_complexity, 3)
        }
    except Exception as e:
        print(f"Error in linguistic analysis: {str(e)}")
        # Return default values if analysis fails
        return {
            "pause_count": 0,
            "pause_per_sentence": 0,
            "hesitation_count": 0,
            "hesitation_rate": 0,
            "word_substitution_count": 0,
            "word_substitution_rate": 0,
            "incomplete_sentences": 0,
            "incomplete_sentence_rate": 0,
            "word_finding_difficulty": 0,
            "sentence_complexity": 0
        }