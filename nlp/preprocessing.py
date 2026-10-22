"""
Text Preprocessing Module for NLP Grammar Error Detector.

This module provides preprocessing routines used before linguistic analysis
and grammar verification, including whitespace normalization, contraction
utilities, and boundary sanitization.
"""

import re
import unicodedata

# Common contraction mapping for linguistic analysis
CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "I would",
    "i'll": "I will",
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}


def clean_whitespace(text: str) -> str:
    """
    Normalizes consecutive whitespace while retaining sentence structure.
    Converts various unicode spaces to standard ASCII spaces.
    """
    if not text:
        return ""
    # Normalize unicode characters (NFKC)
    normalized = unicodedata.normalize('NFKC', text)
    # Replace multiple horizontal spaces/tabs with single space
    cleaned = re.sub(r'[ \t]+', ' ', normalized)
    # Remove carriage returns
    cleaned = cleaned.replace('\r', '')
    # Strip leading and trailing whitespace
    return cleaned.strip()


def expand_contractions(text: str) -> str:
    """
    Expands common English contractions for semantic POS and lemma inspection.
    E.g. "don't" -> "do not", "I'm" -> "I am"
    """
    if not text:
        return ""
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in CONTRACTIONS.keys()) + r')\b', re.IGNORECASE)
    
    def replace_match(match):
        word = match.group(0)
        lower_word = word.lower()
        expanded = CONTRACTIONS.get(lower_word, word)
        # Preserve original capitalization
        if word.istitle():
            return expanded.capitalize()
        elif word.isupper():
            return expanded.upper()
        return expanded

    return pattern.sub(replace_match, text)


def preprocess_text(text: str) -> dict:
    """
    Executes core text preprocessing pipeline on raw user input.
    
    Returns:
        dict: Preprocessing report with cleaned text, raw text, and token count estimates.
    """
    raw = text or ""
    cleaned = clean_whitespace(raw)
    
    return {
        'original_text': raw,
        'cleaned_text': cleaned,
        'char_count': len(cleaned),
        'is_empty': len(cleaned) == 0,
        'has_special_chars': bool(re.search(r'[^\w\s.,!?;:\'\"-]', cleaned))
    }
