"""
NLP Package for Grammar Error Detection, Correction, and Linguistic Analysis.
"""
from .preprocessing import preprocess_text, clean_whitespace
from .analyzer import analyze_text, get_pos_explanation
from .grammar_checker import check_grammar, correct_text

__all__ = [
    'preprocess_text',
    'clean_whitespace',
    'analyze_text',
    'get_pos_explanation',
    'check_grammar',
    'correct_text'
]
