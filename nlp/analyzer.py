"""
Linguistic Analysis Module for NLP Academic Demonstration.

Implements core NLP tasks:
1. Sentence Segmentation (Sentence Tokenization)
2. Word & Punctuation Tokenization
3. Part-of-Speech (POS) Tagging (Penn Treebank tagset)
4. Lemmatization using WordNet
5. Lexical Diversity & Syntactic Metrics
Supports serverless deployment environments (e.g. Vercel / AWS Lambda).
"""

import os
import re
import string
import tempfile
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

# Configure serverless-safe NLTK data directory
NLTK_DATA_DIR = os.path.join(tempfile.gettempdir(), 'nltk_data')
try:
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_DIR)
except Exception:
    pass

_NLTK_DOWNLOADED = False

def ensure_nltk_corpora():
    """Silently ensures essential NLTK data is available in the writable temp directory."""
    global _NLTK_DOWNLOADED
    if _NLTK_DOWNLOADED:
        return
    packages = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'wordnet', 'omw-1.4']
    for pkg in packages:
        try:
            nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)
        except Exception:
            pass
    _NLTK_DOWNLOADED = True

# Try initializing corpora on import
try:
    ensure_nltk_corpora()
except Exception:
    pass

# Initialize lemmatizer singleton safely
try:
    lemmatizer = WordNetLemmatizer()
except Exception:
    lemmatizer = None

# Penn Treebank POS tag descriptions and category groupings
POS_TAG_MAP = {
    'CC': {'desc': 'Coordinating conjunction', 'cat': 'Conjunction', 'color': 'tag-conj'},
    'CD': {'desc': 'Cardinal number', 'cat': 'Numeral', 'color': 'tag-num'},
    'DT': {'desc': 'Determiner / Article', 'cat': 'Determiner', 'color': 'tag-det'},
    'EX': {'desc': 'Existential there', 'cat': 'Pronoun', 'color': 'tag-pron'},
    'FW': {'desc': 'Foreign word', 'cat': 'Other', 'color': 'tag-other'},
    'IN': {'desc': 'Preposition / Subordinating conjunction', 'cat': 'Preposition', 'color': 'tag-prep'},
    'JJ': {'desc': 'Adjective (base)', 'cat': 'Adjective', 'color': 'tag-adj'},
    'JJR': {'desc': 'Adjective, comparative', 'cat': 'Adjective', 'color': 'tag-adj'},
    'JJS': {'desc': 'Adjective, superlative', 'cat': 'Adjective', 'color': 'tag-adj'},
    'LS': {'desc': 'List item marker', 'cat': 'Other', 'color': 'tag-other'},
    'MD': {'desc': 'Modal auxiliary verb', 'cat': 'Verb', 'color': 'tag-verb'},
    'NN': {'desc': 'Noun, singular or mass', 'cat': 'Noun', 'color': 'tag-noun'},
    'NNS': {'desc': 'Noun, plural', 'cat': 'Noun', 'color': 'tag-noun'},
    'NNP': {'desc': 'Proper noun, singular', 'cat': 'Noun', 'color': 'tag-noun'},
    'NNPS': {'desc': 'Proper noun, plural', 'cat': 'Noun', 'color': 'tag-noun'},
    'PDT': {'desc': 'Predeterminer', 'cat': 'Determiner', 'color': 'tag-det'},
    'POS': {'desc': 'Possessive ending', 'cat': 'Punctuation', 'color': 'tag-punct'},
    'PRP': {'desc': 'Personal pronoun', 'cat': 'Pronoun', 'color': 'tag-pron'},
    'PRP$': {'desc': 'Possessive pronoun', 'cat': 'Pronoun', 'color': 'tag-pron'},
    'RB': {'desc': 'Adverb (base)', 'cat': 'Adverb', 'color': 'tag-adv'},
    'RBR': {'desc': 'Adverb, comparative', 'cat': 'Adverb', 'color': 'tag-adv'},
    'RBS': {'desc': 'Adverb, superlative', 'cat': 'Adverb', 'color': 'tag-adv'},
    'RP': {'desc': 'Particle', 'cat': 'Other', 'color': 'tag-other'},
    'SYM': {'desc': 'Symbol', 'cat': 'Other', 'color': 'tag-other'},
    'TO': {'desc': "to' marker (infinitive/preposition)", 'cat': 'Preposition', 'color': 'tag-prep'},
    'UH': {'desc': 'Interjection', 'cat': 'Other', 'color': 'tag-other'},
    'VB': {'desc': 'Verb, base form (infinitive)', 'cat': 'Verb', 'color': 'tag-verb'},
    'VBD': {'desc': 'Verb, past tense', 'cat': 'Verb', 'color': 'tag-verb'},
    'VBG': {'desc': 'Verb, gerund or present participle', 'cat': 'Verb', 'color': 'tag-verb'},
    'VBN': {'desc': 'Verb, past participle', 'cat': 'Verb', 'color': 'tag-verb'},
    'VBP': {'desc': 'Verb, non-3rd person singular present', 'cat': 'Verb', 'color': 'tag-verb'},
    'VBZ': {'desc': 'Verb, 3rd person singular present', 'cat': 'Verb', 'color': 'tag-verb'},
    'WDT': {'desc': 'Wh-determiner', 'cat': 'Determiner', 'color': 'tag-det'},
    'WP': {'desc': 'Wh-pronoun', 'cat': 'Pronoun', 'color': 'tag-pron'},
    'WP$': {'desc': 'Possessive wh-pronoun', 'cat': 'Pronoun', 'color': 'tag-pron'},
    'WRB': {'desc': 'Wh-adverb', 'cat': 'Adverb', 'color': 'tag-adv'},
    '.': {'desc': 'Sentence-final punctuation', 'cat': 'Punctuation', 'color': 'tag-punct'},
    ',': {'desc': 'Comma', 'cat': 'Punctuation', 'color': 'tag-punct'},
    ':': {'desc': 'Colon or ellipsis', 'cat': 'Punctuation', 'color': 'tag-punct'},
    '$': {'desc': 'Currency sign', 'cat': 'Symbol', 'color': 'tag-other'},
    '#': {'desc': 'Pound sign / hashtag', 'cat': 'Symbol', 'color': 'tag-other'},
    "''": {'desc': 'Closing quotation mark', 'cat': 'Punctuation', 'color': 'tag-punct'},
    '``': {'desc': 'Opening quotation mark', 'cat': 'Punctuation', 'color': 'tag-punct'},
    '(': {'desc': 'Opening parenthesis/bracket', 'cat': 'Punctuation', 'color': 'tag-punct'},
    ')': {'desc': 'Closing parenthesis/bracket', 'cat': 'Punctuation', 'color': 'tag-punct'}
}


def get_pos_explanation(tag: str) -> dict:
    """Returns human-friendly explanation, general category, and UI styling badge for a POS tag."""
    return POS_TAG_MAP.get(tag, {
        'desc': f'Part-of-speech tag ({tag})',
        'cat': 'Other',
        'color': 'tag-other'
    })


def penn_to_wordnet_pos(penn_tag: str):
    """Maps Penn Treebank POS tag to WordNet POS tag for lemmatization."""
    try:
        from nltk.corpus import wordnet
        if penn_tag.startswith('J'):
            return wordnet.ADJ
        elif penn_tag.startswith('V'):
            return wordnet.VERB
        elif penn_tag.startswith('N'):
            return wordnet.NOUN
        elif penn_tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN
    except Exception:
        return 'n'


def lemmatize_word(word: str, penn_tag: str) -> str:
    """Lemmatizes a single word using WordNet based on its POS tag."""
    if not lemmatizer:
        return word.lower()
    wn_tag = penn_to_wordnet_pos(penn_tag)
    try:
        lemma = lemmatizer.lemmatize(word.lower(), pos=wn_tag)
        if word.istitle():
            return lemma.capitalize()
        return lemma
    except Exception:
        return word.lower()


def _fallback_pos_tag(tokens: list) -> list:
    """Lightweight rule-based POS tagger used if NLTK models are missing in cloud sandbox."""
    tags = []
    pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'this', 'that'}
    verbs = {'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'go', 'goes', 'went', 'like', 'likes', 'play', 'plays', 'playing'}
    determiners = {'a', 'an', 'the', 'some', 'any', 'every', 'each'}
    prepositions = {'in', 'on', 'at', 'to', 'for', 'with', 'from', 'by', 'about', 'over'}
    wh_words = {'what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how'}

    for t in tokens:
        tl = t.lower()
        if t in string.punctuation:
            tags.append((t, '.' if t in '.!?' else ',' if t == ',' else ':'))
        elif tl in wh_words:
            tags.append((t, 'WRB' if tl in {'where', 'when', 'why', 'how'} else 'WP'))
        elif tl in pronouns:
            tags.append((t, 'PRP'))
        elif tl in determiners:
            tags.append((t, 'DT'))
        elif tl in verbs:
            tags.append((t, 'VBZ' if tl.endswith('s') else 'VBD' if tl in {'was', 'were', 'had', 'did', 'went'} else 'VBP'))
        elif tl in prepositions:
            tags.append((t, 'IN'))
        elif t.isdigit():
            tags.append((t, 'CD'))
        elif t[0].isupper() and len(tags) > 0 and tags[-1][1] != '.':
            tags.append((t, 'NNP'))
        else:
            tags.append((t, 'NN'))
    return tags


def analyze_text(text: str) -> dict:
    """
    Performs comprehensive NLP linguistic analysis on the provided text.
    
    Tasks performed:
    1. Sentence Tokenization
    2. Word Tokenization
    3. Part-of-Speech Tagging
    4. Lemmatization
    5. Structural & Lexical Metrics
    
    Returns:
        dict: Complete NLP linguistic breakdown
    """
    if not text or not text.strip():
        return {
            'sentences': [],
            'sentence_count': 0,
            'word_count': 0,
            'token_count': 0,
            'unique_words': 0,
            'lexical_diversity': 0.0,
            'avg_sentence_length': 0.0,
            'tokens': []
        }

    raw_text = text.strip()

    # Step 1: Sentence Tokenization
    try:
        sentences = sent_tokenize(raw_text)
    except Exception:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw_text) if s.strip()]
        if not sentences:
            sentences = [raw_text]

    # Step 2 & 3: Word Tokenization and POS Tagging
    try:
        tokens_raw = word_tokenize(raw_text)
    except Exception:
        tokens_raw = re.findall(r"\w+(?:'\w+)?|[^\w\s]", raw_text)

    try:
        pos_tags = pos_tag(tokens_raw)
    except Exception:
        pos_tags = _fallback_pos_tag(tokens_raw)

    # Step 4: Token Detail Construction & Lemmatization
    analyzed_tokens = []
    pure_words = []
    lemmas_list = []

    for idx, (token, tag) in enumerate(pos_tags):
        is_punct = token in string.punctuation or tag in ['.', ',', ':', "''", '``', '(', ')']
        is_alpha = token.isalpha()
        tag_info = get_pos_explanation(tag)
        lemma = lemmatize_word(token, tag) if is_alpha else token

        if is_alpha:
            pure_words.append(token.lower())
            lemmas_list.append(lemma.lower())

        analyzed_tokens.append({
            'index': idx + 1,
            'token': token,
            'tag': tag,
            'category': tag_info['cat'],
            'description': tag_info['desc'],
            'color_class': tag_info['color'],
            'lemma': lemma,
            'is_punctuation': is_punct,
            'is_alpha': is_alpha
        })

    # Step 5: Syntactic and Lexical Metrics
    total_words = len(pure_words)
    unique_words = len(set(pure_words))
    lexical_diversity = round((unique_words / total_words) * 100, 1) if total_words > 0 else 0.0
    sentence_count = len(sentences) if sentences else 1
    avg_sentence_length = round(total_words / sentence_count, 1) if sentence_count > 0 else 0.0

    return {
        'sentence_count': len(sentences),
        'sentences': sentences,
        'word_count': total_words,
        'token_count': len(analyzed_tokens),
        'unique_words': unique_words,
        'lexical_diversity': lexical_diversity,
        'avg_sentence_length': avg_sentence_length,
        'tokens': analyzed_tokens
    }
