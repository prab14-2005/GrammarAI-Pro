"""
Grammar Error Detection and Correction Engine.

Combines rule-based NLP algorithms with statistical/linguistic tools:
1. Custom syntactic rules (Subject-Verb Agreement, Subject-Auxiliary Inversion, Articles, Duplication)
2. Statistical / LanguageTool linguistic checker with fallback
3. Error classification (Grammar, Spelling, Punctuation, Style)
4. Automated text correction & interactive highlight mapping
5. Accuracy & linguistic quality metrics
"""

import re
import requests
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from .analyzer import analyze_text

# Public LanguageTool endpoint for advanced language model checks
LT_API_URL = "https://api.languagetool.org/v2/check"
REQUEST_TIMEOUT = 5.0  # seconds


# -------------------------------------------------------------
# Custom Rule-Based NLP Checks
# -------------------------------------------------------------

def check_subject_aux_inversion(text: str) -> list:
    """
    Detects missing subject-auxiliary inversion in direct questions.
    Example:
        'Where you are going?' -> 'Where are you going?'
        'What you are doing?' -> 'What are you doing?'
        'Why you are crying?' -> 'Why are you crying?'
    """
    errors = []
    # Pattern: Wh-word + Pronoun + Auxiliary + Verb/Rest + '?'
    # Matches: Where/What/Why/When/How + you/they/he/she/we + are/is/were/was/will/can/do/did/have/has
    wh_pattern = re.compile(
        r'\b(Where|What|Why|When|How)\s+(you|he|she|they|we)\s+(are|is|were|was|will|can|do|does|did|have|has)\b',
        re.IGNORECASE
    )

    for match in wh_pattern.finditer(text):
        wh_word = match.group(1)
        subject = match.group(2)
        aux = match.group(3)
        
        # Verify if sentence or clause ends with '?'
        end_idx = match.end()
        remaining_text = text[end_idx:]
        sentence_end = remaining_text.find('?')
        dot_end = remaining_text.find('.')
        
        # If there's a question mark before any period, it's a direct question
        is_question = sentence_end != -1 and (dot_end == -1 or sentence_end < dot_end)
        if not is_question and '?' not in remaining_text:
            # Also trigger if text is short and starts with Wh-word
            if len(text.strip().split()) <= 8:
                is_question = True

        if is_question:
            # Build correction: e.g. "Where are you"
            correction = f"{wh_word} {aux} {subject}"
            # Case preservation
            if match.group(0)[0].isupper():
                correction = correction[0].upper() + correction[1:]

            start = match.start()
            end = match.end()
            errors.append({
                'offset': start,
                'length': end - start,
                'error_text': text[start:end],
                'suggestions': [correction],
                'explanation': "In direct questions starting with a question word, the auxiliary verb must precede the subject (Subject-Auxiliary Inversion).",
                'category': 'grammar',
                'rule_id': 'SUBJECT_AUX_INVERSION'
            })

    return errors


def check_subject_verb_agreement(text: str) -> list:
    """
    Detects classic Subject-Verb Agreement (SVA) violations:
    1. 'He don't' -> 'He doesn't' / 'She don't' -> 'She doesn't' / 'It don't' -> 'It doesn't'
    2. 'I has' -> 'I have' / 'They has' -> 'They have' / 'We has' -> 'We have' / 'You has' -> 'You have'
    3. 'They was' -> 'They were' / 'We was' -> 'We were' / 'You was' -> 'You were'
    4. 'He were' -> 'He was' / 'She were' -> 'She was' / 'It were' -> 'It was'
    5. 'She go' -> 'She goes' / 'He walk' -> 'He walks'
    """
    errors = []

    # 1. Contraction SVA: "He don't", "She don't", "It don't"
    dont_pattern = re.compile(r'\b(he|she|it|this|that)\s+(don\'t|dont)\b', re.IGNORECASE)
    for match in dont_pattern.finditer(text):
        subj = match.group(1)
        err_word = match.group(2)
        start = match.start(2)
        length = len(err_word)
        replacement = "doesn't" if "'" in err_word else "doesnt"
        if err_word.isupper():
            replacement = replacement.upper()
        elif err_word[0].isupper():
            replacement = replacement.capitalize()

        errors.append({
            'offset': start,
            'length': length,
            'error_text': err_word,
            'suggestions': [replacement, "does not"],
            'explanation': f"Subject-verb agreement error: Third-person singular pronoun '{subj}' takes '{replacement}' instead of '{err_word}'.",
            'category': 'grammar',
            'rule_id': 'SVA_DONT_DOESNT'
        })

    # 2. Pronoun + 'has': "I has", "They has", "We has", "You has"
    has_pattern = re.compile(r'\b(I|they|we|you)\s+(has)\b', re.IGNORECASE)
    for match in has_pattern.finditer(text):
        subj = match.group(1)
        start = match.start(2)
        length = len(match.group(2))
        err_word = match.group(2)
        replacement = "HAVE" if err_word.isupper() else "have"
        errors.append({
            'offset': start,
            'length': length,
            'error_text': err_word,
            'suggestions': [replacement],
            'explanation': f"Subject-verb agreement error: Pronoun '{subj}' requires the plural/first-person auxiliary 'have', not 'has'.",
            'category': 'grammar',
            'rule_id': 'SVA_HAS_HAVE'
        })

    # 3. Pronoun + 'was': "They was", "We was", "You was"
    was_pattern = re.compile(r'\b(they|we|you)\s+(was)\b', re.IGNORECASE)
    for match in was_pattern.finditer(text):
        subj = match.group(1)
        start = match.start(2)
        length = len(match.group(2))
        err_word = match.group(2)
        replacement = "WERE" if err_word.isupper() else "were"
        errors.append({
            'offset': start,
            'length': length,
            'error_text': err_word,
            'suggestions': [replacement, "are"],
            'explanation': f"Subject-verb agreement error: Plural pronoun '{subj}' requires plural past-tense verb 'were' instead of 'was'.",
            'category': 'grammar',
            'rule_id': 'SVA_WAS_WERE'
        })

    # 4. Pronoun + 'were' (indicative): "He were", "She were", "It were"
    were_pattern = re.compile(r'\b(he|she|it)\s+(were)\b', re.IGNORECASE)
    for match in were_pattern.finditer(text):
        subj = match.group(1)
        # Avoid subjunctive like "if he were"
        pre_text = text[:match.start()].strip().lower()
        if not pre_text.endswith(('if', 'as if', 'wish', 'wishes', 'wished')):
            start = match.start(2)
            length = len(match.group(2))
            err_word = match.group(2)
            replacement = "WAS" if err_word.isupper() else "was"
            errors.append({
                'offset': start,
                'length': length,
                'error_text': err_word,
                'suggestions': [replacement, "is"],
                'explanation': f"Subject-verb agreement error: Singular pronoun '{subj}' requires singular past-tense verb 'was' instead of 'were'.",
                'category': 'grammar',
                'rule_id': 'SVA_WERE_WAS'
            })

    # 5. Base form verbs after 3rd person singular pronoun: "She go", "He walk", "She play", "He like"
    # Specific common base verbs
    common_base_verbs = {
        'go': 'goes',
        'do': 'does',
        'have': 'has',
        'like': 'likes',
        'play': 'plays',
        'walk': 'walks',
        'run': 'runs',
        'eat': 'eats',
        'know': 'knows',
        'say': 'says',
        'see': 'sees',
        'come': 'comes',
        'want': 'wants',
        'work': 'works',
        'read': 'reads',
        'write': 'writes'
    }
    base_verb_regex = r'\b(he|she|it)\s+(' + '|'.join(common_base_verbs.keys()) + r')\b'
    for match in re.finditer(base_verb_regex, text, re.IGNORECASE):
        subj = match.group(1)
        err_verb = match.group(2)
        start = match.start(2)
        length = len(err_verb)
        rep = common_base_verbs.get(err_verb.lower(), err_verb + 's')
        if err_verb.isupper():
            rep = rep.upper()
        elif err_verb.istitle():
            rep = rep.capitalize()
            
        errors.append({
            'offset': start,
            'length': length,
            'error_text': err_verb,
            'suggestions': [rep],
            'explanation': f"Subject-verb agreement error: Singular subject '{subj}' requires the 3rd person singular present tense verb '{rep}'.",
            'category': 'grammar',
            'rule_id': 'SVA_BASE_VERB'
        })

    return errors


def check_articles(text: str) -> list:
    """
    Detects incorrect usage of indefinite articles 'a' vs 'an'.
    E.g. 'a apple' -> 'an apple', 'an book' -> 'a book'
    """
    errors = []
    
    # 'a' before vowel sound (a, e, i, o, u)
    vowel_article_pattern = re.compile(r'\b(a)\s+([aeioAEIO][a-z]+)\b')
    for match in vowel_article_pattern.finditer(text):
        article = match.group(1)
        next_word = match.group(2)
        # Avoid common exceptions like "a European", "a union", "a one"
        if not next_word.lower().startswith(('eu', 'uni', 'one', 'use', 'user')):
            start = match.start(1)
            length = len(article)
            rep = "An" if article[0].isupper() else "an"
            errors.append({
                'offset': start,
                'length': length,
                'error_text': article,
                'suggestions': [rep],
                'explanation': f"Use indefinite article '{rep}' before words starting with a vowel sound ('{next_word}').",
                'category': 'grammar',
                'rule_id': 'EN_A_VS_AN'
            })

    # 'an' before typical consonant sound
    consonant_article_pattern = re.compile(r'\b(an)\s+([b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z][a-z]+)\b')
    for match in consonant_article_pattern.finditer(text):
        article = match.group(1)
        next_word = match.group(2)
        # Avoid silent 'h' words like "an hour", "an honest", "an honor"
        if not next_word.lower().startswith(('hour', 'honest', 'honor', 'heir')):
            start = match.start(1)
            length = len(article)
            rep = "A" if article[0].isupper() else "a"
            errors.append({
                'offset': start,
                'length': length,
                'error_text': article,
                'suggestions': [rep],
                'explanation': f"Use indefinite article '{rep}' before words starting with a consonant sound ('{next_word}').",
                'category': 'grammar',
                'rule_id': 'EN_AN_VS_A'
            })

    return errors


def check_word_duplication(text: str) -> list:
    """Detects accidentally repeated consecutive identical words (e.g. 'the the')."""
    errors = []
    dup_pattern = re.compile(r'\b([a-zA-Z]+)\s+\1\b', re.IGNORECASE)
    for match in dup_pattern.finditer(text):
        first_word = match.group(1)
        start = match.start()
        length = match.end() - match.start()
        errors.append({
            'offset': start,
            'length': length,
            'error_text': match.group(0),
            'suggestions': [first_word],
            'explanation': f"Duplicate word detected: '{first_word}' is repeated consecutively.",
            'category': 'style',
            'rule_id': 'REPEATED_WORD'
        })
    return errors


def check_capitalization_and_punctuation(text: str) -> list:
    """Detects lowercase starts of sentences and missing terminal punctuation."""
    errors = []
    # Capitalization at start of text
    first_char_match = re.search(r'^[a-z]', text.strip())
    if first_char_match:
        char = first_char_match.group(0)
        idx = text.find(char)
        errors.append({
            'offset': idx,
            'length': 1,
            'error_text': char,
            'suggestions': [char.upper()],
            'explanation': "Sentences should begin with an uppercase letter.",
            'category': 'punctuation',
            'rule_id': 'SENTENCE_START_UPPER'
        })

    # Missing terminal punctuation for sentences of reasonable length
    clean_stripped = text.strip()
    if clean_stripped and len(clean_stripped.split()) >= 3:
        if clean_stripped[-1] not in ['.', '!', '?']:
            errors.append({
                'offset': len(clean_stripped),
                'length': 0,
                'error_text': '',
                'suggestions': ['.', '?', '!'],
                'explanation': "The sentence appears to be missing terminal punctuation (period, exclamation, or question mark).",
                'category': 'punctuation',
                'rule_id': 'MISSING_TERMINAL_PUNCTUATION'
            })

    return errors


# -------------------------------------------------------------
# LanguageTool Integration with Error Categorization
# -------------------------------------------------------------

def map_languagetool_category(issue_type: str, rule_id: str, message: str) -> str:
    """Maps LanguageTool issue types and rule IDs into our 4 core UI categories."""
    issue_type_lower = (issue_type or '').lower()
    rule_id_lower = (rule_id or '').lower()
    msg_lower = (message or '').lower()

    if 'spelling' in issue_type_lower or 'typo' in rule_id_lower or 'spell' in rule_id_lower:
        return 'spelling'
    elif 'punctuation' in issue_type_lower or 'comma' in rule_id_lower or 'apostrophe' in rule_id_lower or 'whitespace' in rule_id_lower:
        return 'punctuation'
    elif 'style' in issue_type_lower or 'collocation' in issue_type_lower or 'redundancy' in rule_id_lower:
        return 'style'
    elif 'grammar' in issue_type_lower or 'agreement' in rule_id_lower or 'verb' in rule_id_lower:
        return 'grammar'
    
    # Text hints
    if 'spelling' in msg_lower or 'misspelled' in msg_lower:
        return 'spelling'
    elif 'punctuation' in msg_lower or 'comma' in msg_lower or 'period' in msg_lower:
        return 'punctuation'
    elif 'style' in msg_lower or 'passive voice' in msg_lower:
        return 'style'
    return 'grammar'


def query_languagetool(text: str) -> list:
    """
    Queries LanguageTool API for deep grammatical and spelling analysis.
    Returns standard list of error objects.
    """
    errors = []
    try:
        response = requests.post(
            LT_API_URL,
            data={'text': text, 'language': 'en-US'},
            timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            for match in data.get('matches', []):
                offset = match.get('offset', 0)
                length = match.get('length', 0)
                error_slice = text[offset:offset + length]
                suggestions = [r['value'] for r in match.get('replacements', [])[:4]]
                rule = match.get('rule', {})
                rule_id = rule.get('id', 'LT_RULE')
                issue_type = rule.get('issueType', 'grammar')
                message = match.get('message', 'Potential language error detected.')
                
                category = map_languagetool_category(issue_type, rule_id, message)

                # Special refinement for "He don't" -> LanguageTool returns "does" for "do",
                # let's refine suggestion to "doesn't" when the original word is "don't"
                if error_slice.lower() == "do" and offset + length < len(text) and text[offset:offset + 5].lower() == "don't":
                    length = 5
                    error_slice = text[offset:offset + length]
                    suggestions = ["doesn't", "does not"]
                    category = 'grammar'

                errors.append({
                    'offset': offset,
                    'length': length,
                    'error_text': error_slice,
                    'suggestions': suggestions,
                    'explanation': message,
                    'category': category,
                    'rule_id': rule_id
                })
    except Exception as e:
        # Fallback to local rules gracefully if network fails
        pass

    return errors


# -------------------------------------------------------------
# Conflict Resolution & Merge Engine
# -------------------------------------------------------------

def merge_and_deduplicate_errors(rule_errors: list, lt_errors: list, text_len: int) -> list:
    """
    Combines rule-based and LanguageTool errors, eliminating overlaps.
    Custom rule-based errors take precedence for explicit grammar rules.
    """
    all_errors = list(rule_errors)
    
    for lt_err in lt_errors:
        lt_start = lt_err['offset']
        lt_end = lt_start + lt_err['length']

        # Check if overlaps with any rule error
        overlaps = False
        for r_err in all_errors:
            r_start = r_err['offset']
            r_end = r_start + r_err['length']

            # Overlap condition
            if not (lt_end <= r_start or lt_start >= r_end):
                overlaps = True
                break

        if not overlaps:
            all_errors.append(lt_err)

    # Sort errors strictly by start offset ascending
    all_errors.sort(key=lambda x: (x['offset'], x['length']))
    
    # Assign sequential error IDs
    for idx, err in enumerate(all_errors):
        err['id'] = idx + 1

    return all_errors


# -------------------------------------------------------------
# Correction and Text Generation
# -------------------------------------------------------------

def apply_corrections(original_text: str, errors: list) -> str:
    """
    Applies the primary suggestion for each error to generate the corrected text.
    Processes replacements from right to left (descending offset) to maintain indices.
    """
    if not errors or not original_text:
        return original_text

    text_chars = list(original_text)
    # Sort descending by offset so index positions of earlier text are preserved
    sorted_errors = sorted(errors, key=lambda x: x['offset'], reverse=True)

    for err in sorted_errors:
        offset = err['offset']
        length = err['length']
        suggestions = err.get('suggestions', [])
        
        if suggestions and len(suggestions) > 0:
            best_suggestion = suggestions[0]
            # Replace characters in slice
            text_chars[offset:offset + length] = list(best_suggestion)

    corrected_str = "".join(text_chars)
    
    # Clean up any potential double spaces created during replacement
    corrected_clean = re.sub(r' +', ' ', corrected_str)
    # Ensure correct spacing around punctuation
    corrected_clean = re.sub(r'\s+([.,!?;:])', r'\1', corrected_clean)
    
    return corrected_clean


def generate_highlighted_html(original_text: str, errors: list) -> str:
    """
    Wraps detected errors in interactive semantic HTML tags with metadata attributes.
    Safe against HTML injection.
    """
    if not errors or not original_text:
        import html
        return html.escape(original_text)

    import html
    
    # Build segments
    sorted_errors = sorted(errors, key=lambda x: x['offset'])
    result_pieces = []
    last_idx = 0

    for err in sorted_errors:
        start = err['offset']
        end = start + err['length']

        # Prevent out-of-bounds
        if start > len(original_text):
            continue

        # Add text before error
        if start > last_idx:
            result_pieces.append(html.escape(original_text[last_idx:start]))

        error_chunk = original_text[start:end]
        if not error_chunk and err['length'] == 0:
            error_chunk = " "  # Marker for missing punctuation or insertion
            
        category = err.get('category', 'grammar').lower()
        err_id = err.get('id', 0)
        suggestion = html.escape(err['suggestions'][0]) if err.get('suggestions') else ''
        explanation = html.escape(err.get('explanation', ''))

        # Interactive highlight span
        span_html = (
            f'<span class="error-highlight error-{category}" '
            f'data-error-id="{err_id}" '
            f'data-category="{category}" '
            f'data-word="{html.escape(err.get("error_text", error_chunk))}" '
            f'data-suggestion="{suggestion}" '
            f'data-explanation="{explanation}" '
            f'tabindex="0" role="button" aria-label="{category} error: {html.escape(err.get("error_text", error_chunk))}">'
            f'{html.escape(error_chunk)}'
            f'</span>'
        )
        result_pieces.append(span_html)
        last_idx = end

    # Append any remaining text
    if last_idx < len(original_text):
        result_pieces.append(html.escape(original_text[last_idx:]))

    return "".join(result_pieces)


# -------------------------------------------------------------
# Master Grammar Checking Function
# -------------------------------------------------------------

def check_grammar(text: str) -> dict:
    """
    Master function: Executes full NLP grammar check, linguistic analysis,
    error categorization, text correction, and statistical calculations.
    
    Returns:
        dict: Full report including original text, corrected text, errors,
              NLP analysis, and performance metrics.
    """
    if not text or not text.strip():
        return {
            'original_text': '',
            'corrected_text': '',
            'error_count': 0,
            'errors': [],
            'highlighted_html': '',
            'accuracy_percentage': 100.0,
            'statistics': {
                'total_words': 0,
                'total_sentences': 0,
                'errors_detected': 0,
                'corrections_made': 0,
                'accuracy': 100.0
            },
            'nlp_analysis': analyze_text('')
        }

    raw_text = text.strip()

    # Step 1: Execute rule-based NLP error detectors
    rule_errors = []
    rule_errors.extend(check_subject_aux_inversion(raw_text))
    rule_errors.extend(check_subject_verb_agreement(raw_text))
    rule_errors.extend(check_articles(raw_text))
    rule_errors.extend(check_word_duplication(raw_text))
    # Note: Sentence capitalization and terminal punctuation can be checked if desirable
    rule_errors.extend(check_capitalization_and_punctuation(raw_text))

    # Step 2: Query statistical LanguageTool engine
    lt_errors = query_languagetool(raw_text)

    # Step 3: Merge and deduplicate
    final_errors = merge_and_deduplicate_errors(rule_errors, lt_errors, len(raw_text))

    # Step 4: Generate corrected text
    corrected_text = apply_corrections(raw_text, final_errors)

    # Step 5: Generate interactive HTML highlights
    highlighted_html = generate_highlighted_html(raw_text, final_errors)

    # Step 6: Perform linguistic analysis
    nlp_report = analyze_text(raw_text)

    # Step 7: Calculate statistical metrics
    word_count = nlp_report['word_count']
    sentence_count = nlp_report['sentence_count']
    total_errors = len(final_errors)
    corrections_count = sum(1 for e in final_errors if e.get('suggestions'))

    # Calculate Grammar Accuracy Percentage
    # Formula: max(0, 100 - (errors / words * 100))
    if word_count > 0:
        accuracy = max(0.0, round(100.0 - (total_errors / word_count * 100.0), 1))
    else:
        accuracy = 100.0 if total_errors == 0 else 0.0

    return {
        'original_text': raw_text,
        'corrected_text': corrected_text,
        'error_count': total_errors,
        'errors': final_errors,
        'highlighted_html': highlighted_html,
        'accuracy_percentage': accuracy,
        'statistics': {
            'total_words': word_count,
            'total_sentences': sentence_count,
            'errors_detected': total_errors,
            'corrections_made': corrections_count,
            'accuracy': accuracy
        },
        'nlp_analysis': nlp_report
    }


def correct_text(text: str) -> str:
    """Convenience helper returning just the corrected text string."""
    result = check_grammar(text)
    return result['corrected_text']
