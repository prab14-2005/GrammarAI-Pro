"""
Flask Application for Grammar Error Detection and Correction.

Micro-project: Grammar Error Detection and Correction Using Natural Language Processing.
Provides RESTful APIs for grammar verification, linguistic inspection, and historical telemetry.
"""

import os
import sys
import logging
from flask import Flask, render_template, request, jsonify

try:
    from flask_cors import CORS
    has_cors = True
except ImportError:
    has_cors = False

from database.db import (
    init_db,
    add_history,
    get_history,
    get_history_item,
    delete_history,
    clear_history,
    get_statistics
)
from nlp.preprocessing import preprocess_text
from nlp.grammar_checker import check_grammar
from nlp.analyzer import analyze_text
from nlp.ai_correction import correct_with_ai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
logger = logging.getLogger(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Flask app with explicit template and static directories
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static'),
    static_url_path='/static'
)

class VercelPrefixMiddleware:
    """Normalizes PATH_INFO when invoked through Vercel serverless rewrites."""
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        import urllib.parse
        qs = urllib.parse.parse_qs(environ.get('QUERY_STRING', ''))
        
        if '__orig_path' in qs:
            target = qs['__orig_path'][0]
            new_qs_dict = {k: v for k, v in qs.items() if k != '__orig_path'}
            environ['QUERY_STRING'] = urllib.parse.urlencode(new_qs_dict, doseq=True)
            while '//' in target:
                target = target.replace('//', '/')
            environ['PATH_INFO'] = target if target.startswith('/') else '/' + target
        else:
            path = environ.get('PATH_INFO', '')
            for prefix in ('/api/index.py', '/api/index'):
                if path == prefix:
                    environ['PATH_INFO'] = '/'
                    break
                elif path.startswith(prefix + '/'):
                    environ['PATH_INFO'] = path[len(prefix):] or '/'
                    break

        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPrefixMiddleware(app.wsgi_app)

if has_cors:
    CORS(app)

# Ensure database is initialized safely on startup
with app.app_context():
    try:
        init_db()
        logger.info("SQLite database initialized successfully.")
    except Exception as e:
        logger.warning(f"Database startup warning: {e}")


# -------------------------------------------------------------
# Frontend View Route
# -------------------------------------------------------------

@app.route('/')
def index():
    """Serves the main single-page web interface."""
    return render_template('index.html')


# -------------------------------------------------------------
# RESTful API Endpoints
# -------------------------------------------------------------

@app.route('/api/check', methods=['POST'])
def api_check_grammar():
    """
    Main Grammar Checking Endpoint.
    
    Expects JSON payload:
        { "text": "English sentence or paragraph" }
        
    Workflow:
        User enters text
            ↓
        Text preprocessing & validation
            ↓
        Tokenization & POS tagging
            ↓
        Lemmatization & Syntactic check
            ↓
        Error detection & correction suggestions
            ↓
        Display generation (HTML highlights + statistics)
            ↓
        Store in SQLite history database
    """
    try:
        data = request.get_json(silent=True)
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Invalid request format. JSON object with "text" field is required.'
            }), 400

        raw_text = data.get('text', '')
        prep_report = preprocess_text(raw_text)

        if prep_report['is_empty']:
            return jsonify({
                'success': False,
                'error': 'Input text cannot be empty. Please enter an English sentence.'
            }), 400

        # Maximum payload guard (10,000 characters)
        if prep_report['char_count'] > 10000:
            return jsonify({
                'success': False,
                'error': 'Text exceeds maximum supported length (10,000 characters).'
            }), 400

        # Run NLP Grammar and Linguistic Pipeline
        result = check_grammar(prep_report['cleaned_text'])

        # Store result in SQLite database
        history_id = add_history(
            original_text=result['original_text'],
            corrected_text=result['corrected_text'],
            error_count=result['error_count'],
            errors=result['errors'],
            nlp_summary={
                'accuracy': result['accuracy_percentage'],
                'total_words': result['statistics']['total_words'],
                'total_sentences': result['statistics']['total_sentences'],
                'lexical_diversity': result['nlp_analysis']['lexical_diversity']
            }
        )

        response_payload = {
            'success': True,
            'id': history_id,
            'original_text': result['original_text'],
            'corrected_text': result['corrected_text'],
            'error_count': result['error_count'],
            'errors': result['errors'],
            'highlighted_html': result['highlighted_html'],
            'accuracy_percentage': result['accuracy_percentage'],
            'statistics': result['statistics'],
            'nlp_analysis': result['nlp_analysis']
        }

        return jsonify(response_payload), 200

    except Exception as e:
        logger.error(f"Error in /api/check: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'An internal NLP processing error occurred: {str(e)}'
        }), 500


@app.route('/api/analyze', methods=['POST'])
def api_analyze_nlp():
    """
    Dedicated endpoint for in-depth NLP token and linguistic breakdown.
    Expects: { "text": "..." }
    """
    try:
        data = request.get_json(silent=True) or {}
        text = data.get('text', '')
        if not text.strip():
            return jsonify({'success': False, 'error': 'Text is required for NLP analysis.'}), 400

        analysis = analyze_text(text)
        return jsonify({'success': True, 'analysis': analysis}), 200
    except Exception as e:
        logger.error(f"Error in /api/analyze: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai-correct', methods=['POST'])
def api_ai_correct():
    """Optional pretrained-model correction; existing rule checker remains separate."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get('text'), str):
        return jsonify({'success': False, 'error': 'Invalid request format. JSON string field "text" is required.'}), 400
    text = data['text'].strip()
    if not text:
        return jsonify({'success': False, 'error': 'Input text cannot be empty.'}), 400
    if len(text) > 5000:
        return jsonify({'success': False, 'error': 'AI correction supports up to 5,000 characters.'}), 400
    try:
        ai_result = correct_with_ai(text)
        analysis = analyze_text(text)
        history_id = add_history(
            original_text=text, corrected_text=ai_result['corrected_text'], error_count=0,
            errors=[], nlp_summary={'total_words': analysis['word_count'], 'total_sentences': analysis['sentence_count']},
            ai_metadata={'provider': 'Hugging Face local model', 'model': ai_result['model'], 'method': 'generative'}
        )
        return jsonify({
            'success': True, 'id': history_id, 'original_text': text,
            'corrected_text': ai_result['corrected_text'], 'model': ai_result['model'],
            'explanations': ['Generative correction produced by the pretrained model. Review changes because model output can alter meaning.'],
            'nlp_analysis': analysis
        }), 200
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 503
    except Exception:
        logger.exception('Unexpected error in /api/ai-correct')
        return jsonify({'success': False, 'error': 'AI correction is temporarily unavailable.'}), 500


@app.route('/api/history', methods=['GET'])
def api_get_history():
    """Returns stored check history items."""
    try:
        limit = request.args.get('limit', default=50, type=int)
        history_records = get_history(limit=limit)
        return jsonify({
            'success': True,
            'count': len(history_records),
            'history': history_records
        }), 200
    except Exception as e:
        logger.error(f"Error in /api/history: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/history/<int:item_id>', methods=['GET'])
def api_get_history_item(item_id):
    """Retrieves a single history entry by its primary key ID."""
    try:
        item = get_history_item(item_id)
        if not item:
            return jsonify({'success': False, 'error': f'Item with id {item_id} not found.'}), 404
        return jsonify({'success': True, 'item': item}), 200
    except Exception as e:
        logger.error(f"Error retrieving item {item_id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/history/<int:item_id>', methods=['DELETE'])
def api_delete_history(item_id):
    """Deletes an individual history item by ID."""
    try:
        success = delete_history(item_id)
        if not success:
            return jsonify({'success': False, 'error': f'History item with id {item_id} not found.'}), 404
        return jsonify({'success': True, 'message': f'Record {item_id} successfully deleted.'}), 200
    except Exception as e:
        logger.error(f"Error deleting history {item_id}: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/clear-history', methods=['POST'])
def api_clear_history():
    """Wipes all check history records from the database."""
    try:
        clear_history()
        return jsonify({'success': True, 'message': 'All history cleared successfully.'}), 200
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def api_get_statistics():
    """Returns aggregated system metrics for the dashboard."""
    try:
        stats = get_statistics()
        return jsonify({'success': True, 'statistics': stats}), 200
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# -------------------------------------------------------------
# Global Error Handlers
# -------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'error': 'API endpoint not found'}), 404
    return render_template('index.html'), 200


@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=======================================================")
    print(f" Grammar Error Detector & Corrector NLP Project")
    print(f" Running at: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG') == '1')
