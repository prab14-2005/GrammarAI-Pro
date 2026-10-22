"""
Database Module for Grammar Error Detector & Corrector
Uses SQLite to persist check history, errors detected, corrections, and NLP metrics.
Supports both local file persistence and serverless (e.g. Vercel/AWS Lambda) /tmp storage.
"""

import os
import sqlite3
import json
import tempfile
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def _resolve_default_db_path():
    """
    Resolves the appropriate SQLite database file path.
    In serverless environments (e.g. Vercel, AWS Lambda) or when the project
    directory is read-only, uses the writable temp directory.
    """
    if os.environ.get('VERCEL') or os.environ.get('AWS_LAMBDA_FUNCTION_NAME'):
        return os.path.join(tempfile.gettempdir(), 'grammar_detector_history.db')

    local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_path = os.path.join(local_dir, 'database.db')

    # Test if the directory is actually writable
    try:
        test_file = os.path.join(local_dir, '.write_check')
        with open(test_file, 'w') as f:
            f.write('')
        os.remove(test_file)
        return target_path
    except (OSError, IOError, PermissionError):
        # Fallback to temp directory on read-only environments
        return os.path.join(tempfile.gettempdir(), 'grammar_detector_history.db')

DEFAULT_DB_PATH = _resolve_default_db_path()


def get_db_connection(db_path=None):
    """Establishes a connection to the SQLite database with row factory enabled."""
    path = db_path or _resolve_default_db_path()
    # Ensure parent directory exists if using temp
    parent_dir = os.path.dirname(path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    """Initializes database schema and indexes."""
    path = db_path or _resolve_default_db_path()
    try:
        conn = get_db_connection(path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS grammar_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_text TEXT NOT NULL,
                corrected_text TEXT NOT NULL,
                error_count INTEGER NOT NULL DEFAULT 0,
                errors_json TEXT DEFAULT '[]',
                nlp_summary TEXT DEFAULT '{}',
                ai_metadata_json TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Index for fast retrieval sorted by timestamp
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_grammar_history_created 
            ON grammar_history (created_at DESC)
        """)
        # Backward-compatible migration for databases created before AI support.
        columns = {row['name'] for row in cursor.execute("PRAGMA table_info(grammar_history)")}
        if 'ai_metadata_json' not in columns:
            cursor.execute("ALTER TABLE grammar_history ADD COLUMN ai_metadata_json TEXT DEFAULT '{}'")

        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Database initialization warning for path {path}: {e}")


def add_history(original_text, corrected_text, error_count, errors=None, nlp_summary=None, ai_metadata=None, db_path=None):
    """
    Saves a grammar check result into the history table.
    
    Args:
        original_text (str): The user's input text
        corrected_text (str): The text after applying grammatical corrections
        error_count (int): Number of detected errors
        errors (list): List of error dictionaries
        nlp_summary (dict): High-level NLP metrics (word count, sentence count, accuracy, etc.)
        db_path (str, optional): Custom database path for testing
        
    Returns:
        int: The inserted record ID (or 0 if write failed)
    """
    path = db_path or _resolve_default_db_path()
    try:
        init_db(path)
        conn = get_db_connection(path)
        cursor = conn.cursor()

        errors_str = json.dumps(errors if errors is not None else [], ensure_ascii=False)
        nlp_str = json.dumps(nlp_summary if nlp_summary is not None else {}, ensure_ascii=False)
        ai_str = json.dumps(ai_metadata if ai_metadata is not None else {}, ensure_ascii=False)

        cursor.execute("""
            INSERT INTO grammar_history (original_text, corrected_text, error_count, errors_json, nlp_summary, ai_metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now', 'localtime'))
        """, (original_text, corrected_text, error_count, errors_str, nlp_str, ai_str))

        item_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return item_id
    except Exception as e:
        logger.warning(f"add_history warning: {e}")
        return 0


def get_history(limit=50, db_path=None):
    """
    Retrieves the most recent grammar check entries.
    
    Returns:
        list of dicts: History records
    """
    path = db_path or _resolve_default_db_path()
    try:
        init_db(path)
        conn = get_db_connection(path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, original_text, corrected_text, error_count, errors_json, nlp_summary, created_at
            FROM grammar_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        history = []
        for row in rows:
            try:
                errors_data = json.loads(row['errors_json']) if row['errors_json'] else []
            except Exception:
                errors_data = []

            try:
                nlp_data = json.loads(row['nlp_summary']) if row['nlp_summary'] else {}
            except Exception:
                nlp_data = {}

            history.append({
                'id': row['id'],
                'original_text': row['original_text'],
                'corrected_text': row['corrected_text'],
                'error_count': row['error_count'],
                'errors': errors_data,
                'nlp_summary': nlp_data,
                'created_at': row['created_at']
            })

        conn.close()
        return history
    except Exception as e:
        logger.warning(f"get_history warning: {e}")
        return []


def get_history_item(item_id, db_path=None):
    """Retrieves a single history item by ID."""
    path = db_path or _resolve_default_db_path()
    try:
        conn = get_db_connection(path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, original_text, corrected_text, error_count, errors_json, nlp_summary, created_at
            FROM grammar_history
            WHERE id = ?
        """, (item_id,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        try:
            errors_data = json.loads(row['errors_json']) if row['errors_json'] else []
        except Exception:
            errors_data = []

        try:
            nlp_data = json.loads(row['nlp_summary']) if row['nlp_summary'] else {}
        except Exception:
            nlp_data = {}

        return {
            'id': row['id'],
            'original_text': row['original_text'],
            'corrected_text': row['corrected_text'],
            'error_count': row['error_count'],
            'errors': errors_data,
            'nlp_summary': nlp_data,
            'created_at': row['created_at']
        }
    except Exception as e:
        logger.warning(f"get_history_item warning: {e}")
        return None


def delete_history(item_id, db_path=None):
    """Deletes a specific history record by ID."""
    path = db_path or _resolve_default_db_path()
    try:
        conn = get_db_connection(path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM grammar_history WHERE id = ?", (item_id,))
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        return rows_affected > 0
    except Exception as e:
        logger.warning(f"delete_history warning: {e}")
        return False


def clear_history(db_path=None):
    """Clears all history records from the table."""
    path = db_path or _resolve_default_db_path()
    try:
        conn = get_db_connection(path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM grammar_history")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.warning(f"clear_history warning: {e}")
        return False


def get_statistics(db_path=None):
    """
    Computes aggregate metrics for dashboard analytics.
    
    Returns:
        dict: Aggregated metrics
    """
    path = db_path or _resolve_default_db_path()
    category_counts = {
        'grammar': 0,
        'spelling': 0,
        'punctuation': 0,
        'style': 0,
        'other': 0
    }
    empty_stats = {
        'total_checks': 0,
        'total_errors': 0,
        'total_corrections': 0,
        'avg_accuracy': 100.0,
        'category_counts': category_counts
    }
    try:
        init_db(path)
        conn = get_db_connection(path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                COUNT(*) as total_checks,
                COALESCE(SUM(error_count), 0) as total_errors
            FROM grammar_history
        """)
        totals = cursor.fetchone()
        total_checks = totals['total_checks'] if totals else 0
        total_errors = totals['total_errors'] if totals else 0

        # Retrieve all errors to calculate categories and corrections
        cursor.execute("SELECT errors_json, nlp_summary FROM grammar_history")
        rows = cursor.fetchall()
        conn.close()

        total_corrections = 0
        accuracies = []

        for row in rows:
            try:
                errs = json.loads(row['errors_json']) if row['errors_json'] else []
                for e in errs:
                    cat = e.get('category', 'grammar').lower()
                    if cat in category_counts:
                        category_counts[cat] += 1
                    else:
                        category_counts['other'] += 1
                    if e.get('suggestions'):
                        total_corrections += 1
            except Exception:
                pass

            try:
                nlp = json.loads(row['nlp_summary']) if row['nlp_summary'] else {}
                if 'accuracy' in nlp:
                    accuracies.append(float(nlp['accuracy']))
            except Exception:
                pass

        avg_accuracy = round(sum(accuracies) / len(accuracies), 1) if accuracies else 100.0

        return {
            'total_checks': total_checks,
            'total_errors': total_errors,
            'total_corrections': total_corrections,
            'avg_accuracy': avg_accuracy,
            'category_counts': category_counts
        }
    except Exception as e:
        logger.warning(f"get_statistics warning: {e}")
        return empty_stats
