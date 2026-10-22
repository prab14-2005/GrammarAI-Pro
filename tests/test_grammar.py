"""
Unit and Integration Tests for Grammar Error Detection & Correction System.
"""

import os
import unittest
import tempfile
from unittest.mock import patch
from app import app
from nlp.grammar_checker import check_grammar, correct_text
from nlp.analyzer import analyze_text
from database.db import init_db, add_history, get_history, get_history_item, delete_history, clear_history, get_statistics


class TestGrammarChecker(unittest.TestCase):

    def test_sample_case_1_she_go(self):
        """Test Case 1: 'She go to school every day.' -> 'She goes to school every day.'"""
        text = "She go to school every day."
        result = check_grammar(text)
        self.assertEqual(result['corrected_text'], "She goes to school every day.")
        self.assertGreaterEqual(result['error_count'], 1)
        err_words = [e['error_text'] for e in result['errors']]
        self.assertIn("go", err_words)

    def test_sample_case_2_he_dont(self):
        """Test Case 2: 'He don't like apples.' -> 'He doesn't like apples.'"""
        text = "He don't like apples."
        result = check_grammar(text)
        self.assertEqual(result['corrected_text'], "He doesn't like apples.")
        self.assertGreaterEqual(result['error_count'], 1)

    def test_sample_case_3_i_has(self):
        """Test Case 3: 'I has a book.' -> 'I have a book.'"""
        text = "I has a book."
        result = check_grammar(text)
        self.assertEqual(result['corrected_text'], "I have a book.")
        self.assertGreaterEqual(result['error_count'], 1)

    def test_sample_case_4_they_was(self):
        """Test Case 4: 'They was playing cricket.' -> 'They were playing cricket.'"""
        text = "They was playing cricket."
        result = check_grammar(text)
        self.assertEqual(result['corrected_text'], "They were playing cricket.")
        self.assertGreaterEqual(result['error_count'], 1)

    def test_sample_case_5_where_you_are(self):
        """Test Case 5: 'Where you are going?' -> 'Where are you going?'"""
        text = "Where you are going?"
        result = check_grammar(text)
        self.assertEqual(result['corrected_text'], "Where are you going?")
        self.assertGreaterEqual(result['error_count'], 1)

    def test_articles_correction(self):
        """Test indefinite article correction 'a apple' -> 'an apple'"""
        text = "I saw a apple on the table."
        result = check_grammar(text)
        self.assertIn("an apple", result['corrected_text'])

    def test_word_duplication(self):
        """Test duplicate consecutive word removal."""
        text = "This is the the best day."
        result = check_grammar(text)
        self.assertEqual(result['corrected_text'], "This is the best day.")

    def test_empty_input(self):
        """Test empty string handling."""
        result = check_grammar("")
        self.assertEqual(result['error_count'], 0)
        self.assertEqual(result['corrected_text'], "")
        self.assertEqual(result['accuracy_percentage'], 100.0)


class TestNLPAnalyzer(unittest.TestCase):

    def test_analyzer_tokenization_and_pos(self):
        """Verify tokens, pos tags, and lemmatization."""
        text = "The quick brown fox jumps over lazy dogs."
        res = analyze_text(text)
        self.assertEqual(res['sentence_count'], 1)
        self.assertEqual(res['word_count'], 8)
        self.assertGreater(len(res['tokens']), 0)
        
        # Check token attributes
        first_token = res['tokens'][0]
        self.assertEqual(first_token['token'], 'The')
        self.assertIn('tag', first_token)
        self.assertIn('description', first_token)
        self.assertIn('lemma', first_token)


class TestDatabaseOperations(unittest.TestCase):

    def setUp(self):
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix='.db')
        init_db(self.temp_db_path)

    def tearDown(self):
        os.close(self.temp_db_fd)
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_add_and_retrieve_history(self):
        item_id = add_history(
            original_text="She go home.",
            corrected_text="She goes home.",
            error_count=1,
            errors=[{"error_text": "go", "category": "grammar"}],
            nlp_summary={"accuracy": 66.7, "total_words": 3},
            db_path=self.temp_db_path
        )
        self.assertIsNotNone(item_id)
        
        history = get_history(db_path=self.temp_db_path)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['original_text'], "She go home.")
        self.assertEqual(history[0]['corrected_text'], "She goes home.")

        stats = get_statistics(db_path=self.temp_db_path)
        self.assertEqual(stats['total_checks'], 1)
        self.assertEqual(stats['total_errors'], 1)

    def test_delete_and_clear_history(self):
        id1 = add_history("Error 1", "Correction 1", 1, db_path=self.temp_db_path)
        id2 = add_history("Error 2", "Correction 2", 1, db_path=self.temp_db_path)
        self.assertEqual(len(get_history(db_path=self.temp_db_path)), 2)

        # Delete single item
        delete_history(id1, db_path=self.temp_db_path)
        remaining = get_history(db_path=self.temp_db_path)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]['id'], id2)

        # Clear all
        clear_history(db_path=self.temp_db_path)
        self.assertEqual(len(get_history(db_path=self.temp_db_path)), 0)


class TestAIEndpoint(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    @patch('app.correct_with_ai')
    def test_ai_correct_uses_model_result(self, mock_correct):
        mock_correct.return_value = {
            'corrected_text': 'She goes to college every day.',
            'model': 'test/grammar-model'
        }
        response = self.client.post('/api/ai-correct', json={'text': 'She go to college every day.'})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body['success'])
        self.assertEqual(body['corrected_text'], 'She goes to college every day.')
        self.assertEqual(body['model'], 'test/grammar-model')
        mock_correct.assert_called_once()

    def test_ai_correct_rejects_empty_input(self):
        response = self.client.post('/api/ai-correct', json={'text': '  '})
        self.assertEqual(response.status_code, 400)

    @patch('app.correct_with_ai', side_effect=RuntimeError('model unavailable'))
    def test_ai_correct_reports_model_failure(self, _mock_correct):
        response = self.client.post('/api/ai-correct', json={'text': 'I has a book.'})
        self.assertEqual(response.status_code, 503)
        self.assertIn('model unavailable', response.get_json()['error'])


if __name__ == '__main__':
    unittest.main()
