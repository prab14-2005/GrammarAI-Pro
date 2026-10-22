"""
Database access package for Grammar Error Detector.
"""
from .db import init_db, get_history, add_history, delete_history, clear_history, get_statistics

__all__ = ['init_db', 'get_history', 'add_history', 'delete_history', 'clear_history', 'get_statistics']
