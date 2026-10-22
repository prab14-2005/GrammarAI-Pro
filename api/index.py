"""
Vercel Serverless Function Entry Point for Grammar Error Detector.
Routes all requests to the Flask application instance.
"""

import os
import sys

# Ensure project root directory is on the Python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app import app

# Export WSGI application instance
app = app
