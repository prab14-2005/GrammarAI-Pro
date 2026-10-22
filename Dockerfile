# Use official lightweight Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    NLTK_DATA=/app/nltk_data

# Set working directory
WORKDIR /app

# Install system dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-full.txt /app/
RUN pip install --no-cache-dir -r requirements-full.txt

# Pre-download NLTK corpora into the image for fast container startup
RUN python -c "import nltk; [nltk.download(x, download_dir='/app/nltk_data', quiet=True) for x in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'wordnet', 'omw-1.4']]"

# Copy application source code
COPY . /app/

# Create a non-root user for security (required by Hugging Face Spaces guidelines)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

# Hugging Face Spaces listens on port 7860
EXPOSE 7860

# Run the Flask app with Gunicorn
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--timeout", "180", "--bind", "0.0.0.0:7860", "app:app"]
