"""Lazy, optional Hugging Face grammar-correction inference with Cloud API fallback."""

import logging
import os
import threading
import requests

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "prithivida/grammar_error_correcter_v1"
_load_lock = threading.Lock()
_model = None
_tokenizer = None
_load_error = None


def model_name():
    return os.getenv("GRAMMAR_AI_MODEL", DEFAULT_MODEL)


def _load_model():
    """Load one local model instance; importing transformers stays optional."""
    global _model, _tokenizer, _load_error
    if _model is not None:
        return _model, _tokenizer
    if _load_error is not None:
        raise RuntimeError(_load_error)
    with _load_lock:
        if _model is not None:
            return _model, _tokenizer
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            name = model_name()
            _tokenizer = AutoTokenizer.from_pretrained(name)
            _model = AutoModelForSeq2SeqLM.from_pretrained(name)
            _model.eval()
            logger.info("Loaded optional grammar AI model: %s", name)
            return _model, _tokenizer
        except Exception as exc:
            _load_error = "Local PyTorch/Transformers not installed."
            logger.info("Local AI model unavailable (%s). Falling back to Cloud API if configured.", exc)
            raise RuntimeError(_load_error) from exc


def _correct_with_hf_api(text: str) -> dict:
    """
    Fallback to Hugging Face Free Serverless Inference API.
    Enables AI Correct on Vercel and lightweight hosts without installing 1GB PyTorch.
    """
    name = model_name()
    api_url = f"https://api-inference.huggingface.co/models/{name}"
    hf_token = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
    
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    prompt = "gec: " + text.strip()

    try:
        resp = requests.post(
            api_url,
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_length": 256}},
            timeout=25
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                corrected = item.get("generated_text") or item.get("summary_text") or ""
                corrected = corrected.strip()
                if corrected:
                    return {"original_text": text.strip(), "corrected_text": corrected, "model": f"{name} (via Hugging Face API)"}
            elif isinstance(data, dict) and "generated_text" in data:
                return {"original_text": text.strip(), "corrected_text": data["generated_text"].strip(), "model": f"{name} (via Hugging Face API)"}

        if resp.status_code == 503:
            raise RuntimeError("The AI model is currently waking up on Hugging Face cloud. Please retry in 15 seconds.")
    except requests.exceptions.RequestException as e:
        logger.warning("HF API request failed: %s", e)

    # Informative message for serverless environments like Vercel
    if not hf_token:
        raise RuntimeError(
            "On serverless hosts like Vercel, add a free 'HF_TOKEN' in your Vercel Environment Variables to enable cloud AI corrections."
        )
    raise RuntimeError("Cloud AI model request timed out or was temporarily unavailable. Please retry shortly.")


def correct_with_ai(text: str) -> dict:
    """Generate a correction using a real seq2seq model, locally or via free Hugging Face API."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text is required for AI correction.")
    if len(text) > 5000:
        raise ValueError("AI correction supports up to 5,000 characters per request.")

    # 1. Attempt local model inference first (works on localhost & dedicated containers)
    try:
        model, tokenizer = _load_model()
        prompt = "gec: " + text.strip()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        generated = model.generate(**inputs, max_new_tokens=256, num_beams=4, early_stopping=True)
        corrected = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        if not corrected:
            raise RuntimeError("The AI model returned an empty correction.")
        return {"original_text": text.strip(), "corrected_text": corrected, "model": model_name()}
    except Exception:
        # 2. Fall back to Hugging Face Free Serverless Inference API (works on Vercel)
        return _correct_with_hf_api(text)
