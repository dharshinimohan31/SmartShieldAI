"""
predict.py
------------
Loads the trained Logistic Regression + TF-IDF model and the trained
LSTM model, runs a message through both, and returns the combined
ensemble prediction. This is the single entry point the Streamlit
app calls for inference.
"""

import sys
import os
import pickle
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.logger import get_logger
from src.preprocessing import preprocess_for_ml, preprocess_for_dl
from src.ensemble import combine_predictions

logger = get_logger(__name__)

_LOGISTIC_MODEL = None
_TFIDF_VECTORIZER = None
_LSTM_MODEL = None
_TOKENIZER = None


def _lazy_load():
    """Load all model artifacts once and cache them in module globals."""
    global _LOGISTIC_MODEL, _TFIDF_VECTORIZER, _LSTM_MODEL, _TOKENIZER

    if _LOGISTIC_MODEL is None:
        with open(config.LOGISTIC_MODEL_PATH, "rb") as f:
            _LOGISTIC_MODEL = pickle.load(f)
        logger.info("Logistic Regression model loaded.")

    if _TFIDF_VECTORIZER is None:
        with open(config.TFIDF_VECTORIZER_PATH, "rb") as f:
            _TFIDF_VECTORIZER = pickle.load(f)
        logger.info("TF-IDF vectorizer loaded.")

    if _TOKENIZER is None:
        with open(config.TOKENIZER_PATH, "rb") as f:
            _TOKENIZER = pickle.load(f)
        logger.info("Keras tokenizer loaded.")

    if _LSTM_MODEL is None:
        # Imported lazily so importing predict.py doesn't force a TF import
        # for code paths (like simple keyword tests) that don't need it.
        from tensorflow.keras.models import load_model
        _LSTM_MODEL = load_model(config.LSTM_MODEL_PATH)
        logger.info("LSTM model loaded.")


def preload_models():
    """
    Explicitly load every model artifact ahead of time AND run one
    dummy prediction through each pipeline. Call this once at app
    startup (e.g. inside a cached Streamlit resource loader) so that
    per-message inference timings reported to the user reflect only
    steady-state inference cost, not one-time disk-load, WordNet
    initialization, or TensorFlow graph-tracing overhead that would
    otherwise silently inflate the very first real prediction.
    """
    start = time.time()
    _lazy_load()
    # Warm-up pass: absorbs NLTK lemmatizer first-call cost and Keras
    # graph tracing so subsequent timings are representative.
    predict_logistic("warmup message")
    predict_lstm("warmup message")
    elapsed = time.time() - start
    logger.info(f"All models preloaded and warmed up in {elapsed:.2f}s")
    return elapsed


def predict_logistic(text: str) -> float:
    """Return P(spam) from the Logistic Regression model."""
    _lazy_load()
    cleaned = preprocess_for_ml(text)
    vec = _TFIDF_VECTORIZER.transform([cleaned])
    proba = _LOGISTIC_MODEL.predict_proba(vec)[0][1]
    return float(proba)


def predict_lstm(text: str) -> float:
    """Return P(spam) from the LSTM model."""
    _lazy_load()
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned = preprocess_for_dl(text)
    seq = _TOKENIZER.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=config.MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    proba = _LSTM_MODEL.predict(padded, verbose=0)[0][0]
    return float(proba)


def predict_message(text: str) -> dict:
    """
    Full hybrid inference: run both models, time each, and combine
    into a single ensemble verdict.

    Returns:
        dict with per-model confidences, timings, and ensemble verdict.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")

    start_lr = time.time()
    lr_conf = predict_logistic(text)
    lr_time = time.time() - start_lr

    start_lstm = time.time()
    lstm_conf = predict_lstm(text)
    lstm_time = time.time() - start_lstm

    result = combine_predictions(lr_conf, lstm_conf)
    result["logistic_time_ms"] = round(lr_time * 1000, 2)
    result["lstm_time_ms"] = round(lstm_time * 1000, 2)
    result["total_time_ms"] = round((lr_time + lstm_time) * 1000, 2)

    return result


if __name__ == "__main__":
    examples = [
        "Congratulations!! You won ₹50,000. Click here immediately to claim your prize.",
        "Hey, are we still meeting for lunch tomorrow at 1pm?",
        "URGENT: Your bank account has been suspended. Verify your OTP now at bit.ly/verify123",
    ]
    for msg in examples:
        logger.info(f"Message: {msg}")
        logger.info(predict_message(msg))
        logger.info("-" * 60)
