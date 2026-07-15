"""
config.py
----------
Central configuration for the SmartShield AI project.
Every path, hyperparameter, and constant used across the codebase
is defined here so nothing is hardcoded inside individual modules.
"""

import os

# ---------------------------------------------------------------------------
# BASE PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

# ---------------------------------------------------------------------------
# DATASET FILES
# ---------------------------------------------------------------------------
SPAM_DATASET_PATH = os.path.join(DATASET_DIR, "spam.csv")
PHISHING_DATASET_PATH = os.path.join(DATASET_DIR, "phishing.csv")

# ---------------------------------------------------------------------------
# MODEL ARTIFACT PATHS
# ---------------------------------------------------------------------------
LOGISTIC_MODEL_PATH = os.path.join(MODELS_DIR, "logistic_model.pkl")
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf.pkl")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.keras")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# ---------------------------------------------------------------------------
# NLP / DEEP LEARNING HYPERPARAMETERS
# ---------------------------------------------------------------------------
MAX_VOCAB_SIZE = 8000          # Tokenizer vocabulary cap
MAX_SEQUENCE_LENGTH = 60       # Padding length for LSTM input
EMBEDDING_DIM = 64             # Embedding layer output dimension
LSTM_UNITS = 64                # LSTM hidden units
DENSE_UNITS = 32               # Dense layer units before output
DROPOUT_RATE = 0.35
BATCH_SIZE = 32
EPOCHS = 8
VALIDATION_SPLIT = 0.15
TEST_SIZE = 0.20
RANDOM_STATE = 42

# TF-IDF
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# ---------------------------------------------------------------------------
# ENSEMBLE SETTINGS
# ---------------------------------------------------------------------------
# Weighted average between the classical ML model and the deep learning model
ENSEMBLE_WEIGHT_LOGISTIC = 0.4
ENSEMBLE_WEIGHT_LSTM = 0.6

# If the two model confidences differ by more than this threshold,
# the message is flagged as "Suspicious" rather than a hard Spam/Ham call.
DISAGREEMENT_THRESHOLD = 0.35

# Final decision thresholds (on the 0-1 ensemble spam probability)
SPAM_THRESHOLD = 0.60
SUSPICIOUS_LOWER_BOUND = 0.40

# ---------------------------------------------------------------------------
# APP METADATA
# ---------------------------------------------------------------------------
APP_NAME = "SmartShield AI"
APP_TAGLINE = "Hybrid Explainable Spam & Phishing Detection System"
APP_VERSION = "1.0.0"

DEVELOPER_NAME = "DharshiniMohan"
DEVELOPER_ROLE = "Pre Final year B.Tech Student (AI&DS)"
DEVELOPER_COLLEGE = "SRM MADURAI COLLEGE FOR ENGINEERING AND TECHNOLOGY"
DEVELOPER_GITHUB = "https://github.com/dharshinimohan31"
DEVELOPER_LINKEDIN = "https://linkedin.com/in/dharshini-mohan-3o"

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "smartshield.log")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
