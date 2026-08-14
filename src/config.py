"""
config.py

This module stores all configurable parameters used throughout the project.

It includes:
- Dataset directory paths
- Output file locations
- MediaPipe configuration parameters
- Image processing settings
- Training configuration constants

Keeping all configuration values in one place makes the project easier to
maintain and prevents hard-coded values from being scattered across multiple files.
"""

from pathlib import Path

# =============================================================================
# Project Directories
# =============================================================================
PROJECT_ROOT = Path("d:/computer_vision_WP/Gesture Volume Control")

RAW_DATASET_DIR = PROJECT_ROOT / "DataSet" / "raw" / "ASL_Alphabet_Dataset"

TRAIN_DIR = RAW_DATASET_DIR / "asl_alphabet_train"

PROCESSED_DATASET_DIR = PROJECT_ROOT / "DataSet" / "processed"

#Files that will be extacted

LANDMARKS_CSV = PROCESSED_DATASET_DIR / "landmarks.csv"

TRAIN_CSV = PROCESSED_DATASET_DIR / "train.csv"

VALIDATION_CSV = PROCESSED_DATASET_DIR / "validation.csv"

TEST_CSV = PROCESSED_DATASET_DIR / "test.csv"


LSTM_TRAIN_CSV = PROCESSED_DATASET_DIR / "lstm_train.csv"

LSTM_VALIDATION_CSV = PROCESSED_DATASET_DIR / "lstm_validation.csv"

LSTM_TEST_CSV = PROCESSED_DATASET_DIR / "lstm_test.csv"

#Model directory

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "asl_classifier.pkl"

#MediaPipe configuration

MAX_NUM_HANDS = 1

MIN_DETECTION_CONFIDENCE = 0.7

MIN_TRACKING_CONFIDENCE = 0.5

#independent images from a dataset so Static image mode= True
STATIC_IMAGE_MODE = True 

#Random seed
RANDOM_SEED = 42