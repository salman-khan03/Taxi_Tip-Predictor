"""
Configuration file for Taxi Tip Predictor project.
Adjust these settings based on your environment and dataset.
"""

import os

# Dataset Configuration
DATA_DIR = "data"
DATASET_FILE = "Distilled_2023_Yellow_Taxi_Trip_Data.csv"  # Default dataset filename
USE_CUDF = True  # Set to False if cuDF is not available (CPU fallback)

# GPU Memory Management (RMM)
ENABLE_RMM = True
RMM_POOL_SIZE = "4GB"  # Adjust based on your GPU memory
RMM_INITIAL_POOL_SIZE = "2GB"

# Data Processing
CHUNK_SIZE = 1_000_000  # Process data in chunks if needed
SAMPLE_SIZE = None  # Set to int for sampling (e.g., 5_000_000 for Colab)

# Feature Engineering
FEATURES_TO_DROP = [
    'vendor_id',  # If not useful
    'store_and_fwd_flag',  # If not useful
]

# Model Configuration
XGBOOST_PARAMS = {
    'tree_method': 'hist',  # Use 'gpu_hist' for GPU, 'hist' for CPU
    'device': 'cuda',  # 'cuda' for GPU, 'cpu' for CPU
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbosity': 1,
}

# Training Configuration
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50

# Output Configuration
MODEL_DIR = "models"
RESULTS_DIR = "results"
SAVE_MODEL = True
SAVE_PREDICTIONS = True

# Logging
VERBOSE = True
LOG_LEVEL = "INFO"
