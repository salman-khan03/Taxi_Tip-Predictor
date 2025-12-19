# 🔄 Project Updates - Matching Notebook Workflow

This document describes the updates made to match the professional XGBoost workflow from the tutorial notebooks.

## ✅ Key Updates

### 1. Data Cleaning Workflow
**Updated to match notebook exactly:**

- **Initial Cleanup** (`clean_data_initial`):
  - `airport_fee`: Fill missing with `0` (no fee charged)
  - `congestion_surcharge`: Fill missing with `0` (no fee charged)
  - `passenger_count`: Fill missing with `mean()` (driver forgot to enter)
  - `RatecodeID`: Fill missing with `99` (unknown label)
  - `store_and_fwd_flag`: Drop column entirely (not meaningful for tip prediction)

- **Anomaly Removal** (`clean_data_anomalies`):
  - `trip_distance`: Keep only `0.5 < distance < 100` miles
  - Remove all rows with any negative numeric values
  - `total_amount`: Keep only `> 3.7` (NYC base fare minimum)

### 2. Date/Time Feature Engineering
**Matches notebook format exactly:**

- Date format: `"%m/%d/%Y %I:%M:%S %p"` (e.g., "05/01/2023 03:01:24 PM")
- Column names: `tpep_pickup_datetime` and `tpep_dropoff_datetime`
- Extracted features:
  - `pickup_month`, `dropoff_month` (1-12)
  - `pickup_dow`, `dropoff_dow` (0-6, day of week)
  - `pickup_ampm`, `dropoff_ampm` (AM/PM as category)
  - `trip_duration_min` (float32, in minutes)
- Trip duration filtering: `1 < duration < 120` minutes

### 3. Train/Test Split
**Matches notebook approach:**

- Shuffle with `random_state=432`
- Last 1,000,000 rows for testing
- Rest for training
- Drop datetime columns from features (already extracted)

### 4. K-Fold Validation
**New feature matching advanced notebook:**

- 5-fold cross-validation
- Separate validation MSE for each fold
- Mean and standard deviation of validation MSE
- Final model trained on full training data

### 5. Hyperparameter Tuning
**New module matching advanced notebook:**

- Grid search over:
  - Learning rates: `[0.1, 0.05]`
  - N estimators: `[200, 500, 1000]`
- K-Fold validation for each combination
- Results DataFrame sorted by validation MSE
- Visualization of tuning results

### 6. Model Training
**Updated to match notebook:**

- XGBoost with `device="cuda"` for GPU acceleration
- `enable_categorical=True` for categorical features
- K-Fold validation integrated
- Hyperparameter support

### 7. Configuration
**Updated defaults:**

- Dataset filename: `Distilled_2023_Yellow_Taxi_Trip_Data.csv`
- Default test size: 1,000,000 rows (matching notebook)

## 📁 New Files

1. **`hyperparameter_tuning.py`**: Hyperparameter tuning module
2. **`main_advanced.py`**: Advanced training script with hyperparameter tuning

## 🔧 Modified Files

1. **`data_loader.py`**:
   - Split `clean_data()` into `clean_data_initial()` and `clean_data_anomalies()`
   - Updated cleaning logic to match notebook exactly

2. **`feature_engineering.py`**:
   - Updated date parsing to use exact format from notebook
   - Updated trip duration filtering (1-120 minutes)
   - Added support for `tpep_pickup_datetime` column names

3. **`model_trainer.py`**:
   - Updated `prepare_data()` to match notebook split approach
   - Added `train_with_kfold()` method
   - Support for hyperparameter overrides

4. **`main.py`**:
   - Updated to use K-Fold validation
   - Updated data preparation flow

5. **`config.py`**:
   - Updated default dataset filename

## 🎯 Workflow Comparison

### Basic Workflow (`main.py`)
1. Load data
2. Initial cleanup (missing values)
3. Feature engineering (dates, etc.)
4. Remove anomalies
5. Train with K-Fold validation
6. Evaluate

### Advanced Workflow (`main_advanced.py`)
1. Load data
2. Initial cleanup (missing values)
3. Feature engineering (dates, etc.)
4. Remove anomalies
5. Train baseline with K-Fold
6. **Hyperparameter tuning**
7. Train best model
8. Evaluate and compare

## 📊 Expected Results

Based on the notebook workflow:

- **Model 1 (baseline)**: MSE ~1.12
- **Model 2 (after anomaly removal)**: MSE ~0.82
- **Model 3 (with date features)**: MSE ~0.97 (validation), ~0.81 (validation mean)
- **Best tuned model**: MSE ~0.75-0.93 (depending on hyperparameters)

## 🚀 Usage

```bash
# Basic workflow (K-Fold validation)
python main.py

# Advanced workflow (with hyperparameter tuning)
python main_advanced.py
```

## 📝 Notes

- The workflow now exactly matches the tutorial notebooks
- All cleaning steps, thresholds, and feature engineering match
- K-Fold validation and hyperparameter tuning are fully integrated
- GPU acceleration is supported throughout
- Both pandas (CPU) and cuDF (GPU) are supported
