# 🚕 Taxi Tip Predictor - Real ML Project with XGBoost & GPU Acceleration

A professional-grade machine learning project that predicts taxi tips using XGBoost with NVIDIA GPU acceleration. This project demonstrates real-world data science workflows including data cleaning, memory optimization, and GPU-accelerated machine learning.

## 💡 What You'll Learn

- **Handling Real-World Datasets**: Cleanup, missing values, anomalies, aggregation
- **Solving Memory Limitations**: Using cuDF (GPU-accelerated Pandas) + RMM for memory management
- **GPU-Accelerated ML**: XGBoost on NVIDIA GPUs for faster training
- **Model Evaluation**: Comprehensive performance metrics and validation
- **Professional Workflow**: Think like a data scientist, solve problems systematically

## 🚀 Environment Setup

### Option 1: Google Colab (Recommended for Beginners)

1. Open Google Colab
2. Change runtime to **T4 GPU** (Runtime → Change runtime type → GPU: T4)
3. Use the smaller dataset (5 million rows)
4. Install dependencies (see `setup_colab.ipynb`)

### Option 2: Local Setup (Full Dataset - 38M Rows)

**Requirements:**
- CUDA-compatible GPU (NVIDIA)
- WSL (Windows Subsystem for Linux) - **MUST**
- Miniforge/Conda - **MUST**
- Follow current RAPIDS installation guide for your CUDA version

**Installation Steps:**

1. **Install RAPIDS** (check [RAPIDS Installation Guide](https://rapids.ai/start.html)):
   ```bash
   conda create -n rapids-env -c rapidsai -c conda-forge -c nvidia \
       cudf=24.04 python=3.10 cudatoolkit=11.8
   conda activate rapids-env
   ```

2. **Install XGBoost with GPU support**:
   ```bash
   pip install xgboost
   ```

3. **Install other dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

Download the NYC Taxi dataset:
- **Colab version**: 5 million rows (smaller subset)
- **Local version**: 38 million rows (full dataset)

Place the dataset in the `data/` directory.

## 🏗️ Project Structure

```
taxi-tip-predictor/
├── README.md
├── requirements.txt
├── setup_colab.ipynb          # Colab setup notebook
├── config.py                   # Configuration settings
├── data_loader.py              # Data loading and preprocessing
├── feature_engineering.py      # Feature engineering pipeline
├── model_trainer.py            # XGBoost model training
├── model_evaluator.py          # Model evaluation and metrics
├── hyperparameter_tuning.py    # Hyperparameter tuning module
├── main.py                     # Main training script (K-Fold validation)
├── main_advanced.py            # Advanced training (with hyperparameter tuning)
└── data/                       # Dataset directory
    └── (place your CSV files here)
```

## ⚡ Quick Start

1. **Check your environment**:
   ```bash
   python check_environment.py
   ```

2. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the NYC Taxi dataset** (`Distilled_2023_Yellow_Taxi_Trip_Data.csv`) and place it in the `data/` directory

4. **Update config.py** if needed (dataset filename, paths, etc.)

5. **Run training**:
   - **Basic workflow** (K-Fold validation):
     ```bash
     python main.py
     ```
   - **Advanced workflow** (with hyperparameter tuning):
     ```bash
     python main_advanced.py
     ```

## 🎯 Usage

### Training the Model

```bash
python main.py
```

### Configuration

Edit `config.py` to adjust:
- Dataset path
- GPU memory settings
- Model hyperparameters
- Training parameters

## 📈 Features

- **GPU-Accelerated Data Processing**: cuDF for fast data manipulation
- **Memory Management**: RMM for efficient GPU memory allocation
- **Feature Engineering**: Time-based, distance, and aggregated features
- **Model Training**: XGBoost with GPU acceleration
- **Comprehensive Evaluation**: RMSE, MAE, R², and feature importance

## 🧠 What Makes This Different

This isn't a beginner demo—it's a **real workflow** based on:
- Real data challenges (huge datasets, missing values, anomalies)
- Real problems (CPU/GPU memory limits, runtime crashes)
- Professional solutions (explained step-by-step)
- Decision-making rationale (why we make each choice)

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

Based on professional data science workflows and best practices for GPU-accelerated machine learning.
