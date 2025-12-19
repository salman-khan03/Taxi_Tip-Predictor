"""
Model training module.
Handles XGBoost model training with GPU acceleration.
"""

import os
import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
import config

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    import pandas as pd

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    raise ImportError("XGBoost is required. Install with: pip install xgboost")


class ModelTrainer:
    """
    XGBoost model trainer with GPU support.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize ModelTrainer.
        
        Args:
            use_gpu: Whether to use GPU for training
        """
        self.use_gpu = use_gpu
        self.model = None
        self.feature_names = None
        
        # Adjust XGBoost params based on device
        self.params = config.XGBOOST_PARAMS.copy()
        if use_gpu:
            # Try GPU-accelerated method first, fall back to hist if needed
            try:
                # Check if GPU is available
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=2)
                if result.returncode == 0:
                    self.params['tree_method'] = 'hist'  # 'hist' with device='cuda' uses GPU in XGBoost 2.0+
                    self.params['device'] = 'cuda'
                else:
                    raise FileNotFoundError("nvidia-smi not found")
            except:
                if config.VERBOSE:
                    print("⚠️  GPU not detected, falling back to CPU")
                self.params['tree_method'] = 'hist'
                self.params['device'] = 'cpu'
                self.use_gpu = False
        else:
            self.params['tree_method'] = 'hist'
            self.params['device'] = 'cpu'
        
        if config.VERBOSE:
            print(f"ModelTrainer initialized: GPU={use_gpu}")
            print(f"XGBoost params: {self.params}")
    
    def prepare_data(self, df: Union['cudf.DataFrame', 'pd.DataFrame'], 
                     target_col: str = 'tip_amount',
                     test_size: int = None) -> Tuple:
        """
        Prepare data for training: split into train/test (matching notebook workflow).
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Number of rows for testing (default: 1,000,000)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        if test_size is None:
            test_size = 1_000_000  # Default from notebook
        
        if config.VERBOSE:
            print(f"\n=== Preparing Data ===")
            print(f"Total samples: {len(df):,}")
        
        # Shuffle dataset (matching notebook)
        df = df.sample(frac=1, random_state=432).reset_index(drop=True)
        
        # Separate features and target
        # Drop datetime columns (they're already extracted as features)
        cols_to_drop = [target_col]
        if 'tpep_pickup_datetime' in df.columns:
            cols_to_drop.append('tpep_pickup_datetime')
        elif 'pickup_datetime' in df.columns:
            cols_to_drop.append('pickup_datetime')
        if 'tpep_dropoff_datetime' in df.columns:
            cols_to_drop.append('tpep_dropoff_datetime')
        elif 'dropoff_datetime' in df.columns:
            cols_to_drop.append('dropoff_datetime')
        
        feature_cols = [col for col in df.columns if col not in cols_to_drop]
        X = df[feature_cols]
        y = df[target_col]
        
        self.feature_names = feature_cols
        
        # Convert to numpy (XGBoost works with numpy arrays)
        if isinstance(X, cudf.DataFrame) if CUDF_AVAILABLE else False:
            X = X.to_pandas()
        if isinstance(y, cudf.Series) if CUDF_AVAILABLE else False:
            y = y.to_pandas()
        
        # Split: last test_size rows for testing, rest for training (matching notebook)
        X_train = X.iloc[:-test_size]
        X_test = X.iloc[-test_size:]
        y_train = y.iloc[:-test_size]
        y_test = y.iloc[-test_size:]
        
        if config.VERBOSE:
            print(f"Train samples: {len(X_train):,}")
            print(f"Test samples: {len(X_test):,}")
            print(f"Features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_with_kfold(self, X_train, y_train, X_test, y_test, 
                        n_splits: int = 5, learning_rate: Optional[float] = None,
                        n_estimators: Optional[int] = None) -> Tuple:
        """
        Train model with K-Fold validation (matching notebook workflow).
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            n_splits: Number of folds for K-Fold
            learning_rate: Learning rate (overrides config)
            n_estimators: Number of estimators (overrides config)
            
        Returns:
            Tuple of (model, predictions, test_mse, validation_mses)
        """
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=432)
        valid_mses = []
        
        if config.VERBOSE:
            print(f"\n=== Training with {n_splits}-Fold Validation ===")
        
        # Convert to pandas if needed for iloc access
        if not hasattr(X_train, 'iloc'):
            X_train = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
            y_train = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
        
        # K-Fold validation
        for fold, (train_idx, valid_idx) in enumerate(kf.split(np.arange(len(X_train)))):
            if config.VERBOSE:
                print(f"Fold {fold + 1}/{n_splits}...")
            
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[valid_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[valid_idx]
            
            # Create model with hyperparameters
            model_params = self.params.copy()
            if learning_rate is not None:
                model_params['learning_rate'] = learning_rate
            if n_estimators is not None:
                model_params['n_estimators'] = n_estimators
            
            model = xgb.XGBRegressor(**model_params, enable_categorical=True)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            valid_preds = model.predict(X_val)
            valid_mse = mean_squared_error(y_val, valid_preds)
            valid_mses.append(valid_mse)
        
        # Train final model on full training data
        if config.VERBOSE:
            print("Training final model on full training data...")
        
        final_params = self.params.copy()
        if learning_rate is not None:
            final_params['learning_rate'] = learning_rate
        if n_estimators is not None:
            final_params['n_estimators'] = n_estimators
        
        final_model = xgb.XGBRegressor(**final_params, enable_categorical=True)
        final_model.fit(X_train, y_train, verbose=False)
        
        # Test predictions
        test_preds = final_model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_preds)
        
        self.model = final_model
        
        if config.VERBOSE:
            valid_mse_mean = np.mean(valid_mses)
            valid_mse_std = np.std(valid_mses)
            print(f"\nValidation MSE: {valid_mse_mean:.6f} (±{valid_mse_std:.6f})")
            print(f"Test MSE: {test_mse:.6f}")
        
        return final_model, test_preds, test_mse, valid_mses
    
    def train(self, X_train, y_train, X_val, y_val) -> 'xgb.XGBRegressor':
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Trained XGBoost model
        """
        if config.VERBOSE:
            print(f"\n=== Training Model ===")
            print(f"Training samples: {len(X_train):,}")
            print(f"Validation samples: {len(X_val):,}")
        
        # Create DMatrix for XGBoost (optional, but can be faster)
        try:
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            use_dmatrix = True
        except:
            use_dmatrix = False
            if config.VERBOSE:
                print("Using numpy arrays directly (DMatrix not available)")
        
        # Train model
        if use_dmatrix:
            self.model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.params['n_estimators'],
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
                verbose_eval=50 if config.VERBOSE else False
            )
        else:
            # Use sklearn API
            self.model = xgb.XGBRegressor(**self.params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
                verbose=50 if config.VERBOSE else False
            )
        
        if config.VERBOSE:
            print("Training completed!")
            if hasattr(self.model, 'best_iteration'):
                print(f"Best iteration: {self.model.best_iteration}")
            elif hasattr(self.model, 'best_ntree_limit'):
                print(f"Best ntree limit: {self.model.best_ntree_limit}")
        
        return self.model
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions with trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            # XGBoost Booster object
            dmat = xgb.DMatrix(X)
            return self.model.predict(dmat)
    
    def save_model(self, filepath: Optional[str] = None):
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        if filepath is None:
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            filepath = os.path.join(config.MODEL_DIR, "xgb_taxi_tip_model.json")
        
        if hasattr(self.model, 'save_model'):
            # XGBoost Booster
            self.model.save_model(filepath)
        else:
            # sklearn API
            import pickle
            with open(filepath.replace('.json', '.pkl'), 'wb') as f:
                pickle.dump(self.model, f)
        
        if config.VERBOSE:
            print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        if filepath.endswith('.json'):
            self.model = xgb.Booster()
            self.model.load_model(filepath)
        else:
            import pickle
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
        
        if config.VERBOSE:
            print(f"Model loaded from: {filepath}")
