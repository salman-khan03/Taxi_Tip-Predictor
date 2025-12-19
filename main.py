"""
Main training script for Taxi Tip Predictor.
This script orchestrates the entire pipeline: data loading, preprocessing,
feature engineering, model training, and evaluation.
"""

import os
import sys
import warnings
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config
import numpy as np
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator

warnings.filterwarnings('ignore')


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("TAXI TIP PREDICTOR - ML Training Pipeline")
    print("=" * 60)
    print()
    
    try:
        # Step 1: Initialize components
        print("[*] Initializing components...")
        data_loader = DataLoader(
            use_cudf=config.USE_CUDF,
            enable_rmm=config.ENABLE_RMM
        )
        feature_engineer = FeatureEngineer(use_cudf=config.USE_CUDF)
        model_trainer = ModelTrainer(use_gpu=config.XGBOOST_PARAMS.get('device') == 'cuda')
        evaluator = ModelEvaluator()
        
        # Step 2: Load data
        print("\n[*] Loading data...")
        df = data_loader.load_data(sample_size=config.SAMPLE_SIZE)
        
        # Step 3: Initial cleanup (missing values)
        print("\n[*] Initial cleanup - handling missing values...")
        df = data_loader.clean_data_initial(df)
        
        # Step 4: Feature engineering (includes date extraction)
        print("\n[*] Engineering features...")
        df = feature_engineer.create_features(df)
        
        # Step 5: Remove anomalies
        print("\n[*] Removing anomalies...")
        df = data_loader.clean_data_anomalies(df)
        
        # Step 6: Prepare data for training
        print("\n[*] Preparing data splits...")
        X_train, X_test, y_train, y_test, feature_names = \
            model_trainer.prepare_data(df, target_col='tip_amount')
        
        # Step 7: Train model with K-Fold validation
        print("\n[*] Training model with K-Fold validation...")
        model, y_test_pred, test_mse, valid_mses = model_trainer.train_with_kfold(
            X_train, y_train, X_test, y_test
        )
        
        # Step 8: Evaluate on test set
        print("\n[*] Evaluating on test set...")
        # Convert to numpy if needed for evaluation
        if hasattr(y_test, 'values'):
            y_test = y_test.values
        if hasattr(y_test_pred, 'values'):
            y_test_pred = y_test_pred.values
        
        test_metrics = evaluator.evaluate(y_test, y_test_pred, dataset_name="test")
        evaluator.plot_predictions(y_test, y_test_pred, dataset_name="test")
        
        # Print validation results
        if valid_mses:
            valid_mse_mean = np.mean(valid_mses)
            valid_mse_std = np.std(valid_mses)
            print(f"\n[*] K-Fold Validation Results:")
            print(f"   Mean Validation MSE: {valid_mse_mean:.6f}")
            print(f"   Std Validation MSE:  {valid_mse_std:.6f}")
            print(f"   Test MSE:             {test_mse:.6f}")
        
        # Step 9: Feature importance
        print("\n[*] Analyzing feature importance...")
        evaluator.plot_feature_importance(model, feature_names)
        
        # Step 10: Save results
        print("\n[*] Saving results...")
        evaluator.save_results()
        
        if config.SAVE_MODEL:
            model_trainer.save_model()
        
        if config.SAVE_PREDICTIONS:
            os.makedirs(config.RESULTS_DIR, exist_ok=True)
            predictions_path = os.path.join(config.RESULTS_DIR, 'test_predictions.npy')
            np.save(predictions_path, y_test_pred)
            if config.VERBOSE:
                print(f"Predictions saved to: {predictions_path}")
        
        # Final summary
        print("\n" + "=" * 60)
        print("[SUCCESS] TRAINING COMPLETE!")
        print("=" * 60)
        print(f"\n[*] Final Test Results:")
        print(f"   RMSE: ${test_metrics['rmse']:.4f}")
        print(f"   MAE:  ${test_metrics['mae']:.4f}")
        print(f"   R²:   {test_metrics['r2']:.4f}")
        print(f"\n[*] Results saved in: {config.RESULTS_DIR}/")
        print(f"[*] Model saved in:   {config.MODEL_DIR}/")
        print()
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\n[INFO] Make sure to:")
        print("   1. Download the NYC Taxi dataset")
        print("   2. Place it in the 'data/' directory")
        print("   3. Update DATASET_FILE in config.py if needed")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
