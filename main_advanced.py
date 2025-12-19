"""
Advanced main training script with hyperparameter tuning.
This script includes K-Fold validation and hyperparameter tuning.
"""

import os
import sys
import warnings
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config
import numpy as np
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from hyperparameter_tuning import HyperparameterTuner

warnings.filterwarnings('ignore')


def main():
    """Advanced training pipeline with hyperparameter tuning."""
    print("=" * 60)
    print("🚕 TAXI TIP PREDICTOR - Advanced Training Pipeline")
    print("=" * 60)
    print()
    
    try:
        # Step 1: Initialize components
        print("📦 Initializing components...")
        data_loader = DataLoader(
            use_cudf=config.USE_CUDF,
            enable_rmm=config.ENABLE_RMM
        )
        feature_engineer = FeatureEngineer(use_cudf=config.USE_CUDF)
        model_trainer = ModelTrainer(use_gpu=config.XGBOOST_PARAMS.get('device') == 'cuda')
        evaluator = ModelEvaluator()
        tuner = HyperparameterTuner(model_trainer)
        
        # Step 2: Load data
        print("\n📊 Loading data...")
        df = data_loader.load_data(sample_size=config.SAMPLE_SIZE)
        
        # Step 3: Initial cleanup (missing values)
        print("\n🧹 Initial cleanup - handling missing values...")
        df = data_loader.clean_data_initial(df)
        
        # Step 4: Feature engineering (includes date extraction)
        print("\n🔧 Engineering features...")
        df = feature_engineer.create_features(df)
        
        # Step 5: Remove anomalies
        print("\n🚦 Removing anomalies...")
        df = data_loader.clean_data_anomalies(df)
        
        # Step 6: Prepare data for training
        print("\n📐 Preparing data splits...")
        X_train, X_test, y_train, y_test, feature_names = \
            model_trainer.prepare_data(df, target_col='tip_amount')
        
        # Step 7: Initial training run (baseline)
        print("\n🤖 Training baseline model with K-Fold validation...")
        model1, predictions1, mse1, valid_mses1 = model_trainer.train_with_kfold(
            X_train, y_train, X_test, y_test
        )
        
        valid_mse1 = float(np.mean(valid_mses1))
        valid_mse_std1 = float(np.std(valid_mses1))
        test_mse1 = float(mse1)
        
        print(f"\n📊 Baseline Model Results:")
        print(f"   Validation MSE: {valid_mse1:.6f} (±{valid_mse_std1:.6f})")
        print(f"   Test MSE:       {test_mse1:.6f}")
        
        # Step 8: Hyperparameter tuning
        print("\n🧩 Performing hyperparameter tuning...")
        results_df = tuner.tune_hyperparameters(
            X_train, y_train, X_test, y_test,
            learning_rates=[0.1, 0.05],
            n_estimators_list=[200, 500, 1000]
        )
        
        # Plot tuning results
        tuner.plot_tuning_results(results_df)
        
        # Step 9: Train best model
        print("\n🏆 Training best model...")
        best_lr = results_df.iloc[0]['learning_rate']
        best_ne = int(results_df.iloc[0]['n_estimators'])
        
        best_model, best_predictions, best_mse, best_valid_mses = \
            model_trainer.train_with_kfold(
                X_train, y_train, X_test, y_test,
                learning_rate=best_lr,
                n_estimators=best_ne
            )
        
        # Step 10: Evaluate best model
        print("\n📈 Evaluating best model...")
        if hasattr(y_test, 'values'):
            y_test = y_test.values
        if hasattr(best_predictions, 'values'):
            best_predictions = best_predictions.values
        
        best_metrics = evaluator.evaluate(y_test, best_predictions, dataset_name="best_model")
        evaluator.plot_predictions(y_test, best_predictions, dataset_name="best_model")
        
        # Step 11: Feature importance
        print("\n📊 Analyzing feature importance...")
        evaluator.plot_feature_importance(best_model, feature_names)
        
        # Step 12: Save results
        print("\n💾 Saving results...")
        evaluator.save_results()
        
        if config.SAVE_MODEL:
            model_trainer.save_model()
        
        # Final summary
        print("\n" + "=" * 60)
        print("✅ ADVANCED TRAINING COMPLETE!")
        print("=" * 60)
        print(f"\n📊 Final Results:")
        print(f"   Baseline Test MSE:  {test_mse1:.6f}")
        print(f"   Best Test MSE:      {best_metrics['rmse']**2:.6f}")
        print(f"   Improvement:        {((test_mse1 - best_metrics['rmse']**2) / test_mse1 * 100):.2f}%")
        print(f"\n🏆 Best Hyperparameters:")
        print(f"   Learning Rate:      {best_lr}")
        print(f"   N Estimators:       {best_ne}")
        print(f"\n📁 Results saved in: {config.RESULTS_DIR}/")
        print(f"📁 Model saved in:   {config.MODEL_DIR}/")
        print()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure to:")
        print("   1. Download the NYC Taxi dataset")
        print("   2. Place it in the 'data/' directory")
        print("   3. Update DATASET_FILE in config.py if needed")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
