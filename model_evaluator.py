"""
Model evaluation module.
Provides comprehensive evaluation metrics and visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional
import config

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Some metrics may not work.")


class ModelEvaluator:
    """
    Model evaluation and visualization.
    """
    
    def __init__(self):
        """Initialize ModelEvaluator."""
        self.results = {}
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 dataset_name: str = "test") -> dict:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            dataset_name: Name of dataset (for logging)
            
        Returns:
            Dictionary of metrics
        """
        if not SKLEARN_AVAILABLE:
            # Manual calculation
            mse = np.mean((y_true - y_pred) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - y_pred))
            r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        else:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mean_actual = np.mean(y_true)
        mean_predicted = np.mean(y_pred)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        
        metrics = {
            'dataset': dataset_name,
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'mape': mape,
            'mean_actual': mean_actual,
            'mean_predicted': mean_predicted,
        }
        
        self.results[dataset_name] = metrics
        
        if config.VERBOSE:
            print(f"\n=== Evaluation Results ({dataset_name.upper()}) ===")
            print(f"RMSE:  ${rmse:.4f}")
            print(f"MAE:   ${mae:.4f}")
            print(f"MSE:   ${mse:.4f}")
            print(f"R²:    {r2:.4f}")
            print(f"MAPE:  {mape:.2f}%")
            print(f"Mean Actual:    ${mean_actual:.4f}")
            print(f"Mean Predicted: ${mean_predicted:.4f}")
        
        return metrics
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                        dataset_name: str = "test", save_path: Optional[str] = None):
        """
        Create visualization plots for predictions.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            dataset_name: Name of dataset
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Evaluation: {dataset_name.upper()}', fontsize=16, fontweight='bold')
        
        # 1. Scatter plot: Predicted vs Actual
        ax1 = axes[0, 0]
        ax1.scatter(y_true, y_pred, alpha=0.5, s=1)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Tip Amount ($)')
        ax1.set_ylabel('Predicted Tip Amount ($)')
        ax1.set_title('Predicted vs Actual')
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals plot
        ax2 = axes[0, 1]
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=1)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Tip Amount ($)')
        ax2.set_ylabel('Residuals ($)')
        ax2.set_title('Residuals Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. Distribution of residuals
        ax3 = axes[1, 0]
        ax3.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='r', linestyle='--', lw=2)
        ax3.set_xlabel('Residuals ($)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Residuals')
        ax3.grid(True, alpha=0.3)
        
        # 4. Error distribution by tip amount
        ax4 = axes[1, 1]
        # Bin actual values and calculate mean error per bin
        bins = np.linspace(y_true.min(), y_true.max(), 20)
        bin_indices = np.digitize(y_true, bins)
        bin_errors = [np.mean(np.abs(residuals[bin_indices == i])) 
                     for i in range(1, len(bins))]
        bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
        ax4.plot(bin_centers, bin_errors, marker='o', linewidth=2, markersize=4)
        ax4.set_xlabel('Actual Tip Amount ($)')
        ax4.set_ylabel('Mean Absolute Error ($)')
        ax4.set_title('Error by Tip Amount')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            os.makedirs(config.RESULTS_DIR, exist_ok=True)
            save_path = os.path.join(config.RESULTS_DIR, f'evaluation_{dataset_name}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if config.VERBOSE:
            print(f"Evaluation plot saved to: {save_path}")
        
        plt.close()
    
    def plot_feature_importance(self, model, feature_names: list, 
                               top_n: int = 20, save_path: Optional[str] = None):
        """
        Plot feature importance.
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
            top_n: Number of top features to show
            save_path: Path to save plot
        """
        # Get feature importance
        if hasattr(model, 'get_score'):
            # XGBoost Booster
            importance_dict = model.get_score(importance_type='gain')
            # Convert to list
            importances = [importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))]
        elif hasattr(model, 'feature_importances_'):
            # sklearn API
            importances = model.feature_importances_
        else:
            print("Cannot extract feature importance from model.")
            return
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), [importances[i] for i in indices])
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance (Gain)')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path is None:
            os.makedirs(config.RESULTS_DIR, exist_ok=True)
            save_path = os.path.join(config.RESULTS_DIR, 'feature_importance.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if config.VERBOSE:
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.close()
    
    def save_results(self, filepath: Optional[str] = None):
        """Save evaluation results to file."""
        if filepath is None:
            os.makedirs(config.RESULTS_DIR, exist_ok=True)
            filepath = os.path.join(config.RESULTS_DIR, 'evaluation_results.txt')
        
        with open(filepath, 'w') as f:
            f.write("=== Model Evaluation Results ===\n\n")
            for dataset_name, metrics in self.results.items():
                f.write(f"\n{dataset_name.upper()}:\n")
                f.write("-" * 40 + "\n")
                for key, value in metrics.items():
                    if key != 'dataset':
                        if isinstance(value, float):
                            f.write(f"{key.upper()}: {value:.4f}\n")
                        else:
                            f.write(f"{key.upper()}: {value}\n")
        
        if config.VERBOSE:
            print(f"Results saved to: {filepath}")
