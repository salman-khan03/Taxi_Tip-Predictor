"""
Data loading and preprocessing module.
Handles loading NYC Taxi data with cuDF (GPU) or pandas (CPU fallback),
memory management with RMM, and initial data cleaning.
"""

import os
import warnings
from typing import Optional, Union
import numpy as np

try:
    import cudf
    import rmm
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False
    import pandas as pd
    warnings.warn("cuDF not available. Falling back to pandas (CPU).")

import config


class DataLoader:
    """
    Data loader with GPU acceleration support.
    Handles memory management and data preprocessing.
    """
    
    def __init__(self, use_cudf: Optional[bool] = None, enable_rmm: bool = True):
        """
        Initialize DataLoader.
        
        Args:
            use_cudf: Whether to use cuDF (GPU). If None, uses config setting.
            enable_rmm: Whether to enable RMM memory pool.
        """
        self.use_cudf = use_cudf if use_cudf is not None else config.USE_CUDF
        self.enable_rmm = enable_rmm and config.ENABLE_RMM
        
        # Initialize RMM if available and enabled
        if self.use_cudf and CUDF_AVAILABLE and self.enable_rmm:
            self._init_rmm()
        
        if config.VERBOSE:
            print(f"DataLoader initialized: cuDF={self.use_cudf and CUDF_AVAILABLE}, RMM={self.enable_rmm}")
    
    def _init_rmm(self):
        """Initialize RMM memory pool for GPU memory management."""
        try:
            rmm.reinitialize(
                pool_allocator=True,
                initial_pool_size=config.RMM_INITIAL_POOL_SIZE,
                maximum_pool_size=config.RMM_POOL_SIZE
            )
            if config.VERBOSE:
                print(f"RMM initialized: pool_size={config.RMM_POOL_SIZE}")
        except Exception as e:
            warnings.warn(f"Failed to initialize RMM: {e}. Continuing without RMM.")
            self.enable_rmm = False
    
    def load_data(self, filepath: Optional[str] = None, sample_size: Optional[int] = None) -> Union['cudf.DataFrame', 'pd.DataFrame']:
        """
        Load taxi data from CSV file.
        
        Args:
            filepath: Path to CSV file. If None, uses config default.
            sample_size: Number of rows to sample. If None, loads all data.
            
        Returns:
            DataFrame (cuDF or pandas)
        """
        if filepath is None:
            filepath = os.path.join(config.DATA_DIR, config.DATASET_FILE)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        if config.VERBOSE:
            print(f"Loading data from: {filepath}")
        
        # Use cuDF if available and enabled
        if self.use_cudf and CUDF_AVAILABLE:
            df = cudf.read_csv(filepath)
        else:
            df = pd.read_csv(filepath)
        
        # Sample data if requested
        if sample_size is not None and len(df) > sample_size:
            if config.VERBOSE:
                print(f"Sampling {sample_size:,} rows from {len(df):,} total rows")
            df = df.sample(n=sample_size, random_state=config.RANDOM_STATE)
        
        if config.VERBOSE:
            print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
            print(f"Memory usage: {self._get_memory_usage(df):.2f} MB")
        
        return df
    
    def _get_memory_usage(self, df) -> float:
        """Get memory usage of DataFrame in MB."""
        if self.use_cudf and CUDF_AVAILABLE:
            return df.memory_usage(deep=True).sum() / 1024**2
        else:
            return df.memory_usage(deep=True).sum() / 1024**2
    
    def clean_data_initial(self, df: Union['cudf.DataFrame', 'pd.DataFrame']) -> Union['cudf.DataFrame', 'pd.DataFrame']:
        """
        Initial cleanup: handle missing values (matching notebook workflow).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if config.VERBOSE:
            print("\n=== Initial Cleanup - Missing Values ===")
            print(f"Initial shape: {df.shape}")
            print(f"Missing values:\n{df.isna().sum()}")
        
        # Fill fees with 0 (missing means no fee was charged)
        if 'airport_fee' in df.columns:
            df['airport_fee'] = df['airport_fee'].fillna(0)
        if 'congestion_surcharge' in df.columns:
            df['congestion_surcharge'] = df['congestion_surcharge'].fillna(0)
        
        # Fill passenger_count with mean (missing means driver forgot to enter)
        if 'passenger_count' in df.columns:
            mean_passengers = df['passenger_count'].mean()
            df['passenger_count'] = df['passenger_count'].fillna(mean_passengers)
            if config.VERBOSE:
                print(f"Filled passenger_count missing values with mean: {mean_passengers:.2f}")
        
        # Fill RatecodeID with 99 (unknown label)
        if 'RatecodeID' in df.columns:
            df['RatecodeID'] = df['RatecodeID'].fillna(99)
        
        # Drop store_and_fwd_flag (not meaningful for tip prediction)
        if 'store_and_fwd_flag' in df.columns:
            df = df.drop(columns=['store_and_fwd_flag'])
            if config.VERBOSE:
                print("Dropped store_and_fwd_flag column (not meaningful for tip prediction)")
        
        if config.VERBOSE:
            print(f"After initial cleanup - Missing values:\n{df.isna().sum()}")
        
        return df
    
    def clean_data_anomalies(self, df: Union['cudf.DataFrame', 'pd.DataFrame']) -> Union['cudf.DataFrame', 'pd.DataFrame']:
        """
        Remove anomalies: negative values, unrealistic distances, etc.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if config.VERBOSE:
            print("\n=== Data Optimization - Finding Anomalies ===")
            print("Summary statistics:")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary = df[numeric_cols].agg(['min', 'max', 'mean']).T
                print(summary)
        
        initial_rows = len(df)
        
        # Remove unrealistic trip distances (0.5 < distance < 100 miles)
        if 'trip_distance' in df.columns:
            before = len(df)
            df = df[df['trip_distance'] > 0.5]
            df = df[df['trip_distance'] < 100]
            if config.VERBOSE:
                print(f"Removed {before - len(df):,} rows with unrealistic trip distances")
        
        # Remove all negative values from numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            before = len(df)
            # Keep rows where all numeric columns are >= 0
            mask = ~df[numeric_cols].lt(0).any(axis=1)
            df = df[mask]
            if config.VERBOSE:
                print(f"Removed {before - len(df):,} rows with negative values")
        
        # Handle total_amount anomaly (minimum should be ~$3.70 based on NYC base fare)
        if 'total_amount' in df.columns:
            before = len(df)
            df = df[df['total_amount'] > 3.7]
            if config.VERBOSE:
                print(f"Removed {before - len(df):,} rows with total_amount <= $3.70")
        
        # Reset index after filtering
        df = df.reset_index(drop=True)
        
        if config.VERBOSE:
            print(f"\nRemoved {initial_rows - len(df):,} rows total")
            print(f"Final shape: {df.shape}")
            print("\nAfter anomaly removal - Summary statistics:")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary = df[numeric_cols].agg(['min', 'max', 'mean']).T
                print(summary)
        
        return df
    
    def clean_data(self, df: Union['cudf.DataFrame', 'pd.DataFrame']) -> Union['cudf.DataFrame', 'pd.DataFrame']:
        """
        Complete data cleaning pipeline (initial cleanup + anomalies).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = self.clean_data_initial(df)
        df = self.clean_data_anomalies(df)
        return df
    
    def to_pandas(self, df: Union['cudf.DataFrame', 'pd.DataFrame']) -> 'pd.DataFrame':
        """Convert cuDF DataFrame to pandas if needed."""
        if self.use_cudf and CUDF_AVAILABLE and isinstance(df, cudf.DataFrame):
            return df.to_pandas()
        return df
