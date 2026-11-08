"""
Utility functions for data loading and processing.
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def load_gnss_data(data_dir: Union[str, Path], 
                  file_pattern: str = "*.csv") -> Dict[str, pd.DataFrame]:
    """
    Load GNSS data from CSV files in the specified directory.
    
    Args:
        data_dir: Directory containing the data files
        file_pattern: Pattern to match data files (e.g., "*.csv")
        
    Returns:
        Dictionary mapping satellite IDs to DataFrames
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    data_files = list(data_dir.glob(file_pattern))
    if not data_files:
        raise FileNotFoundError(f"No files matching {file_pattern} found in {data_dir}")
    
    data = {}
    for file_path in data_files:
        try:
            sat_id = file_path.stem.split('_')[0]
            df = pd.read_csv(file_path)
            
            if 'utc_time' in df.columns:
                df['utc_time'] = pd.to_datetime(df['utc_time'])
                df = df.sort_values('utc_time')
            
            data[sat_id] = df
            logger.info(f"Loaded data for {sat_id} from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            continue
    
    return data

def preprocess_gnss_data(df: pd.DataFrame, 
                        target_columns: List[str],
                        drop_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Preprocess GNSS data by handling missing values and feature engineering.
    
    Args:
        df: Input DataFrame
        target_columns: List of target column names
        drop_columns: Columns to drop (optional)
        
    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()
    
    if drop_columns:
        df = df.drop(columns=[col for col in drop_columns if col in df.columns])
    
    for col in df.columns:
        if df[col].isnull().any():
            if col in target_columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            else:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def create_sequences(data: np.ndarray, 
                    sequence_length: int, 
                    target_length: int = 1,
                    step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences of data for time series prediction.
    
    Args:
        data: Input data array of shape (samples, features)
        sequence_length: Length of input sequences
        target_length: Length of target sequences
        step: Step size between sequences
        
    Returns:
        Tuple of (sequences, targets)
    """
    sequences = []
    targets = []
    
    for i in range(0, len(data) - sequence_length - target_length + 1, step):
        sequences.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length:i + sequence_length + target_length])
    
    return np.array(sequences), np.array(targets)

def split_train_test(data: Dict[str, pd.DataFrame], 
                    test_size: float = 0.2,
                    time_based: bool = True) -> Tuple[Dict, Dict]:
    """
    Split data into training and testing sets.
    
    Args:
        data: Dictionary mapping satellite IDs to DataFrames
        test_size: Fraction of data to use for testing
        time_based: If True, split based on time (last portion)
                   If False, use random split
                   
    Returns:
        Tuple of (train_data, test_data) dictionaries
    """
    train_data = {}
    test_data = {}
    
    for sat_id, df in data.items():
        if time_based and 'utc_time' in df.columns:
            split_idx = int(len(df) * (1 - test_size))
            train_data[sat_id] = df.iloc[:split_idx].copy()
            test_data[sat_id] = df.iloc[split_idx:].copy()
        else:
            mask = np.random.rand(len(df)) < (1 - test_size)
            train_data[sat_id] = df[mask].copy()
            test_data[sat_id] = df[~mask].copy()
    
    return train_data, test_data

def save_data(data: Dict, output_dir: Union[str, Path]) -> None:
    """
    Save processed data to disk.
    
    Args:
        data: Dictionary of DataFrames to save
        output_dir: Directory to save the data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, df in data.items():
        output_path = output_dir / f"{name}.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {name} to {output_path}")

def load_processed_data(data_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Load processed data from disk.
    
    Args:
        data_dir: Directory containing the processed data files
        
    Returns:
        Dictionary mapping dataset names to DataFrames
    """
    data_dir = Path(data_dir)
    data = {}
    
    for file_path in data_dir.glob("*.parquet"):
        name = file_path.stem
        data[name] = pd.read_parquet(file_path)
        logger.info(f"Loaded {name} from {file_path}")
    
    return data
