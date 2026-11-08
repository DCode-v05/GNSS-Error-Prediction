from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import Dict, List, Tuple, Optional
import logging
import torch
from sklearn.feature_selection import f_regression
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GNSSDataProcessor:
    def __init__(self, save_dir: Optional[Path] = None):
        self.error_scaler = RobustScaler()
        self.time_scaler = RobustScaler()
        self.rolling_scaler = RobustScaler()
        self.derivative_scaler = RobustScaler()
        self.selected_features = None
        self.save_dir = save_dir
        self.min_vals = None
        self.max_vals = None
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate the raw data."""
        try:
            df = pd.read_csv(data_path)
            df['utc_time'] = pd.to_datetime(df['utc_time'])
            logger.info(f"Loaded data from {data_path} with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical time-based features."""
        df['hour'] = df['utc_time'].dt.hour
        df['day_of_week'] = df['utc_time'].dt.dayofweek
        df['day_of_month'] = df['utc_time'].dt.day
        df['month'] = df['utc_time'].dt.month
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        df['hours_since_start'] = (df['utc_time'] - df['utc_time'].min()).dt.total_seconds() / 3600
        
        logger.info("Added cyclical time features")
        return df

    def add_rolling_features(self, df: pd.DataFrame, windows: List[int] = [6, 12, 24]) -> pd.DataFrame:
        """Add rolling statistics features."""
        error_cols = [col for col in df.columns if 'error' in col.lower()]
        
        for col in error_cols:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
        
        df = df.fillna(0)
        logger.info(f"Added rolling features with windows {windows}")
        return df

    def add_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate first and second derivatives of error columns."""
        error_cols = [col for col in df.columns if 'error' in col.lower()]
        
        for col in error_cols:
            df[f'{col}_diff1'] = df[col].diff().fillna(0)
            df[f'{col}_diff2'] = df[f'{col}_diff1'].diff().fillna(0)
            
            clip_value = df[col].std() * 3
            df[f'{col}_diff1'] = df[f'{col}_diff1'].clip(-clip_value, clip_value)
            df[f'{col}_diff2'] = df[f'{col}_diff2'].clip(-clip_value, clip_value)
        
        logger.info("Added derivative features")
        return df

    def validate_data(self, df: pd.DataFrame, stage: str) -> None:
        """Validate data at different stages of preprocessing."""
        logger.info(f"Validating data at stage: {stage}")
        logger.info(f"Shape: {df.shape}")
        
        nan_cols = df.isna().sum()
        if nan_cols.any():
            logger.warning(f"NaN values found in columns: {nan_cols[nan_cols > 0]}")
            
        inf_cols = df.isin([np.inf, -np.inf]).sum()
        if inf_cols.any():
            logger.warning(f"Infinite values found in columns: {inf_cols[inf_cols > 0]}")
            
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            logger.info(f"Data statistics at {stage}:")
            logger.info(f"Mean: {df[numeric_cols].mean().mean():.3f}")
            logger.info(f"Std: {df[numeric_cols].std().mean():.3f}")
            logger.info(f"Min: {df[numeric_cols].min().min():.3f}")
            logger.info(f"Max: {df[numeric_cols].max().max():.3f}")

    def transform_error_features(self, df: pd.DataFrame, error_cols: List[str]) -> pd.DataFrame:
        """Apply appropriate transformations to error features."""
        df = df.copy()
        
        for col in error_cols:
            signs = np.sign(df[col].values)
            abs_values = np.abs(df[col].values)
            
            transformed = np.log1p(abs_values)
            
            df[col] = transformed * signs
            
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            is_outlier = (df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))
            df[f'{col}_is_outlier'] = is_outlier.astype(int)
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Normalize different feature groups with appropriate transformations."""
        df = df.copy()
        
        if 'utc_time' in df.columns:
            df = df.drop('utc_time', axis=1)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype('float64')
        
        df = df.replace([np.inf, -np.inf], [1e10, -1e10])
        
        error_cols = [col for col in df.columns if 'error' in col.lower() and not col.endswith('_is_outlier')]
        time_cols = [col for col in df.columns if any(x in col.lower() for x in ['sin', 'cos', 'hour', 'day', 'month', 'hours_since'])]
        rolling_cols = [col for col in df.columns if 'rolling' in col.lower()]
        derivative_cols = [col for col in df.columns if 'diff' in col.lower()]
        
        if error_cols:
            df = self.transform_error_features(df, error_cols)
        
        if is_training:
            if error_cols:
                self.error_scaler.fit(df[error_cols])
            if time_cols:
                self.time_scaler.fit(df[time_cols])
            if rolling_cols:
                self.rolling_scaler.fit(df[rolling_cols])
            if derivative_cols:
                self.derivative_scaler.fit(df[derivative_cols])
        
        if error_cols:
            df[error_cols] = self.error_scaler.transform(df[error_cols])
        if time_cols:
            df[time_cols] = self.time_scaler.transform(df[time_cols])
        if rolling_cols:
            df[rolling_cols] = self.rolling_scaler.transform(df[rolling_cols])
        if derivative_cols:
            df[derivative_cols] = self.derivative_scaler.transform(df[derivative_cols])
        
        if is_training:
            logger.info("\nFeature statistics after transformation:")
            for col in df.columns:
                stats = df[col].describe()
                logger.info(f"\n{col}:")
                logger.info(f"  Mean: {stats['mean']:.6f}")
                logger.info(f"  Std: {stats['std']:.6f}")
                logger.info(f"  Min: {stats['min']:.6f}")
                logger.info(f"  Max: {stats['max']:.6f}")
            
            self.min_vals = df.min().values
            self.max_vals = df.max().values
            
            if self.save_dir:
                self.save_normalization_params()
        
        return df

    def save_normalization_params(self):
        """
        Save the min and max values of the normalized features to a JSON file.
        """
        if self.save_dir and self.min_vals is not None and self.max_vals is not None:
            norm_params = {
                'min_vals': self.min_vals.tolist(),
                'max_vals': self.max_vals.tolist()
            }
            with open(self.save_dir / 'norm_params.json', 'w') as f:
                json.dump(norm_params, f, indent=4)
            logger.info(f"Normalization parameters saved to {self.save_dir / 'norm_params.json'}")

    def create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length):
            seq = data[i:i + sequence_length]
            target = data[i + sequence_length][-1]
            sequences.append(seq)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)

    def preprocess_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Preprocess data with enhanced features."""
        self.validate_data(df, "raw")
        
        df = self.add_time_features(df)
        
        df = self.add_rolling_features(df)
        
        df = self.add_derivatives(df)
        
        self.validate_data(df, "feature_engineering")
        
        df = self.normalize_features(df, is_training)
        
        self.validate_data(df, "normalization")
        
        return df

    def prepare_data(self, data_path: str, sequence_length: int, train_split: float = 0.8) -> Dict:
        """Prepare data for training and validation."""
        df = self.load_data(data_path)
        df = self.preprocess_data(df, is_training=True)
        
        data = df.values.astype('float32')
        
        X, y = self.create_sequences(data, sequence_length)
        
        train_size = int(len(X) * train_split)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        y_train = torch.FloatTensor(y_train)
        y_val = torch.FloatTensor(y_val)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val
        }
        y_val = torch.FloatTensor(y_val)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val
        }