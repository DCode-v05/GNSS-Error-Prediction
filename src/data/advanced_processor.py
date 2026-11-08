from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

logger = logging.getLogger(__name__)

class AdvancedGNSSProcessor:
    def __init__(self, config: Dict[str, Any], save_dir: Optional[Union[str, Path]] = None):
        """Initialize the AdvancedGNSSProcessor.
        
        Args:
            config: Configuration dictionary containing processing parameters
            save_dir: Directory to save processed data and models
        """
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")
            
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else Path("data/processed")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        scaler_params = config.get('scaler_params', {})
        self.scaler = RobustScaler(**scaler_params)
        self.feature_processor = None
        self.target_columns = None
        self._fitted = False
        
    def _add_satellite_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add satellite-specific features.
        
        Args:
            df: Input DataFrame containing satellite data
            
        Returns:
            DataFrame with added satellite features
            
        Raises:
            ValueError: If required columns are missing
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
            
        if df.empty:
            return df
            
        df = df.copy()
        
        if 'sat_id' in df.columns:
            valid_sat_ids = df['sat_id'].notna()
            df.loc[valid_sat_ids, 'satellite_type'] = df.loc[valid_sat_ids, 'sat_id'].astype(str).str[0].map({
                'G': 'GEO',
                'M': 'MEO',
                'E': 'MEO',
                'C': 'MEO',
                'R': 'MEO',
                'J': 'MEO',
                'I': 'MEO'
            }).fillna('UNKNOWN')
            
            orbital_periods = {
                'GEO': 24.0,
                'MEO': 12.0,
                'UNKNOWN': 12.0
            }
            df['orbital_period'] = df['satellite_type'].map(orbital_periods)
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced temporal features."""
        if 'utc_time' not in df.columns:
            return df
            
        df['utc_time'] = pd.to_datetime(df['utc_time'])
        
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
        
        if 'orbital_period' in df.columns:
            df['orbital_phase'] = (df['hour'] % df['orbital_period']) / df['orbital_period']
            df['orbital_phase_sin'] = np.sin(2 * np.pi * df['orbital_phase'])
            df['orbital_phase_cos'] = np.cos(2 * np.pi * df['orbital_phase'])
        
        return df
    
    def _add_error_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add error-based features.
        
        Args:
            df: Input DataFrame containing satellite data
            
        Returns:
            DataFrame with added error-based features
            
        Notes:
            - Rolling statistics are calculated per satellite (grouped by 'sat_id')
            - NaN values in rolling statistics are filled with 0 for mean and 1 for std
              to maintain numerical stability
        """
        if df.empty:
            return df
            
        df = df.copy()
        
        error_cols = [col for col in df.columns if 'error' in col.lower()]
        if not error_cols:
            return df
        
        grouped = df.groupby('sat_id', group_keys=False)
        
        rolling_windows = self.config.get('data', {}).get('rolling_windows', [6, 12, 24])
        
        for col in error_cols:
            for window in rolling_windows:
                mean_col = f'{col}_rolling_mean_{window}'
                df[mean_col] = grouped[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                ).fillna(0)
                
                std_col = f'{col}_rolling_std_{window}'
                df[std_col] = grouped[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                ).fillna(1.0)
                
                df[std_col] = df[std_col].replace([np.inf, -np.inf], 1.0)
        
        for col in error_cols:
            diff1 = f'{col}_diff1'
            df[diff1] = grouped[col].diff().fillna(0)
            
            diff2 = f'{col}_diff2'
            df[diff2] = grouped[diff1].diff().fillna(0)
            
        return df
    
    def fit(self, df: pd.DataFrame, target_columns: List[str]) -> 'AdvancedGNSSProcessor':
        """Fit the processor to the training data.
        
        Args:
            df: Input DataFrame containing training data
            target_columns: List of column names to be used as targets
            
        Returns:
            self: Returns the instance itself
            
        Raises:
            ValueError: If input data is invalid or target columns are missing
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input must be a non-empty pandas DataFrame")
            
        if not isinstance(target_columns, list) or not all(isinstance(c, str) for c in target_columns):
            raise ValueError("target_columns must be a list of strings")
            
        if not all(col in df.columns for col in target_columns):
            missing = [col for col in target_columns if col not in df.columns]
            raise ValueError(f"Target columns not found in DataFrame: {missing}")
            
        self.target_columns = target_columns
        
        logger.info("Adding satellite features...")
        df = self._add_satellite_features(df)
        
        logger.info("Adding temporal features...")
        df = self._add_temporal_features(df)
        
        logger.info("Adding error features...")
        df = self._add_error_features(df)
    
        numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        categorical_features = [col for col in ['satellite_type'] if col in df.columns]
        
        non_feature_cols = ['utc_time', 'sat_id', 'timestamps'] + target_columns
        numeric_features = [col for col in numeric_features if col not in non_feature_cols]
        
        if not numeric_features and not categorical_features:
            raise ValueError("No valid features found after preprocessing")
    
        try:
            numeric_transformer = Pipeline(steps=[
                ('scaler', RobustScaler())
            ])
            
            transformers = [
                ('num', numeric_transformer, numeric_features)
            ]
            
            if categorical_features:
                categorical_transformer = Pipeline(steps=[
                    ('onehot', OneHotEncoder(
                        handle_unknown='ignore',
                        sparse_output=False
                    ))
                ])
                transformers.append(('cat', categorical_transformer, categorical_features))
            
            self.feature_processor = ColumnTransformer(
                transformers=transformers,
                remainder='drop',
                n_jobs=-1
            )
            
            X = df.drop(columns=non_feature_cols, errors='ignore')
            if X.empty:
                raise ValueError("No features remaining after preprocessing")
                
            logger.info(f"Fitting feature processor with {X.shape[1]} features...")
            self.feature_processor.fit(X)
            
            logger.info("Fitting target scaler...")
            self.scaler.fit(df[target_columns])
            
            self._fitted = True
            logger.info("Successfully fitted the processor")
            return self
            
        except Exception as e:
            self._fitted = False
            logger.error(f"Error during fitting: {str(e)}")
            raise
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform the input data.
        
        Args:
            df: Input DataFrame to transform
            
        Returns:
            Tuple of (transformed_features, transformed_targets) where
            transformed_targets is None if target columns are not present
            
        Raises:
            RuntimeError: If the processor has not been fitted
            ValueError: If input data is invalid
        """
        if not self._fitted:
            raise RuntimeError("Processor has not been fitted. Call fit() first.")
            
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Input must be a non-empty pandas DataFrame")
            
        try:
            df = self._add_satellite_features(df)
            df = self._add_temporal_features(df)
            df = self._add_error_features(df)
            
            non_feature_cols = ['utc_time', 'sat_id', 'timestamps'] + (self.target_columns if self.target_columns else [])
            X = df.drop(columns=non_feature_cols, errors='ignore')
            
            if X.empty:
                raise ValueError("No features remaining after preprocessing")
                
            has_targets = (self.target_columns is not None and 
                          all(col in df.columns for col in self.target_columns))
            
            y = df[self.target_columns].values if has_targets else None
            
            logger.info("Transforming features...")
            X_transformed = self.feature_processor.transform(X)
            
            if y is not None:
                y_scaled = self.scaler.transform(y)
                return X_transformed, y_scaled
            
            return X_transformed, None
            
        except Exception as e:
            logger.error(f"Error during transformation: {str(e)}")
            raise
    
    def inverse_transform_targets(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform the target values."""
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return self.scaler.inverse_transform(y)
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save the processor."""
        path = path or self.save_dir / 'advanced_processor.joblib'
        joblib.dump({
            'feature_processor': self.feature_processor,
            'scaler': self.scaler,
            'target_columns': self.target_columns,
            'config': self.config
        }, path)
    
    @classmethod
    def load(cls, path: Path) -> 'AdvancedGNSSProcessor':
        """Load a saved processor."""
        data = joblib.load(path)
        processor = cls(data['config'])
        processor.feature_processor = data['feature_processor']
        processor.scaler = data['scaler']
        processor.target_columns = data['target_columns']
        return processor
