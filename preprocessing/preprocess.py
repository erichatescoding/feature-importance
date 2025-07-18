"""
Data Preprocessing Module

Handles automated preprocessing for feature importance evaluation including:
- Missing value imputation
- Categorical encoding
- Feature type detection
- Data validation and warnings
- Caching for performance

Usage:
    from preprocessing.preprocess import preprocess_data
    
    processed_df, info = preprocess_data(
        df=your_dataframe,
        target='target_column',
        problem_type='regression'  # or 'classification'
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Union, List, Any
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import hashlib
import os
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cache_utils import clean_cache_directory

# Import PySpark if available
try:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql.functions import col, count, when, isnan, isnull
    from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler as SparkStandardScaler
    from pyspark.ml import Pipeline
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    SparkDataFrame = type(None)


class DataPreprocessor:
    """
    Automated data preprocessing for feature importance evaluation.
    """
    
    def __init__(self, cache_dir: str = ".preprocessing_cache", clean_cache: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            cache_dir: Directory for caching preprocessed data
            clean_cache: Whether to clean the cache directory on initialization
        """
        self.cache_dir = cache_dir
        
        if clean_cache:
            clean_cache_directory(cache_dir, "preprocessing cache")
        elif not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        self.encoders = {}
        self.imputers = {}
        self.scalers = {}
        self.feature_types = {}
        self.warnings_log = []
    
    def preprocess(
        self,
        df: Union[pd.DataFrame, SparkDataFrame],
        target: str,
        problem_type: str,
        scale_features: bool = False,
        max_cardinality: int = 100,
        min_samples_per_class: int = 10
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Preprocess data for feature importance evaluation.
        
        Args:
            df: Input dataframe
            target: Target column name
            problem_type: 'classification' or 'regression'
            scale_features: Whether to scale numeric features
            max_cardinality: Maximum cardinality for categorical features
            min_samples_per_class: Minimum samples per class for classification
            
        Returns:
            Tuple of (processed_dataframe, preprocessing_info)
        """
        
        print("Starting data preprocessing...")
        print(f"Problem type: {problem_type}")
        print(f"Target column: {target}")
        
        # Convert Spark DataFrame if needed
        if SPARK_AVAILABLE and isinstance(df, SparkDataFrame):
            df = self._preprocess_spark(df, target, problem_type)
        
        # Check cache
        cache_key = self._generate_cache_key(df, target, problem_type, scale_features)
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            print("Loaded preprocessed data from cache")
            return cached_result
        
        # Reset state
        self.warnings_log = []
        
        # Validate target column
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target]
        X = df[feature_cols].copy()
        y = df[target].copy()
        
        # Handle missing target values
        if problem_type == 'regression':
            # For regression, impute with median
            if y.isna().any():
                n_missing = y.isna().sum()
                median_value = y.median()
                y.fillna(median_value, inplace=True)
                self.warnings_log.append(
                    f"Imputed {n_missing} missing target values with median: {median_value:.4f}"
                )
        else:
            # For classification, remove rows with missing labels
            if y.isna().any():
                n_missing = y.isna().sum()
                valid_idx = ~y.isna()
                X = X[valid_idx]
                y = y[valid_idx]
                self.warnings_log.append(
                    f"Removed {n_missing} rows with missing target values"
                )
        
        # Detect feature types
        print("\nDetecting feature types...")
        self.feature_types = self._detect_feature_types(X)
        
        # Report feature types
        n_numeric = len(self.feature_types['numeric'])
        n_categorical = len(self.feature_types['categorical'])
        print(f"Found {n_numeric} numeric and {n_categorical} categorical features")
        
        # Process numeric features
        if self.feature_types['numeric']:
            print("\nProcessing numeric features...")
            X = self._process_numeric_features(X, self.feature_types['numeric'], scale_features)
        
        # Process categorical features
        if self.feature_types['categorical']:
            print("\nProcessing categorical features...")
            X = self._process_categorical_features(X, self.feature_types['categorical'], max_cardinality)
        
        # Validate processed data
        print("\nValidating processed data...")
        self._validate_processed_data(X, y, problem_type, min_samples_per_class)
        
        # Combine features and target
        processed_df = pd.concat([X, y], axis=1)
        
        # Create preprocessing info
        info = {
            'original_shape': df.shape,
            'processed_shape': processed_df.shape,
            'feature_types': self.feature_types,
            'numeric_features': self.feature_types['numeric'],
            'categorical_features': self.feature_types['categorical'],
            'n_features': len(feature_cols),
            'n_samples': len(processed_df),
            'target_stats': self._get_target_stats(y, problem_type),
            'warnings': self.warnings_log,
            'preprocessing_params': {
                'scale_features': scale_features,
                'max_cardinality': max_cardinality,
                'min_samples_per_class': min_samples_per_class
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache results
        self._save_to_cache(cache_key, (processed_df, info))
        
        # Print summary
        self._print_preprocessing_summary(info)
        
        return processed_df, info
    
    def _detect_feature_types(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect numeric and categorical features."""
        
        numeric_features = []
        categorical_features = []
        
        for col in X.columns:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(X[col]):
                # Check if it's actually categorical (low cardinality integers)
                unique_ratio = X[col].nunique() / len(X)
                if unique_ratio < 0.05 and X[col].nunique() < 20:
                    categorical_features.append(col)
                else:
                    numeric_features.append(col)
            else:
                categorical_features.append(col)
        
        return {
            'numeric': numeric_features,
            'categorical': categorical_features
        }
    
    def _process_numeric_features(
        self,
        X: pd.DataFrame,
        numeric_cols: List[str],
        scale: bool
    ) -> pd.DataFrame:
        """Process numeric features."""
        
        for col in numeric_cols:
            # Handle missing values
            if X[col].isna().any():
                n_missing = X[col].isna().sum()
                missing_pct = n_missing / len(X) * 100
                
                # Use median imputation
                imputer = SimpleImputer(strategy='median')
                X[col] = imputer.fit_transform(X[[col]]).ravel()
                self.imputers[col] = imputer
                
                self.warnings_log.append(
                    f"Imputed {n_missing} ({missing_pct:.1f}%) missing values in '{col}' with median"
                )
        
        # Scale features if requested
        if scale:
            scaler = StandardScaler()
            X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            self.scalers['numeric'] = scaler
            print(f"Scaled {len(numeric_cols)} numeric features")
        
        return X
    
    def _process_categorical_features(
        self,
        X: pd.DataFrame,
        categorical_cols: List[str],
        max_cardinality: int
    ) -> pd.DataFrame:
        """Process categorical features."""
        
        for col in categorical_cols:
            # Handle missing values
            if X[col].isna().any():
                n_missing = X[col].isna().sum()
                missing_pct = n_missing / len(X) * 100
                X[col].fillna('unknown', inplace=True)
                
                self.warnings_log.append(
                    f"Filled {n_missing} ({missing_pct:.1f}%) missing values in '{col}' with 'unknown'"
                )
            
            # Check cardinality
            n_unique = X[col].nunique()
            if n_unique > max_cardinality:
                self.warnings_log.append(
                    f"High cardinality warning: '{col}' has {n_unique} unique values"
                )
            
            # Encode categorical features
            if X[col].dtype == 'object' or X[col].dtype == 'category':
                # Convert to string to handle mixed types
                X[col] = X[col].astype(str)
                
                # Use LabelEncoder
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col])
                self.encoders[col] = encoder
                
                print(f"Encoded '{col}' with {n_unique} unique values")
        
        return X
    
    def _validate_processed_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
        min_samples_per_class: int
    ):
        """Validate the processed data and generate warnings."""
        
        # Check for remaining missing values
        missing_features = X.columns[X.isna().any()].tolist()
        if missing_features:
            self.warnings_log.append(
                f"Warning: Features still contain missing values: {missing_features}"
            )
        
        # Check for constant features
        constant_features = []
        for col in X.columns:
            if X[col].nunique() == 1:
                constant_features.append(col)
        
        if constant_features:
            self.warnings_log.append(
                f"Warning: Found {len(constant_features)} constant features: {constant_features[:5]}..."
            )
        
        # Check class balance for classification
        if problem_type == 'classification':
            class_counts = y.value_counts()
            
            # Check for imbalanced classes
            min_class_count = class_counts.min()
            max_class_count = class_counts.max()
            imbalance_ratio = max_class_count / min_class_count
            
            if imbalance_ratio > 10:
                self.warnings_log.append(
                    f"Severe class imbalance detected: ratio = {imbalance_ratio:.1f}"
                )
            elif imbalance_ratio > 3:
                self.warnings_log.append(
                    f"Class imbalance detected: ratio = {imbalance_ratio:.1f}"
                )
            
            # Check for rare classes
            rare_classes = class_counts[class_counts < min_samples_per_class]
            if len(rare_classes) > 0:
                self.warnings_log.append(
                    f"Found {len(rare_classes)} classes with < {min_samples_per_class} samples"
                )
    
    def _get_target_stats(self, y: pd.Series, problem_type: str) -> Dict[str, Any]:
        """Get statistics about the target variable."""
        
        if problem_type == 'classification':
            value_counts = y.value_counts()
            return {
                'type': 'classification',
                'n_classes': y.nunique(),
                'class_distribution': value_counts.to_dict(),
                'class_balance': {
                    'min_samples': int(value_counts.min()),
                    'max_samples': int(value_counts.max()),
                    'imbalance_ratio': float(value_counts.max() / value_counts.min())
                }
            }
        else:
            return {
                'type': 'regression',
                'mean': float(y.mean()),
                'median': float(y.median()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max()),
                'q25': float(y.quantile(0.25)),
                'q75': float(y.quantile(0.75))
            }
    
    def _preprocess_spark(
        self,
        df: SparkDataFrame,
        target: str,
        problem_type: str
    ) -> pd.DataFrame:
        """Preprocess Spark DataFrame before converting to Pandas."""
        
        print("Preprocessing Spark DataFrame...")
        
        # Check size before conversion
        row_count = df.count()
        if row_count > 1000000:
            warnings.warn(
                f"Large Spark DataFrame ({row_count} rows). Consider sampling for better performance."
            )
            # Sample if too large
            fraction = min(1.0, 1000000 / row_count)
            df = df.sample(fraction=fraction, seed=42)
            print(f"Sampled {fraction*100:.1f}% of data")
        
        # Convert to Pandas
        return df.toPandas()
    
    def _generate_cache_key(
        self,
        df: pd.DataFrame,
        target: str,
        problem_type: str,
        scale_features: bool
    ) -> str:
        """Generate a unique cache key for the preprocessing configuration."""
        
        # Create a string representation of the configuration
        config_str = f"{df.shape}_{target}_{problem_type}_{scale_features}_{list(df.columns)}"
        
        # Add sample of data to ensure data changes are detected
        sample_data = df.head(100).values.flatten()
        config_str += str(sample_data)
        
        # Generate hash
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Tuple[pd.DataFrame, Dict]]:
        """Load preprocessed data from cache if available."""
        
        cache_path = os.path.join(self.cache_dir, f'preprocessed_{cache_key}.pkl')
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                warnings.warn(f"Failed to load from cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: Tuple[pd.DataFrame, Dict]):
        """Save preprocessed data to cache."""
        
        cache_path = os.path.join(self.cache_dir, f'preprocessed_{cache_key}.pkl')
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            warnings.warn(f"Failed to save to cache: {e}")
    
    def _print_preprocessing_summary(self, info: Dict[str, Any]):
        """Print a summary of preprocessing results."""
        
        print("\n" + "="*50)
        print("PREPROCESSING SUMMARY")
        print("="*50)
        
        print(f"Original shape: {info['original_shape']}")
        print(f"Processed shape: {info['processed_shape']}")
        print(f"Numeric features: {len(info['numeric_features'])}")
        print(f"Categorical features: {len(info['categorical_features'])}")
        
        if info['warnings']:
            print("\nWarnings:")
            for warning in info['warnings']:
                print(f"  ⚠️  {warning}")
        
        print("\nTarget Statistics:")
        target_stats = info['target_stats']
        if target_stats['type'] == 'classification':
            print(f"  Classes: {target_stats['n_classes']}")
            print(f"  Imbalance ratio: {target_stats['class_balance']['imbalance_ratio']:.2f}")
        else:
            print(f"  Mean: {target_stats['mean']:.4f}")
            print(f"  Std: {target_stats['std']:.4f}")
            print(f"  Range: [{target_stats['min']:.4f}, {target_stats['max']:.4f}]")


# Convenience function
def preprocess_data(
    df: Union[pd.DataFrame, SparkDataFrame],
    target: str,
    problem_type: str,
    **kwargs
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to preprocess data.
    
    Args:
        df: Input dataframe
        target: Target column name
        problem_type: 'classification' or 'regression'
        **kwargs: Additional arguments passed to DataPreprocessor.preprocess()
        
    Returns:
        Tuple of (processed_dataframe, preprocessing_info)
    """
    preprocessor = DataPreprocessor()
    return preprocessor.preprocess(df, target, problem_type, **kwargs) 