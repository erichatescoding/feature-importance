"""
Boosted Tree Regression Model (CatBoost)

This module implements the CatBoost regressor for feature importance evaluation.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    
from ..utils import clean_catboost_info


class BoostedTreeRegressor:
    """CatBoost Regressor for feature importance evaluation."""
    
    def __init__(self, **params):
        """
        Initialize CatBoost Regressor.
        
        Args:
            **params: Parameters for CatBoostRegressor
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Install with: pip install catboost")
        
        default_params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'random_state': 42,
            'verbose': False
        }
        default_params.update(params)
        self.model = CatBoostRegressor(**default_params)
        self.feature_names = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model."""
        # Clean catboost_info directory before training
        clean_catboost_info()
        
        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X_test)
        
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2_score': r2_score(y_test, y_pred)
        }
        
        return metrics
        
    def get_feature_importance(self, importance_type: str = 'default') -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('default' uses built-in importance)
            
        Returns:
            DataFrame with features and importance scores
        """
        if importance_type == 'default':
            importance = self.model.get_feature_importance()
            
            # Normalize
            importance = importance / importance.sum()
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            raise ValueError(f"Importance type '{importance_type}' not supported")
            
    def get_model(self) -> 'CatBoostRegressor':
        """Get the underlying CatBoost model."""
        return self.model 