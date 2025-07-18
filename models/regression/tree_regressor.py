"""
Tree-based Regression Model (Random Forest)

Part of the Feature Importance Evaluator framework.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, Tuple, Optional


class TreeRegressor:
    """Random Forest Regressor for feature importance evaluation."""
    
    def __init__(self, **params):
        """
        Initialize Random Forest Regressor.
        
        Args:
            **params: Parameters for RandomForestRegressor
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(params)
        self.model = RandomForestRegressor(**default_params)
        self.feature_names = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model."""
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
            importance = self.model.feature_importances_
            
            # Normalize
            importance = importance / importance.sum()
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            raise ValueError(f"Importance type '{importance_type}' not supported")
            
    def get_model(self) -> RandomForestRegressor:
        """Get the underlying sklearn model."""
        return self.model 