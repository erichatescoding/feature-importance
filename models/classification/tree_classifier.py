"""
Tree-based Classification Model (Random Forest)

Part of the Feature Importance Evaluator framework.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, Any, Tuple, Optional


class TreeClassifier:
    """Random Forest Classifier for feature importance evaluation."""
    
    def __init__(self, **params):
        """
        Initialize Random Forest Classifier.
        
        Args:
            **params: Parameters for RandomForestClassifier
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(params)
        self.model = RandomForestClassifier(**default_params)
        self.feature_names = None
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train the model."""
        self.feature_names = list(X_train.columns)
        self.model.fit(X_train, y_train)
        
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X_test)
        
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X_test)
        
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Add probabilities if available
        if hasattr(self.model, 'predict_proba'):
            metrics['y_proba'] = self.predict_proba(X_test)
            
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
            
    def get_model(self) -> RandomForestClassifier:
        """Get the underlying sklearn model."""
        return self.model 