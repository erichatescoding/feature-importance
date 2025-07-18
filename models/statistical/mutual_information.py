"""
Mutual Information Statistical Model

Part of the Feature Importance Evaluator framework.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from typing import Dict, Any, Tuple, Optional


class MutualInformationEvaluator:
    """Mutual Information evaluator for statistical feature importance."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize Mutual Information evaluator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        
    def calculate_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str
    ) -> pd.DataFrame:
        """
        Calculate mutual information scores for features.
        
        Args:
            X: Feature dataframe
            y: Target series
            problem_type: 'classification' or 'regression'
            
        Returns:
            DataFrame with features and importance scores
        """
        # Calculate mutual information
        if problem_type == 'classification':
            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
        
        # Normalize scores
        mi_scores = mi_scores / mi_scores.sum() if mi_scores.sum() > 0 else mi_scores
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df 