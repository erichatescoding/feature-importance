"""
Permutation Feature Importance Implementation

This module implements permutation feature importance, which measures
feature importance by shuffling feature values and measuring the decrease
in model performance.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, f1_score
import warnings


def get_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    problem_type: str,
    sample_size: Optional[int] = None,
    model_type: Optional[str] = None,
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate permutation feature importance.
    
    Args:
        model: Trained model with predict method
        X: Feature dataframe
        y: Target values
        problem_type: 'regression' or 'classification'
        sample_size: Number of samples to use (None for all)
        model_type: Type of model (for special handling)
        n_repeats: Number of times to permute each feature
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with feature names and importance scores
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Sample data if needed
    if sample_size is not None and len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[indices].copy()
        y_sample = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
    else:
        X_sample = X.copy()
        y_sample = y
    
    # Get baseline score
    baseline_score = _compute_score(model, X_sample, y_sample, problem_type)
    
    # Calculate importance for each feature
    feature_importances = {}
    
    for feature in X.columns:
        feature_scores = []
        
        for _ in range(n_repeats):
            # Create a copy of the data
            X_permuted = X_sample.copy()
            
            # Shuffle the feature
            X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
            
            # Calculate score with permuted feature
            permuted_score = _compute_score(model, X_permuted, y_sample, problem_type)
            
            # Calculate importance as decrease in performance
            importance = baseline_score - permuted_score
            feature_scores.append(importance)
        
        # Average importance across repeats
        feature_importances[feature] = np.mean(feature_scores)
    
    # Create DataFrame with results
    importance_df = pd.DataFrame({
        'feature': list(feature_importances.keys()),
        'importance': list(feature_importances.values())
    })
    
    # Normalize importances to sum to 1 (if all positive)
    if (importance_df['importance'] >= 0).all():
        total_importance = importance_df['importance'].sum()
        if total_importance > 0:
            importance_df['importance'] = importance_df['importance'] / total_importance
    else:
        # If there are negative importances, scale to [0, 1]
        min_imp = importance_df['importance'].min()
        max_imp = importance_df['importance'].max()
        if max_imp > min_imp:
            importance_df['importance'] = (importance_df['importance'] - min_imp) / (max_imp - min_imp)
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    return importance_df


def _compute_score(model: Any, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], problem_type: str) -> float:
    """
    Compute model score based on problem type.
    
    Args:
        model: Trained model
        X: Features
        y: Target
        problem_type: 'regression' or 'classification'
        
    Returns:
        Score (higher is better)
    """
    try:
        predictions = model.predict(X)
        
        if problem_type == 'regression':
            # Use R² score (higher is better)
            return r2_score(y, predictions)
        else:
            # Use accuracy for classification
            return accuracy_score(y, predictions)
            
    except Exception as e:
        warnings.warn(f"Error computing score: {e}")
        return 0.0 