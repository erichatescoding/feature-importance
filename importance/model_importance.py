"""
Model Feature Importance Extraction

Handles extraction of default feature importance from trained models.
"""

import pandas as pd
import numpy as np
from typing import Any


def get_model_importance(model: Any, model_type: str) -> pd.DataFrame:
    """
    Extract default feature importance from a trained model.
    
    Args:
        model: Trained model instance
        model_type: Type of model ('tree', 'boosted_tree', 'linear')
        
    Returns:
        DataFrame with features and importance scores
    """
    
    if hasattr(model, 'get_feature_importance'):
        # Use model's built-in method
        return model.get_feature_importance('default')
    
    # Fallback for models without the method
    if model_type in ['tree', 'boosted_tree']:
        if hasattr(model.model, 'feature_importances_'):
            importance = model.model.feature_importances_
        else:
            raise ValueError(f"Model {model_type} does not have feature_importances_")
    elif model_type == 'linear':
        if hasattr(model.model, 'coef_'):
            # Use absolute values of coefficients
            coef = model.model.coef_
            if len(coef.shape) > 1:
                # Multi-class classification
                importance = np.abs(coef).mean(axis=0)
            else:
                importance = np.abs(coef)
        else:
            raise ValueError("Linear model does not have coefficients")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Normalize importance
    importance = importance / importance.sum() if importance.sum() > 0 else importance
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': model.feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df 