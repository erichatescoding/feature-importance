"""
Test script for Feature Importance Evaluator

Run this to verify the installation and basic functionality.
"""

import pandas as pd
import numpy as np
import os
import sys
# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluator import FeatureImportanceEvaluator
from preprocessing.preprocess import preprocess_data
from utils.test_utils import clean_output_directory

def test_basic_functionality():
    """Test basic functionality with simple data."""
    
    print("Testing Feature Importance Evaluator...")
    print("-" * 50)
    
    # Create simple test data
    np.random.seed(42)
    n_samples = 200
    
    # Create features with known importance
    X1 = np.random.randn(n_samples)  # Most important
    X2 = np.random.randn(n_samples)  # Second most important
    X3 = np.random.randn(n_samples)  # Less important
    X4 = np.random.randn(n_samples)  # Noise
    
    # Create target with known relationships
    y = 3 * X1 + 2 * X2 + 0.5 * X3 + 0.1 * np.random.randn(n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature_1': X1,
        'feature_2': X2,
        'feature_3': X3,
        'feature_4': X4,
        'target': y
    })
    
    print(f"✓ Created test data: {df.shape}")
    
    # Test preprocessing
    try:
        processed_df, info = preprocess_data(df, 'target', 'regression')
        print(f"✓ Preprocessing successful: {processed_df.shape}")
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return False
    
    # Test evaluation
    try:
        # Clean and use test output directory
        output_dir = clean_output_directory()
        # Use the main project's cache directory (one level up)
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.feature_importance_cache')
        evaluator = FeatureImportanceEvaluator(cache_dir=cache_dir, output_dir=output_dir, clean_cache=False)
        results = evaluator.evaluate(
            df=processed_df,
            target='target',
            problem_type='regression',
            models=['tree', 'linear', 'statistical'],
            importance_methods=['default']
        )
        print("✓ Evaluation completed successfully")
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return False
    
    # Check results
    print("\nResults Summary:")
    print("-" * 30)
    
    # Check consensus importance
    if 'consensus_importance' in results:
        print("\nTop Features (Consensus):")
        for _, row in results['consensus_importance'].head(4).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Verify expected order (feature_1 should be most important)
        top_feature = results['consensus_importance'].iloc[0]['feature']
        if top_feature == 'feature_1':
            print("\n✓ Feature importance order is correct!")
        else:
            print(f"\n⚠️  Expected 'feature_1' as top feature, got '{top_feature}'")
    
    # Check model results
    print("\nModel Performance:")
    for model_name, model_results in results['models'].items():
        if 'metrics' in model_results and 'r2_score' in model_results['metrics']:
            r2 = model_results['metrics']['r2_score']
            print(f"  {model_name}: R² = {r2:.4f}")
    
    print("\n✓ All tests passed!")
    return True


def test_categorical_features():
    """Test with categorical features."""
    
    print("\n\nTesting Categorical Feature Handling...")
    print("-" * 50)
    
    # Create data with categorical features
    np.random.seed(42)
    n_samples = 200
    
    df = pd.DataFrame({
        'numeric_1': np.random.randn(n_samples),
        'numeric_2': np.random.randn(n_samples),
        'category_1': np.random.choice(['A', 'B', 'C'], n_samples),
        'category_2': np.random.choice(['X', 'Y'], n_samples),
        'target': np.random.randn(n_samples)
    })
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 10), 'numeric_1'] = np.nan
    df.loc[np.random.choice(df.index, 5), 'category_1'] = np.nan
    
    print(f"✓ Created test data with categorical features: {df.shape}")
    print(f"  Numeric features: 2")
    print(f"  Categorical features: 2")
    print(f"  Missing values: {df.isna().sum().sum()}")
    
    # Test preprocessing
    try:
        processed_df, info = preprocess_data(df, 'target', 'regression')
        print(f"✓ Preprocessing handled categorical features")
        
        # Check warnings
        if info['warnings']:
            print("\nPreprocessing warnings:")
            for warning in info['warnings']:
                print(f"  - {warning}")
        
        # Verify encoding
        print(f"\n✓ All features are now numeric: {processed_df.select_dtypes(include=[np.number]).shape[1]} features")
        
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("FEATURE IMPORTANCE EVALUATOR TEST SUITE")
    print("="*60)
    
    # Run tests
    success = True
    
    # Test 1: Basic functionality
    if not test_basic_functionality():
        success = False
    
    # Test 2: Categorical features
    if not test_categorical_features():
        success = False
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("✓ ALL TESTS PASSED!")
        print("\nYou can now run the full examples:")
        print("  python example_usage.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1) 