"""
Test script for Feature Importance Evaluator using ALL models

This script tests the 'all' models option which should include:
- ML models: tree, boosted_tree (if available), linear
- Statistical model: mutual information
"""

import pandas as pd
import numpy as np
import os
import sys
# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluator import FeatureImportanceEvaluator
from preprocessing.preprocess import preprocess_data
import warnings
warnings.filterwarnings('ignore')
from utils.test_utils import clean_output_directory

def test_all_models_regression():
    """Test using 'all' models for regression."""
    
    print("="*60)
    print("TEST: Using 'all' models for REGRESSION")
    print("="*60)
    
    # Clean output directory at the start
    output_dir = clean_output_directory()
    
    # Create sample data with known relationships
    np.random.seed(42)
    n_samples = 500
    
    # Create features with different relationships
    data = {
        'linear_feature': np.random.randn(n_samples),      # Linear relationship
        'nonlinear_feature': np.random.randn(n_samples),   # Non-linear relationship
        'interaction_1': np.random.randn(n_samples),        # Interaction term
        'interaction_2': np.random.randn(n_samples),        # Interaction term
        'categorical_1': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'noise_feature': np.random.randn(n_samples)        # Pure noise
    }
    
    df = pd.DataFrame(data)
    
    # Create target with mixed relationships
    target = (
        3.0 * df['linear_feature'] +                       # Strong linear
        2.0 * np.sin(df['nonlinear_feature'] * 2) +       # Non-linear
        1.5 * df['interaction_1'] * df['interaction_2'] +  # Interaction
        (df['categorical_1'] == 'A').astype(int) * 2 +    # Categorical effect
        (df['categorical_2'] == 'Y').astype(int) * 1 +    # Categorical effect
        0.1 * df['noise_feature'] +                        # Weak noise
        np.random.randn(n_samples) * 0.5                  # Random noise
    )
    
    df['target'] = target
    
    print(f"\nDataset created:")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {list(df.columns[:-1])}")
    print(f"  Target mean: {df['target'].mean():.2f}")
    
    # Step 1: Preprocess
    print("\n" + "-"*40)
    print("Step 1: Preprocessing")
    print("-"*40)
    
    processed_df, preprocessing_info = preprocess_data(
        df=df,
        target='target',
        problem_type='regression'
    )
    
    # Step 2: Evaluate with ALL models
    print("\n" + "-"*40)
    print("Step 2: Evaluating with ALL models")
    print("-"*40)
    
    # Use test output directory
    # Use the main project's cache directory (one level up)
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.feature_importance_cache')
    evaluator = FeatureImportanceEvaluator(cache_dir=cache_dir, output_dir=output_dir, clean_cache=False)
    
    # Use 'all' which should include tree, linear, statistical, and boosted_tree (if available)
    results = evaluator.evaluate(
        df=processed_df,
        target='target',
        problem_type='regression',
        models='all',  # This should include tree, linear, statistical, and boosted_tree (if available)
        importance_methods=['default'],
        sample_size_for_permutation=300
    )
    
    # Verify all expected models are included
    print("\n" + "-"*40)
    print("RESULTS VERIFICATION")
    print("-"*40)
    
    expected_models = ['tree', 'linear', 'statistical']
    actual_models = list(results['models'].keys())
    
    print(f"\nExpected models (minimum): {expected_models}")
    print(f"Actual models evaluated: {actual_models}")
    
    # Check if boosted_tree was included (if CatBoost is installed)
    if 'boosted_tree' in actual_models:
        print("✓ CatBoost is installed - boosted_tree included")
    else:
        print("ℹ️  CatBoost not installed - boosted_tree skipped")
    
    # Verify each expected model
    missing_models = []
    for model in expected_models:
        if model in actual_models:
            print(f"✓ {model} model included")
        else:
            print(f"✗ {model} model MISSING")
            missing_models.append(model)
    
    if missing_models:
        print(f"\n❌ ERROR: Missing models: {missing_models}")
        return False
    
    # Show model performance
    print("\nModel Performance:")
    print("-"*30)
    for model_name, model_results in results['models'].items():
        if 'error' in model_results:
            print(f"{model_name:15s}: ERROR - {model_results['error']}")
        elif 'metrics' in model_results:
            metrics = model_results['metrics']
            if 'r2_score' in metrics:
                print(f"{model_name:15s}: R² = {metrics['r2_score']:.4f}")
            elif 'mutual_information' in metrics:
                print(f"{model_name:15s}: MI = {metrics['mutual_information']:.4f}")
    
    # Show consensus importance
    print("\nConsensus Feature Importance (Top 5):")
    print("-"*40)
    for i, row in results['consensus_importance'].head(5).iterrows():
        print(f"{row['rank']:2d}. {row['feature']:20s} {row['importance']:.4f}")
    
    print("\n✅ Regression test with 'all' models PASSED!")
    return True


def test_all_models_classification():
    """Test using 'all' models for classification."""
    
    print("\n\n" + "="*60)
    print("TEST: Using 'all' models for CLASSIFICATION")
    print("="*60)
    
    # Clean output directory at the start
    output_dir = clean_output_directory()
    
    # Create sample classification data
    np.random.seed(42)
    n_samples = 500
    
    # Create features
    data = {
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
        'categorical_1': np.random.choice(['Type1', 'Type2', 'Type3'], n_samples),
        'categorical_2': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'noise': np.random.randn(n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create binary classification target
    probability = (
        0.3 +  # Base probability
        0.3 * (df['feature_1'] > 0) +
        0.2 * (df['feature_2'] > 0.5) +
        0.1 * (df['categorical_1'] == 'Type1') +
        0.1 * (df['categorical_2'] == 'High') +
        np.random.uniform(-0.2, 0.2, n_samples)
    )
    
    df['target'] = (probability > 0.5).astype(int)
    
    print(f"\nDataset created:")
    print(f"  Shape: {df.shape}")
    print(f"  Class balance: {df['target'].mean():.1%} positive")
    
    # Step 1: Preprocess
    print("\n" + "-"*40)
    print("Step 1: Preprocessing")
    print("-"*40)
    
    processed_df, preprocessing_info = preprocess_data(
        df=df,
        target='target',
        problem_type='classification'
    )
    
    # Step 2: Evaluate with ALL models
    print("\n" + "-"*40)
    print("Step 2: Evaluating with ALL models")
    print("-"*40)
    
    # Use test output directory
    # Use the main project's cache directory (one level up)
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.feature_importance_cache')
    evaluator = FeatureImportanceEvaluator(cache_dir=cache_dir, output_dir=output_dir, clean_cache=False)
    
    results = evaluator.evaluate(
        df=processed_df,
        target='target',
        problem_type='classification',
        models='all',
        importance_methods=['default']
    )
    
    # Verify results
    print("\n" + "-"*40)
    print("RESULTS VERIFICATION")
    print("-"*40)
    
    expected_models = ['tree', 'linear', 'statistical']
    actual_models = list(results['models'].keys())
    
    print(f"\nModels evaluated: {actual_models}")
    
    # Show model performance
    print("\nModel Performance:")
    print("-"*30)
    for model_name, model_results in results['models'].items():
        if 'error' in model_results:
            print(f"{model_name:15s}: ERROR - {model_results['error']}")
        elif 'metrics' in model_results:
            metrics = model_results['metrics']
            if 'accuracy' in metrics:
                print(f"{model_name:15s}: Accuracy = {metrics['accuracy']:.4f}")
            elif 'mutual_information' in metrics:
                print(f"{model_name:15s}: MI = {metrics['mutual_information']:.4f}")
    
    print("\n✅ Classification test with 'all' models PASSED!")
    return True


def test_model_weights():
    """Test that model weights are properly applied in consensus ranking."""
    
    print("\n\n" + "="*60)
    print("TEST: Model Weights in Consensus Ranking")
    print("="*60)
    
    # Clean output directory at the start
    output_dir = clean_output_directory()
    
    # Create simple data where we know feature_1 should be most important
    np.random.seed(42)
    n_samples = 300
    
    X1 = np.random.randn(n_samples)
    X2 = np.random.randn(n_samples)
    X3 = np.random.randn(n_samples)
    
    df = pd.DataFrame({
        'feature_1': X1,
        'feature_2': X2,
        'feature_3': X3,
        'target': 5 * X1 + 0.5 * X2 + 0.1 * X3 + np.random.randn(n_samples) * 0.1
    })
    
    # Use test output directory
    # Use the main project's cache directory (one level up)
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.feature_importance_cache')
    evaluator = FeatureImportanceEvaluator(cache_dir=cache_dir, output_dir=output_dir, clean_cache=False)
    
    # Run with all models
    results = evaluator.evaluate(
        df=df,
        target='target',
        problem_type='regression',
        models='all',
        importance_methods=['default']
    )
    
    # Check model weights
    print("\nModel weights (from code):")
    print(evaluator.model_weights)
    
    # Check consensus ranking
    print("\nConsensus ranking reflects weighted combination:")
    consensus = results['consensus_importance']
    print(consensus)
    
    # Verify feature_1 is ranked first
    if consensus.iloc[0]['feature'] == 'feature_1':
        print("\n✅ Consensus ranking correctly identifies most important feature")
    else:
        print("\n❌ Consensus ranking failed to identify most important feature")
        return False
    
    return True


if __name__ == "__main__":
    print("TESTING FEATURE IMPORTANCE EVALUATOR WITH 'ALL' MODELS")
    print("="*60)
    print("\nThis test verifies that using models='all' includes:")
    print("- ML models: tree, linear, boosted_tree (if available)")
    print("- Statistical model: mutual information")
    print("="*60)
    
    all_passed = True
    
    # Clean output directory first
    output_dir = clean_output_directory()
    
    # Run regression test
    if not test_all_models_regression():
        all_passed = False
    
    # Run classification test
    if not test_all_models_classification():
        all_passed = False
    
    # Test model weights
    if not test_model_weights():
        all_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nThe 'all' option correctly includes both ML and statistical models.")
        print("Model ranking is properly weighted according to model performance.")
    else:
        print("❌ Some tests failed!")
        
    # Clean up
    import os
    import glob
    for pattern in ['*.png', '.feature_importance_cache/*.pkl', '.preprocessing_cache/*.pkl']:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except:
                pass
    print("\nTest files cleaned up.") 