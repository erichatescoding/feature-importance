"""
Example usage of the Feature Importance Evaluator

This script demonstrates how to use the Feature Importance Evaluator
for both classification and regression tasks.
"""

import pandas as pd
import numpy as np
from evaluator import FeatureImportanceEvaluator
from preprocessing.preprocess import preprocess_data
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# EXAMPLE 1: REGRESSION TASK
# ============================================================================

def regression_example():
    """Example using regression with house price data."""
    
    print("="*60)
    print("REGRESSION EXAMPLE: House Price Prediction")
    print("="*60)
    
    # Create sample house price data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'square_feet': np.random.randint(800, 4000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'age_years': np.random.randint(0, 50, n_samples),
        'garage_spaces': np.random.randint(0, 4, n_samples),
        'lot_size': np.random.randint(4000, 20000, n_samples),
        'property_tax': np.random.uniform(2000, 15000, n_samples),
        'crime_rate': np.random.uniform(0.1, 8.0, n_samples),
        'school_rating': np.random.uniform(3.0, 10.0, n_samples),
        'distance_to_downtown': np.random.uniform(1, 30, n_samples),
        'has_pool': np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7]),
        'has_fireplace': np.random.choice(['yes', 'no'], n_samples, p=[0.4, 0.6]),
        'neighborhood': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'walkability_score': np.random.uniform(20, 100, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic house prices based on features
    base_price = 50000
    price = (
        base_price +
        df['square_feet'] * 150 +
        df['bedrooms'] * 15000 +
        df['bathrooms'] * 20000 +
        df['garage_spaces'] * 12000 +
        df['lot_size'] * 3 +
        (df['has_pool'] == 'yes') * 25000 +
        (df['has_fireplace'] == 'yes') * 8000 +
        df['school_rating'] * 15000 +
        df['walkability_score'] * 500 +
        -df['age_years'] * 2000 +
        -df['crime_rate'] * 8000 +
        -df['distance_to_downtown'] * 2000 +
        -df['property_tax'] * 3 +
        np.random.normal(0, 25000, n_samples)
    )
    
    price = np.maximum(price, 50000)
    df['price'] = price
    
    # Add some missing values for demonstration
    df.loc[np.random.choice(df.index, 50, replace=False), 'property_tax'] = np.nan
    df.loc[np.random.choice(df.index, 30, replace=False), 'walkability_score'] = np.nan
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target: price (${df['price'].mean():,.0f} average)")
    
    # Step 1: Preprocess the data
    print("\n" + "-"*40)
    print("Step 1: Preprocessing Data")
    print("-"*40)
    
    processed_df, preprocessing_info = preprocess_data(
        df=df,
        target='price',
        problem_type='regression'
    )
    
    # Step 2: Evaluate feature importance
    print("\n" + "-"*40)
    print("Step 2: Evaluating Feature Importance")
    print("-"*40)
    
    evaluator = FeatureImportanceEvaluator()
    
    results = evaluator.evaluate(
        df=processed_df,
        target='price',
        problem_type='regression',
        models=['tree', 'linear', 'statistical'],  # Skip boosted_tree if CatBoost not installed
        importance_methods=['default', 'permutation'],
        sample_size_for_permutation=500  # Use smaller sample for faster demo
    )
    
    print("\nTop 5 Most Important Features (Consensus):")
    for _, row in results['consensus_importance'].head(5).iterrows():
        print(f"  {row['rank']}. {row['feature']:20s} {row['importance']:.4f}")
    
    return results


# ============================================================================
# EXAMPLE 2: CLASSIFICATION TASK
# ============================================================================

def classification_example():
    """Example using classification with customer churn data."""
    
    print("\n\n" + "="*60)
    print("CLASSIFICATION EXAMPLE: Customer Churn Prediction")
    print("="*60)
    
    # Create sample customer churn data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'num_services': np.random.randint(1, 8, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
        'tech_support': np.random.choice(['Yes', 'No', 'No service'], n_samples),
        'online_security': np.random.choice(['Yes', 'No', 'No service'], n_samples),
        'device_protection': np.random.choice(['Yes', 'No', 'No service'], n_samples),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No service'], n_samples),
        'streaming_movies': np.random.choice(['Yes', 'No', 'No service'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create churn based on features (with some randomness)
    churn_probability = (
        0.1 +  # Base probability
        (df['contract_type'] == 'Month-to-month') * 0.3 +
        (df['tenure_months'] < 12) * 0.2 +
        (df['monthly_charges'] > 80) * 0.15 +
        (df['num_services'] < 3) * 0.1 +
        (df['tech_support'] == 'No') * 0.1 +
        (df['online_security'] == 'No') * 0.1 +
        np.random.uniform(-0.2, 0.2, n_samples)
    )
    
    churn_probability = np.clip(churn_probability, 0, 1)
    df['churn'] = (np.random.uniform(0, 1, n_samples) < churn_probability).astype(int)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target: churn (Rate: {df['churn'].mean():.1%})")
    
    # Step 1: Preprocess the data
    print("\n" + "-"*40)
    print("Step 1: Preprocessing Data")
    print("-"*40)
    
    processed_df, preprocessing_info = preprocess_data(
        df=df,
        target='churn',
        problem_type='classification'
    )
    
    # Step 2: Evaluate feature importance
    print("\n" + "-"*40)
    print("Step 2: Evaluating Feature Importance")
    print("-"*40)
    
    evaluator = FeatureImportanceEvaluator()
    
    # Use all available models
    results = evaluator.evaluate(
        df=processed_df,
        target='churn',
        problem_type='classification',
        models=['tree', 'linear', 'statistical'],
        importance_methods=['default'],  # Just default for faster demo
        sample_size_for_permutation=500
    )
    
    print("\nTop 5 Most Important Features (Consensus):")
    for _, row in results['consensus_importance'].head(5).iterrows():
        print(f"  {row['rank']}. {row['feature']:20s} {row['importance']:.4f}")
    
    return results


# ============================================================================
# EXAMPLE 3: USING EXISTING RANDOM FOREST CODE
# ============================================================================

def integrate_with_existing_code():
    """Example showing how to integrate with existing Random Forest code."""
    
    print("\n\n" + "="*60)
    print("INTEGRATION EXAMPLE: Using Existing Random Forest Code")
    print("="*60)
    
    # Import existing functions
    from legacy.random_forest_regressor import run_random_forest_regressor
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Simple dataset for demonstration
    X = np.random.randn(n_samples, 5)
    feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    
    # Create target with known relationships
    y = (
        2.0 * X[:, 0] +      # feature_1 is most important
        1.5 * X[:, 1] +      # feature_2 is second most important
        0.5 * X[:, 2] +      # feature_3 has some importance
        0.1 * X[:, 3] +      # feature_4 has little importance
        0.0 * X[:, 4] +      # feature_5 has no importance
        np.random.randn(n_samples) * 0.5  # Add noise
    )
    
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Use existing Random Forest function
    print("\nUsing existing Random Forest regressor...")
    rf_results = run_random_forest_regressor(df, 'target')
    
    print(f"\nR² Score from existing code: {rf_results['r2_score']:.4f}")
    
    # Now use the new Feature Importance Evaluator for comparison
    print("\nUsing new Feature Importance Evaluator...")
    
    # No preprocessing needed for this simple numeric data
    evaluator = FeatureImportanceEvaluator()
    
    new_results = evaluator.evaluate(
        df=df,
        target='target',
        problem_type='regression',
        models=['tree'],  # Just Random Forest for comparison
        importance_methods=['default']
    )
    
    print("\nFeature Importance Comparison:")
    print("-"*50)
    print("Feature        | Existing RF  | New Evaluator")
    print("-"*50)
    
    # Get importance from existing code
    existing_importance = rf_results['feature_importance'].set_index('feature')['importance']
    
    # Get importance from new evaluator
    new_importance = new_results['models']['tree']['importance']['default'].set_index('feature')['importance']
    
    for feature in feature_names:
        existing = existing_importance.get(feature, 0)
        new = new_importance.get(feature, 0)
        print(f"{feature:14s} | {existing:12.4f} | {new:12.4f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run regression example
    regression_results = regression_example()
    
    # Run classification example
    classification_results = classification_example()
    
    # Show integration with existing code
    integrate_with_existing_code()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nCheck the generated PNG files for visualizations:")
    print("- feature_importance_heatmap_*.png")
    print("- consensus_importance_*.png")
    print("- *_performance_*.png")
    print("\nThe evaluator has created a comprehensive analysis of feature importance")
    print("across multiple models with automated preprocessing and visualization.") 