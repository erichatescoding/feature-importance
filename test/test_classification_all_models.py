"""
Test Feature Importance Evaluator for Classification with ALL models

This script tests classification tasks using all available models:
- ML models: tree, linear, boosted_tree (if available)
- Statistical model: mutual information

The script will generate and keep visualization plots for review.
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

def create_classification_dataset():
    """Create a comprehensive classification dataset with mixed feature types."""
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create diverse features
    data = {
        # Numeric features with varying importance
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.lognormal(10.5, 0.5, n_samples),
        'credit_score': np.random.normal(700, 100, n_samples),
        'account_balance': np.random.exponential(5000, n_samples),
        'days_since_last_purchase': np.random.randint(0, 365, n_samples),
        
        # Categorical features
        'customer_segment': np.random.choice(['Premium', 'Standard', 'Basic'], n_samples, p=[0.2, 0.5, 0.3]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports', 'Books'], n_samples),
        'payment_method': np.random.choice(['Credit', 'Debit', 'PayPal', 'Cash'], n_samples),
        'marketing_channel': np.random.choice(['Email', 'Social', 'Direct', 'Referral'], n_samples),
        
        # Binary features
        'is_member': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'has_discount': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'email_subscriber': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
        
        # Noise features
        'random_numeric': np.random.randn(n_samples),
        'random_categorical': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target with known relationships
    # High value customers (binary classification)
    probability = np.zeros(n_samples)
    
    # Strong predictors
    probability += 0.2 * (df['customer_segment'] == 'Premium')
    probability += 0.15 * (df['income'] > df['income'].median())
    probability += 0.1 * (df['credit_score'] > 750)
    probability += 0.1 * (df['is_member'] == 1)
    
    # Medium predictors
    probability += 0.05 * (df['account_balance'] > df['account_balance'].median())
    probability += 0.05 * (df['days_since_last_purchase'] < 30)
    probability += 0.05 * (df['payment_method'] == 'Credit')
    
    # Weak predictors
    probability += 0.02 * (df['has_discount'] == 1)
    probability += 0.02 * (df['email_subscriber'] == 1)
    probability += 0.01 * (df['region'] == 'North')
    
    # Add some randomness
    probability += np.random.uniform(-0.1, 0.1, n_samples)
    
    # Convert to binary target
    df['high_value_customer'] = (probability > np.random.uniform(0, 1, n_samples)).astype(int)
    
    return df

def main():
    """Test all models with classification data."""
    
    print("="*60)
    print("TESTING ALL MODELS WITH CLASSIFICATION DATA")
    print("="*60)
    
    # Clean output directory at the start
    output_dir = clean_output_directory()
    
    print("="*70)
    print("   TESTING FEATURE IMPORTANCE WITH ALL MODELS (CLASSIFICATION)   ")
    print("="*70)
    print("\nThis test evaluates feature importance for a classification task")
    print("using ALL available models (ML + Statistical)")
    print("="*70)
    
    # Create dataset
    print("\n1. Creating Classification Dataset")
    print("-"*50)
    df = create_classification_dataset()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.shape[1] - 1}")
    print(f"Samples: {df.shape[0]}")
    print(f"\nTarget distribution:")
    print(f"  High-value customers: {df['high_value_customer'].sum()} ({df['high_value_customer'].mean():.1%})")
    print(f"  Regular customers: {(1-df['high_value_customer']).sum()} ({(1-df['high_value_customer']).mean():.1%})")
    
    print("\nFeature types:")
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"  Numeric: {len(numeric_features) - 1}")  # -1 for target
    print(f"  Categorical: {len(categorical_features)}")
    
    # Add some missing values for realistic scenario
    df.loc[np.random.choice(df.index, 50, replace=False), 'income'] = np.nan
    df.loc[np.random.choice(df.index, 30, replace=False), 'credit_score'] = np.nan
    df.loc[np.random.choice(df.index, 20, replace=False), 'region'] = np.nan
    
    # Step 1: Preprocess data
    print("\n2. Preprocessing Data")
    print("-"*50)
    
    processed_df, preprocessing_info = preprocess_data(
        df=df,
        target='high_value_customer',
        problem_type='classification'
    )
    
    # Step 2: Evaluate with ALL models
    print("\n3. Evaluating Feature Importance with ALL Models")
    print("-"*50)
    
    # Use test output directory
    # Use the main project's cache directory (one level up)
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.feature_importance_cache')
    evaluator = FeatureImportanceEvaluator(cache_dir=cache_dir, output_dir=output_dir, clean_cache=False)
    
    # Use 'all' models with both importance methods
    results = evaluator.evaluate(
        df=processed_df,
        target='high_value_customer',
        problem_type='classification',
        models='all',  # This will use tree, linear, statistical, and boosted_tree (if available)
        importance_methods=['default', 'permutation'],  # Use both methods
        sample_size_for_permutation=500,  # Sample for faster permutation importance
        test_size=0.3  # 30% test set
    )
    
    # Display results summary
    print("\n4. RESULTS SUMMARY")
    print("="*70)
    
    # Models evaluated
    print("\nModels Evaluated:")
    for model_name in results['models'].keys():
        print(f"  ✓ {model_name}")
    
    # Model performance
    print("\nModel Performance Metrics:")
    print("-"*50)
    print(f"{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-"*50)
    
    for model_name, model_results in results['models'].items():
        if 'error' in model_results:
            print(f"{model_name:<15} ERROR: {model_results['error']}")
        elif 'metrics' in model_results:
            metrics = model_results['metrics']
            if 'accuracy' in metrics:
                print(f"{model_name:<15} "
                      f"{metrics.get('accuracy', 0):<10.4f} "
                      f"{metrics.get('precision', 0):<10.4f} "
                      f"{metrics.get('recall', 0):<10.4f} "
                      f"{metrics.get('f1_score', 0):<10.4f}")
            else:
                # Statistical model
                print(f"{model_name:<15} Mutual Information: {metrics.get('mutual_information', 0):.4f}")
    
    # Top features by consensus
    print("\n5. CONSENSUS FEATURE IMPORTANCE (Top 10)")
    print("="*70)
    print(f"{'Rank':<6} {'Feature':<30} {'Importance':<12} {'Cumulative':<12}")
    print("-"*70)
    
    cumulative = 0
    for i, row in results['consensus_importance'].head(10).iterrows():
        cumulative += row['importance']
        print(f"{row['rank']:<6} {row['feature']:<30} {row['importance']:<12.4f} {cumulative:<12.4f}")
    
    # Feature importance by model
    print("\n6. FEATURE IMPORTANCE BY MODEL (Top 5 per model)")
    print("="*70)
    
    for model_name, model_results in results['models'].items():
        if 'importance' in model_results and 'default' in model_results['importance']:
            print(f"\n{model_name.upper()} Model:")
            print("-"*40)
            importance_df = model_results['importance']['default'].head(5)
            for _, row in importance_df.iterrows():
                print(f"  {row['feature']:<30} {row['importance']:.4f}")
    
    # Visualizations generated
    print("\n7. VISUALIZATIONS GENERATED")
    print("="*70)
    print("The following visualization files have been created:")
    print()
    for viz_type, filename in results['visualizations'].items():
        print(f"  - {viz_type:<30} → {filename}")
    
    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nThe visualization plots have been generated and saved.")
    print("You can now review the PNG files to see:")
    print("  1. Feature importance heatmap comparing all models")
    print("  2. Consensus feature importance bar chart")
    print("  3. Individual model performance plots (confusion matrices, ROC curves, etc.)")
    print("\nThese visualizations provide insights into:")
    print("  - How different models rank features differently")
    print("  - Which features are consistently important across models")
    print("  - Model performance comparisons")
    print("  - Classification diagnostics for each model")

if __name__ == "__main__":
    main() 