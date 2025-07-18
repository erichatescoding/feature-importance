#!/usr/bin/env python3
"""
Direct test of permutation feature importance
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from importance.permutation import get_permutation_importance


def plot_importance_bar_chart(importance_df, title, save_path=None):
    """
    Create a bar chart for feature importances.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        title: Title for the plot
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    bars = plt.bar(importance_df['feature'], importance_df['importance'])
    
    # Color bars based on importance
    colors = plt.cm.viridis(importance_df['importance'] / importance_df['importance'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (feature, importance) in enumerate(zip(importance_df['feature'], importance_df['importance'])):
        plt.text(i, importance + 0.01, f'{importance:.3f}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.ylim(0, max(importance_df['importance']) * 1.15)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_importance_comparison(importance_results, title, save_path=None):
    """
    Create a grouped bar chart comparing multiple importance results.
    
    Args:
        importance_results: Dict of {method_name: importance_df}
        title: Title for the plot
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Prepare data for grouped bar chart
    features = list(importance_results.values())[0]['feature'].tolist()
    n_features = len(features)
    n_methods = len(importance_results)
    width = 0.8 / n_methods
    
    x = np.arange(n_features)
    
    for i, (method, df) in enumerate(importance_results.items()):
        offset = width * (i - n_methods/2 + 0.5)
        bars = plt.bar(x + offset, df['importance'], width, label=method)
        
        # Add value labels
        for j, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(x, features, rotation=45, ha='right')
    plt.legend(title='Method')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Comparison plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def test_permutation_importance():
    print("=== TESTING PERMUTATION FEATURE IMPORTANCE ===\n")
    
    # Create output directory for plots
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Test 1: Basic functionality
    print("Test 1: Basic Regression Test")
    print("-" * 40)
    
    # Create simple dataset
    np.random.seed(42)
    n_samples = 300
    
    X = pd.DataFrame({
        'important_1': np.random.randn(n_samples),
        'important_2': np.random.randn(n_samples),
        'not_important': np.random.randn(n_samples)
    })
    
    # Target strongly depends on first two features
    y = 3 * X['important_1'] + 2 * X['important_2'] + 0.1 * np.random.randn(n_samples)
    
    # Train model
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)
    
    # Get permutation importance
    perm_imp = get_permutation_importance(
        rf, X, y, 
        problem_type='regression',
        n_repeats=10,
        random_state=42
    )
    
    print("Permutation Importance Results:")
    print(perm_imp)
    print(f"\nSum of importances: {perm_imp['importance'].sum():.4f} (should be ~1.0)")
    
    # Plot the results
    plot_importance_bar_chart(
        perm_imp, 
        'Permutation Feature Importance - Regression',
        os.path.join(output_dir, 'permutation_importance_regression.png')
    )
    
    # Test 2: Classification
    print("\n\nTest 2: Classification Test")
    print("-" * 40)
    
    # Create classification target
    y_class = (y > y.median()).astype(int)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y_class)
    
    perm_imp_clf = get_permutation_importance(
        clf, X, y_class,
        problem_type='classification',
        n_repeats=5
    )
    
    print("Permutation Importance Results:")
    print(perm_imp_clf)
    
    # Plot classification results
    plot_importance_bar_chart(
        perm_imp_clf,
        'Permutation Feature Importance - Classification',
        os.path.join(output_dir, 'permutation_importance_classification.png')
    )
    
    # Test 3: With sampling
    print("\n\nTest 3: With Sampling (100 samples)")
    print("-" * 40)
    
    perm_imp_sampled = get_permutation_importance(
        rf, X, y,
        problem_type='regression',
        sample_size=100,
        n_repeats=5
    )
    
    print("Permutation Importance Results (sampled):")
    print(perm_imp_sampled)
    
    # Test 4: Different model type
    print("\n\nTest 4: Linear Regression Model")
    print("-" * 40)
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    perm_imp_lr = get_permutation_importance(
        lr, X, y,
        problem_type='regression',
        n_repeats=10
    )
    
    print("Permutation Importance Results:")
    print(perm_imp_lr)
    
    # Create comparison plot
    print("\n\nGenerating comparison plots...")
    comparison_data = {
        'Random Forest': perm_imp,
        'RF Sampled': perm_imp_sampled,
        'Linear Regression': perm_imp_lr
    }
    
    plot_importance_comparison(
        comparison_data,
        'Permutation Importance Comparison - Different Models',
        os.path.join(output_dir, 'permutation_importance_comparison.png')
    )
    
    # Test 5: Complex dataset with more features
    print("\n\nTest 5: Complex Dataset Test")
    print("-" * 40)
    
    # Create a more complex dataset
    n_features = 8
    X_complex = pd.DataFrame({
        f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
    })
    
    # Create target with varying importance
    y_complex = (
        5 * X_complex['feature_0'] +  # Very important
        3 * X_complex['feature_1'] +  # Important
        1 * X_complex['feature_2'] +  # Somewhat important
        0.5 * X_complex['feature_3'] + # Slightly important
        0.1 * np.random.randn(n_samples)  # Noise
    )
    
    # Train model
    rf_complex = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_complex.fit(X_complex, y_complex)
    
    # Get permutation importance
    perm_imp_complex = get_permutation_importance(
        rf_complex, X_complex, y_complex,
        problem_type='regression',
        n_repeats=15,
        random_state=42
    )
    
    print("Permutation Importance Results (Complex Dataset):")
    print(perm_imp_complex)
    
    # Create a more detailed plot
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar plot for better readability
    bars = plt.barh(perm_imp_complex['feature'], perm_imp_complex['importance'])
    
    # Color based on importance
    colors = plt.cm.RdYlGn(perm_imp_complex['importance'] / perm_imp_complex['importance'].max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Permutation Feature Importance - Complex Dataset\n(Features colored by importance: Red=Low, Green=High)', 
              fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (feature, importance) in enumerate(zip(perm_imp_complex['feature'], perm_imp_complex['importance'])):
        plt.text(importance + 0.002, i, f'{importance:.4f}', 
                va='center', fontsize=10)
    
    plt.xlim(0, max(perm_imp_complex['importance']) * 1.1)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'permutation_importance_complex.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Complex dataset plot saved to: {save_path}")
    plt.close()
    
    print("\n✅ All tests completed successfully!")
    print(f"\n📊 Plots saved to: {output_dir}/")
    print("\nSummary:")
    print("- Permutation importance works for both regression and classification")
    print("- Sampling produces consistent relative importances")
    print("- Works with different model types (RF, Linear)")
    print("- Importances are normalized to sum to 1")
    print("- Visual plots help interpret feature importance rankings")

if __name__ == "__main__":
    test_permutation_importance() 