"""
Random Forest Classifier with Feature Importance Analysis

This script provides a comprehensive Random Forest classification analysis including:
- Model training and evaluation
- Feature importance ranking
- Confusion matrix visualization
- Performance metrics summary

USAGE WITH YOUR OWN DATA:
========================
1. Import the function:
   from random_forest_classifier import run_random_forest_classifier

2. Use with your dataframe:
   results = run_random_forest_classifier(your_df, 'target_column_name')

3. Access results:
   - results['model'] - trained Random Forest model
   - results['feature_importance'] - feature importance DataFrame
   - results['accuracy'] - model accuracy
   - results['precision'] - precision score
   - results['recall'] - recall score
   - results['f1_score'] - F1 score
   - results['confusion_matrix'] - confusion matrix

EXAMPLE:
========
import pandas as pd
from random_forest_classifier import run_random_forest_classifier

# Load your data
df = pd.read_csv('your_data.csv')

# Run analysis (assuming 'target' is your target column)
results = run_random_forest_classifier(df, 'target')

# Get the trained model
model = results['model']
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# RANDOM FOREST CLASSIFIER FUNCTION
# ============================================================================

def run_random_forest_classifier(df, target_column, test_size=0.2, random_state=42):
    """
    Run Random Forest Classification analysis on a given dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing features and target
    target_column : str
        Name of the target column in the dataframe
    test_size : float, default=0.2
        Proportion of dataset to include in test split
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing model results and metrics
    """
    
    # Prepare features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    print(f"Dataset loaded successfully!")
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Class balance: {y.mean():.2%} positive class")
    print()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Create and train Random Forest model
    rf_model = RandomForestClassifier(
        n_estimators=100,        # Number of trees
        max_depth=10,           # Prevent overfitting
        min_samples_split=5,    # Prevent overfitting
        min_samples_leaf=2,     # Prevent overfitting
        random_state=random_state,
        n_jobs=-1              # Use all CPU cores
    )

    # Train the model
    rf_model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Calculate and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 0    1")
    print(f"Actual    0    {cm[0][0]:3d}  {cm[0][1]:3d}")
    print(f"          1    {cm[1][0]:3d}  {cm[1][1]:3d}")

    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nDetailed Metrics:")
    print(f"True Positives:  {tp}")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision:       {precision:.4f}")
    print(f"Recall:          {recall:.4f}")
    print(f"Specificity:     {specificity:.4f}")
    print(f"F1-Score:        {f1_score:.4f}")

    # Get feature importance scores
    feature_importance = rf_model.feature_importances_

    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

    # Add ranking
    importance_df['Rank'] = range(1, len(importance_df) + 1)

    print("\nFeature Importance Ranking:")
    print(importance_df)

    # Create visualizations - Three charts on the same row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Feature Importance Horizontal Bar Chart
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    y_pos = np.arange(len(importance_df))
    bars = ax1.barh(y_pos, importance_df['Importance'], color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(importance_df['Feature'])
    ax1.set_xlabel('Importance Score')
    ax1.set_title('Feature Importance Ranking', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()  # Highest importance at top

    # Add importance scores as text
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        ax1.text(row['Importance'] + 0.002, i, f'{row["Importance"]:.3f}', 
                 va='center', fontsize=9)

    # 2. Confusion Matrix Heatmap
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'],
                ax=ax2,
                cbar_kws={'shrink': 0.8})
    ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    # 3. Classification Report Summary
    ax3.axis('off')

    # Generate classification summary
    summary_text = f"""CLASSIFICATION REPORT SUMMARY

Model Performance:
• Accuracy: {accuracy:.4f}
• Precision: {precision:.4f}  
• Recall: {recall:.4f}
• F1-Score: {f1_score:.4f}
• Specificity: {specificity:.4f}

Feature Importance:
• Total Features: {len(importance_df)}
• Most Important: {importance_df.iloc[0]['Feature']} ({importance_df.iloc[0]['Importance']:.3f})
• Least Important: {importance_df.iloc[-1]['Feature']} ({importance_df.iloc[-1]['Importance']:.3f})

Classification Matrix:
• True Positives: {tp}
• True Negatives: {tn}
• False Positives: {fp}
• False Negatives: {fn}"""

    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    ax3.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('random_forest_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'random_forest_analysis.png'")
    plt.close()

    # Print classification report for additional details
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

    # Return results dictionary
    return {
        'model': rf_model,
        'feature_importance': importance_df,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'confusion_matrix': cm,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred
    }

# ============================================================================
# EXAMPLE USAGE - Remove this section when using with your own data
# ============================================================================

# For demonstration purposes - remove this when using your own dataframe
def create_sample_data():
    """Create sample data for demonstration. Remove this function when using real data."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    # Create synthetic features with different importance levels
    X = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # Create binary target with some features being more important than others
    y = (2.0 * X[:, 0] +           # Highly important
         1.5 * X[:, 2] +           # Highly important  
         1.2 * X[:, 5] +           # Highly important
         0.8 * X[:, 1] +           # Moderately important
         0.6 * X[:, 3] +           # Moderately important
         0.4 * X[:, 7] +           # Moderately important
         0.1 * np.sum(X[:, 8:], axis=1) +  # Noise features
         np.random.randn(n_samples) * 0.5) > 0  # Add noise and convert to binary
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y.astype(int)
    
    return df

# REMOVE THIS SECTION WHEN USING YOUR OWN DATA
# ============================================
# Using sample data for demonstration
if __name__ == "__main__":
    df = create_sample_data()
    results = run_random_forest_classifier(df, 'target')
    
    # Access results if needed
    # model = results['model']
    # feature_importance = results['feature_importance']
    # accuracy = results['accuracy']
