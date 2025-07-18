"""
Random Forest Regressor with Feature Importance Analysis

This script provides a comprehensive Random Forest regression analysis including:
- Model training and evaluation
- Feature importance ranking
- Visualization dashboard with multiple charts
- Performance metrics summary

USAGE WITH YOUR OWN DATA:
========================
1. Import the function:
   from random_forest_regressor import run_random_forest_regressor

2. Use with your dataframe:
   results = run_random_forest_regressor(your_df, 'target_column_name')

3. Access results:
   - results['model'] - trained Random Forest model
   - results['feature_importance'] - feature importance DataFrame
   - results['mae'] - Mean Absolute Error
   - results['mse'] - Mean Squared Error
   - results['rmse'] - Root Mean Squared Error
   - results['r2_score'] - R² Score

EXAMPLE:
========
import pandas as pd
from random_forest_regressor import run_random_forest_regressor

# Load your data
df = pd.read_csv('your_data.csv')

# Run analysis (assuming 'price' is your target column)
results = run_random_forest_regressor(df, 'price')

# Get the trained model
model = results['model']
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# RANDOM FOREST REGRESSOR FUNCTION
# ============================================================================

def run_random_forest_regressor(df, target_column, test_size=0.2, random_state=42):
    """
    Run Random Forest Regression analysis on a given dataframe.
    
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
    print(f"Features shape: {X.shape}")
    print(f"Target ({target_column}) statistics:")
    print(f"  Mean: {y.mean():.2f}")
    print(f"  Median: {y.median():.2f}")
    print(f"  Min: {y.min():.2f}")
    print(f"  Max: {y.max():.2f}")
    print()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Create and train Random Forest model
    print("Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Calculate regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("="*50)
    print("REGRESSION EVALUATION METRICS")
    print("="*50)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print()

    # ============================================================================
    # FEATURE IMPORTANCE ANALYSIS
    # ============================================================================

    # Get feature importance from the trained model
    feature_importance = rf_model.feature_importances_
    feature_names = X.columns.tolist()

    # Create a DataFrame for easier handling
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    # Add ranking
    importance_df['rank'] = range(1, len(importance_df) + 1)

    print("="*50)
    print("FEATURE IMPORTANCE RANKING")
    print("="*50)
    print(importance_df[['rank', 'feature', 'importance']].to_string(index=False))
    print()

    # ============================================================================
    # VISUALIZATIONS
    # ============================================================================

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create a comprehensive visualization dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Random Forest Regression - {target_column.title()} Prediction Analysis', fontsize=16, fontweight='bold')

    # 1. Feature Importance Horizontal Bar Chart
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    y_pos = np.arange(len(importance_df))
    bars = ax1.barh(y_pos, importance_df['importance'], color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(importance_df['feature'])
    ax1.set_xlabel('Importance Score')
    ax1.set_title('Feature Importance Ranking', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()  # Highest importance at top

    # Add importance scores as text
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        ax1.text(row['importance'] + 0.002, i, f'{row["importance"]:.3f}', 
                 va='center', fontsize=9)

    # 2. Actual vs Predicted Scatter Plot
    ax2 = axes[0, 1]
    ax2.scatter(y_test, y_pred, alpha=0.6, color='blue', s=20)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel(f'Actual {target_column.title()}')
    ax2.set_ylabel(f'Predicted {target_column.title()}')
    ax2.set_title(f'Actual vs Predicted {target_column.title()}')
    ax2.grid(alpha=0.3)

    # Add R² score to the plot
    ax2.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
             verticalalignment='top', fontsize=10)

    # 3. Residuals Plot
    ax3 = axes[0, 2]
    residuals = y_test - y_pred
    ax3.scatter(y_pred, residuals, alpha=0.6, color='green', s=20)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_xlabel(f'Predicted {target_column.title()}')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residuals Plot')
    ax3.grid(alpha=0.3)

    # 4. Hide the empty subplot where Top 10 chart was removed
    ax4 = axes[1, 0]
    ax4.axis('off')

    # 5. Error Distribution Histogram
    ax5 = axes[1, 1]
    ax5.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax5.set_xlabel('Residuals')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distribution of Residuals')
    ax5.grid(alpha=0.3)

    # Add statistics to the plot
    ax5.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {residuals.mean():.3f}')
    ax5.axvline(residuals.std(), color='orange', linestyle='--', linewidth=2, 
                label=f'Std: {residuals.std():.3f}')
    ax5.axvline(-residuals.std(), color='orange', linestyle='--', linewidth=2)
    ax5.legend()

    # 6. Metrics Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    metrics_text = f"""
REGRESSION METRICS SUMMARY

Mean Absolute Error (MAE): {mae:.4f}
Mean Squared Error (MSE): {mse:.4f}
Root Mean Squared Error (RMSE): {rmse:.4f}
R² Score: {r2:.4f}

MODEL INTERPRETATION:
• MAE: Average absolute prediction error
• MSE: Average squared prediction error
• RMSE: Standard deviation of residuals
• R²: Proportion of variance explained
  (1.0 = perfect, 0.0 = no better than mean)

MODEL QUALITY ASSESSMENT:
• R² = {r2:.3f} indicates the model explains 
  {r2*100:.1f}% of the variance in {target_column}
• Average prediction error: ±{mae:.4f}
• Residual standard deviation: {rmse:.4f}

TOP FEATURES:
• {importance_df.iloc[0]['feature']}: {importance_df.iloc[0]['importance']:.3f}
• {importance_df.iloc[1]['feature']}: {importance_df.iloc[1]['importance']:.3f}
• {importance_df.iloc[2]['feature']}: {importance_df.iloc[2]['importance']:.3f}
"""

    ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    
    # Ensure output directory exists
    import os
    os.makedirs('output', exist_ok=True)
    
    # Save to output directory
    output_path = os.path.join('output', 'random_forest_regression_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as '{output_path}'")
    plt.close()

    # ============================================================================
    # DETAILED ANALYSIS
    # ============================================================================

    print("="*50)
    print("DETAILED ANALYSIS")
    print("="*50)

    # Model performance summary
    print(f"Model Performance:")
    print(f"  - The model explains {r2*100:.1f}% of the variance in {target_column}")
    print(f"  - Average prediction error: {mae:.4f}")
    print(f"  - Standard deviation of errors: {rmse:.4f}")
    print()

    # Feature importance insights
    print("Feature Importance Insights:")
    print(f"  - Most important feature: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance']:.3f})")
    print(f"  - Top 3 features contribute {importance_df.head(3)['importance'].sum():.1%} of total importance")
    print(f"  - Number of features with >5% importance: {len(importance_df[importance_df['importance'] > 0.05])}")
    print()

    # Model quality assessment
    if r2 > 0.8:
        quality = "Excellent"
    elif r2 > 0.6:
        quality = "Good"
    elif r2 > 0.4:
        quality = "Moderate"
    else:
        quality = "Poor"

    print(f"Overall Model Quality: {quality} (R² = {r2:.3f})")
    print()

    # Sample predictions
    print("Sample Predictions:")
    print("-" * 40)
    for i in range(min(5, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        error = abs(actual - predicted)
        print(f"Sample {i+1}: Actual={actual:.2f}, Predicted={predicted:.2f}, Error={error:.2f}")
    print()

    print("="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)

    # Return results dictionary
    return {
        'model': rf_model,
        'feature_importance': importance_df,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2_score': r2,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'residuals': residuals
    }

# ============================================================================
# EXAMPLE USAGE - Remove this section when using with your own data
# ============================================================================

# For demonstration purposes - remove this when using your own dataframe
def create_sample_data():
    """Create sample data for demonstration. Remove this function when using real data."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic house features
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
        'has_pool': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'has_fireplace': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
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
        df['has_pool'] * 25000 +
        df['has_fireplace'] * 8000 +
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
    
    return df

# REMOVE THIS SECTION WHEN USING YOUR OWN DATA
# ============================================
# Using sample data for demonstration
if __name__ == "__main__":
    df = create_sample_data()
    results = run_random_forest_regressor(df, 'price')
    
    # Access results if needed
    # model = results['model']
    # feature_importance = results['feature_importance']
    # r2_score = results['r2_score']
