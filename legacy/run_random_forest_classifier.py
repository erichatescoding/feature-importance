import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from random_forest_classifier import run_random_forest_classifier

# Create a sample dataset for demonstration
def create_sample_dataset():
    """Create a sample dataset for Random Forest classification."""
    np.random.seed(42)
    n_samples = 800
    
    # Create features with realistic names
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education_years': np.random.randint(8, 20, n_samples),
        'experience_years': np.random.randint(0, 40, n_samples),
        'hours_per_week': np.random.randint(20, 60, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'debt_to_income': np.random.uniform(0.1, 0.8, n_samples),
        'savings': np.random.exponential(10000, n_samples),
        'num_dependents': np.random.randint(0, 5, n_samples),
        'owns_home': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on logical relationships
    # Higher income, credit score, savings, and home ownership increase approval probability
    target_probability = (
        0.3 * (df['income'] - df['income'].min()) / (df['income'].max() - df['income'].min()) +
        0.2 * (df['credit_score'] - df['credit_score'].min()) / (df['credit_score'].max() - df['credit_score'].min()) +
        0.2 * (df['savings'] - df['savings'].min()) / (df['savings'].max() - df['savings'].min()) +
        0.1 * df['owns_home'] +
        0.1 * (df['education_years'] - df['education_years'].min()) / (df['education_years'].max() - df['education_years'].min()) +
        0.1 * (1 - df['debt_to_income']) +
        np.random.normal(0, 0.1, n_samples)  # Add some noise
    )
    
    # Convert to binary target (loan approval: 1=approved, 0=rejected)
    df['loan_approved'] = (target_probability > 0.5).astype(int)
    
    return df

# Main execution
if __name__ == "__main__":
    print("Creating sample dataset...")
    df = create_sample_dataset()
    
    print(f"Dataset created with {len(df)} samples and {len(df.columns)-1} features")
    print(f"Target column: 'loan_approved'")
    print(f"Dataset preview:")
    print(df.head())
    print()
    
    print("Running Random Forest Classification Analysis...")
    print("=" * 60)
    
    # Run the Random Forest classifier
    results = run_random_forest_classifier(df, 'loan_approved')
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    
    # Display key results
    print(f"\nKey Results Summary:")
    print(f"• Model Accuracy: {results['accuracy']:.4f}")
    print(f"• Precision: {results['precision']:.4f}")
    print(f"• Recall: {results['recall']:.4f}")
    print(f"• F1-Score: {results['f1_score']:.4f}")
    
    print(f"\nTop 5 Most Important Features:")
    top_features = results['feature_importance'].head()
    for idx, row in top_features.iterrows():
        print(f"  {row['Rank']}. {row['Feature']}: {row['Importance']:.4f}")
    
    print(f"\nYou can access the trained model with: results['model']")
    print(f"Full feature importance ranking: results['feature_importance']")
