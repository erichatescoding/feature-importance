import pandas as pd
import numpy as np
import s3fs
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from random_forest_regressor import run_random_forest_regressor
import warnings
warnings.filterwarnings('ignore')

def load_s3_data():
    """Load data from S3 using the specified bucket and path."""
    print("Loading data from S3...")
    
    bucket_prefix = "vungle2-dataeng/dev/floor_training_test/"
    fs = s3fs.S3FileSystem()
    
    try:
        parquet_files = fs.glob(f"{bucket_prefix}*.parquet")
        print(f"Found {len(parquet_files)} parquet files")
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {bucket_prefix}")
        
        # Load all parquet files
        dfs = []
        for i, file in enumerate(parquet_files):
            print(f"Loading file {i+1}/{len(parquet_files)}: {file}")
            df_temp = pd.read_parquet(f"s3://{file}", engine='pyarrow')
            dfs.append(df_temp)
        
        # Concatenate all dataframes
        df = pd.concat(dfs, ignore_index=True)
        print(f"Combined dataset shape: {df.shape}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data from S3: {e}")
        raise

def preprocess_data(df):
    """Preprocess the dataset to handle string-to-float conversion and prepare for Random Forest."""
    print("\nPreprocessing data...")
    
    # Display basic info about the dataset
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing values:")
        print(missing_values[missing_values > 0])
    
    # Remove event_count column if it exists
    if 'event_count' in df.columns:
        df = df.drop('event_count', axis=1)
        print("Removed 'event_count' column")
    
    # Check if net_revenue exists
    if 'net_revenue' not in df.columns:
        raise ValueError("Target column 'net_revenue' not found in dataset")
    
    # Handle missing values in target
    if df['net_revenue'].isnull().sum() > 0:
        print(f"Removing {df['net_revenue'].isnull().sum()} rows with missing net_revenue")
        df = df.dropna(subset=['net_revenue'])
    
    # Identify categorical and numerical columns
    categorical_columns = []
    numerical_columns = []
    
    for col in df.columns:
        if col == 'net_revenue':
            continue
            
        if df[col].dtype == 'object' or df[col].dtype == 'string':
            categorical_columns.append(col)
        else:
            # Check if it's actually numerical
            try:
                pd.to_numeric(df[col], errors='raise')
                numerical_columns.append(col)
            except:
                categorical_columns.append(col)
    
    print(f"\nCategorical columns ({len(categorical_columns)}): {categorical_columns}")
    print(f"Numerical columns ({len(numerical_columns)}): {numerical_columns}")
    
    # Handle categorical columns (this fixes the string-to-float conversion error)
    label_encoders = {}
    
    for col in categorical_columns:
        print(f"Encoding categorical column: {col}")
        # Handle missing values in categorical columns
        df[col] = df[col].fillna('missing')
        
        # Convert to string to handle any remaining non-string values
        df[col] = df[col].astype(str)
        
        # Check for very high cardinality (might be IDs)
        unique_count = df[col].nunique()
        if unique_count > 1000:
            print(f"  WARNING: High cardinality column {col} has {unique_count} unique values")
        
        # Apply label encoding
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
        print(f"  - Encoded {col}: {len(le.classes_)} unique values")
    
    # Handle numerical columns
    for col in numerical_columns:
        # Convert to numeric, replacing errors with NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values with median
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled {col} missing values with median: {median_val}")
    
    # Ensure target is numeric
    df['net_revenue'] = pd.to_numeric(df['net_revenue'], errors='coerce')
    
    # Remove rows where target couldn't be converted
    original_shape = df.shape[0]
    df = df.dropna(subset=['net_revenue'])
    if df.shape[0] < original_shape:
        print(f"Removed {original_shape - df.shape[0]} rows with invalid net_revenue values")
    
    print(f"\nFinal preprocessed dataset shape: {df.shape}")
    print(f"Target (net_revenue) statistics:")
    print(f"  Mean: {df['net_revenue'].mean():.4f}")
    print(f"  Median: {df['net_revenue'].median():.4f}")
    print(f"  Min: {df['net_revenue'].min():.4f}")
    print(f"  Max: {df['net_revenue'].max():.4f}")
    print(f"  Std: {df['net_revenue'].std():.4f}")
    
    return df, label_encoders

def analyze_data_quality(df):
    """Analyze data quality before preprocessing."""
    print("\n" + "="*60)
    print("DATA QUALITY ANALYSIS")
    print("="*60)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print()
    
    print("Column Analysis:")
    print("-" * 40)
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = (null_count / len(df)) * 100
        unique_count = df[col].nunique()
        unique_pct = (unique_count / len(df)) * 100
        
        print(f"{col}:")
        print(f"  Type: {dtype}")
        print(f"  Null: {null_count:,} ({null_pct:.1f}%)")
        print(f"  Unique: {unique_count:,} ({unique_pct:.1f}%)")
        
        if dtype in ['object', 'string']:
            try:
                sample_values = df[col].dropna().astype(str).head(3).tolist()
                print(f"  Sample: {sample_values}")
            except:
                print(f"  Sample: [Could not display samples]")
        elif dtype in ['int64', 'float64']:
            try:
                print(f"  Range: {df[col].min()} to {df[col].max()}")
            except:
                print(f"  Range: [Could not calculate range]")
        print()
    
    print("="*60)

# Main execution
if __name__ == "__main__":
    print("🔍 Running Random Forest Regressor with S3 Data")
    print("=" * 60)
    
    try:
        # Load data from S3
        print("\n📊 STEP 1: Loading S3 Data")
        print("-" * 40)
        df_raw = load_s3_data()
        
        # Analyze data quality
        print("\n🔍 STEP 2: Data Quality Analysis")
        print("-" * 40)
        analyze_data_quality(df_raw)
        
        # Preprocess data
        print("\n⚙️  STEP 3: Data Preprocessing")
        print("-" * 40)
        df_processed, label_encoders = preprocess_data(df_raw)
        
        # Run Random Forest analysis
        print("\n🤖 STEP 4: Random Forest Training & Analysis")
        print("-" * 40)
        print("Running Random Forest Regression Analysis...")
        print("=" * 50)
        
        # Run the Random Forest regressor
        results = run_random_forest_regressor(df_processed, 'net_revenue')
        
        print("\n" + "=" * 50)
        print("S3 DATA ANALYSIS COMPLETE!")
        print("=" * 50)
        
        # Display key results
        print(f"\nKey Results Summary:")
        print(f"• Dataset: {len(df_processed):,} samples, {len(df_processed.columns)-1} features")
        print(f"• R² Score: {results['r2_score']:.4f}")
        print(f"• Mean Absolute Error: {results['mae']:.4f}")
        print(f"• Root Mean Squared Error: {results['rmse']:.4f}")
        
        # Model quality assessment
        if results['r2_score'] > 0.8:
            quality = "Excellent"
        elif results['r2_score'] > 0.6:
            quality = "Good"
        elif results['r2_score'] > 0.4:
            quality = "Moderate"
        else:
            quality = "Poor"
        
        print(f"• Model Quality: {quality}")
        
        print(f"\nTop 10 Most Important Features:")
        top_features = results['feature_importance'].head(10)
        for idx, row in top_features.iterrows():
            print(f"  {row['rank']}. {row['feature']}: {row['importance']:.4f}")
        
        # Show label encoder information
        print(f"\nCategorical Features Encoded:")
        for col, encoder in label_encoders.items():
            print(f"  • {col}: {len(encoder.classes_)} unique values")
        
        # Show feature importance distribution
        print(f"\nFeature Importance Distribution:")
        importance_df = results['feature_importance']
        print(f"  • Top 5 features contribute: {importance_df.head(5)['importance'].sum():.1%}")
        print(f"  • Top 10 features contribute: {importance_df.head(10)['importance'].sum():.1%}")
        print(f"  • Features with >1% importance: {len(importance_df[importance_df['importance'] > 0.01])}")
        
        # Sample predictions
        print(f"\nSample Predictions:")
        print("-" * 50)
        sample_indices = np.random.choice(len(results['y_test']), min(5, len(results['y_test'])), replace=False)
        for i, idx in enumerate(sample_indices):
            actual = results['y_test'].iloc[idx]
            predicted = results['y_pred'][idx]
            error = abs(actual - predicted)
            error_pct = (error / actual) * 100 if actual != 0 else 0
            print(f"Sample {i+1}: Actual={actual:.4f}, Predicted={predicted:.4f}, Error={error:.4f} ({error_pct:.1f}%)")
        
        print(f"\n✅ Random Forest analysis completed successfully!")
        print("📈 Generating comprehensive visualization...")
        # plot_random_forest_analysis(X_test, y_test, y_pred, model, feature_names) # This line was not in the original file, so it's not added.
        print(f"📊 Visualization saved as 'output/random_forest_regression_analysis.png'")
        print(f"🎯 The model explains {results['r2_score']*100:.1f}% of the variance in net_revenue")
        
        # Model interpretation
        print(f"\n🔍 MODEL INTERPRETATION:")
        print(f"• The most important feature is '{importance_df.iloc[0]['feature']}' with {importance_df.iloc[0]['importance']:.1%} importance")
        print(f"• Average prediction error: ±{results['mae']:.4f}")
        print(f"• Model performance: {quality} (R² = {results['r2_score']:.3f})")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Please check your S3 connection and data format.")
        import traceback
        traceback.print_exc()
