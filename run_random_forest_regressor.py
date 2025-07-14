import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from random_forest_regressor import run_random_forest_regressor

# Create a sample dataset for regression testing - Employee Salary Prediction
def create_salary_dataset():
    """Create a sample employee salary dataset for regression testing."""
    np.random.seed(42)
    n_samples = 1200
    
    # Create employee features
    data = {
        'years_experience': np.random.randint(0, 35, n_samples),
        'education_level': np.random.randint(1, 6, n_samples),  # 1=High School, 5=PhD
        'age': np.random.randint(22, 65, n_samples),
        'hours_per_week': np.random.randint(30, 70, n_samples),
        'certifications': np.random.randint(0, 8, n_samples),
        'team_size_managed': np.random.randint(0, 50, n_samples),
        'performance_rating': np.random.uniform(1.0, 5.0, n_samples),
        'company_size': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.2, 0.3, 0.3, 0.15, 0.05]),  # 1=Startup, 5=Enterprise
        'remote_work_days': np.random.randint(0, 5, n_samples),
        'industry_type': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.25, 0.20, 0.20, 0.20, 0.15]),  # Tech, Finance, Healthcare, etc.
        'location_tier': np.random.choice([1, 2, 3], n_samples, p=[0.4, 0.35, 0.25]),  # 1=Major City, 3=Small City
        'has_mba': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'speaks_multiple_languages': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'stock_options': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic salary based on features
    base_salary = 35000
    
    salary = (
        base_salary +
        df['years_experience'] * 2800 +         # Experience: $2.8k per year
        df['education_level'] * 12000 +         # Education: $12k per level
        df['age'] * 400 +                       # Age premium: $400 per year
        df['hours_per_week'] * 350 +            # Hours: $350 per hour/week
        df['certifications'] * 3500 +           # Certifications: $3.5k each
        df['team_size_managed'] * 800 +         # Management: $800 per person managed
        df['performance_rating'] * 8000 +       # Performance: $8k per rating point
        df['company_size'] * 15000 +            # Company size: $15k per tier
        df['remote_work_days'] * 2000 +         # Remote work: $2k per day
        df['industry_type'] * 8000 +            # Industry: $8k per tier
        df['location_tier'] * 12000 +           # Location: $12k per tier
        df['has_mba'] * 25000 +                 # MBA bonus: $25k
        df['speaks_multiple_languages'] * 5000 + # Language bonus: $5k
        df['stock_options'] * 15000 +           # Stock options: $15k
        np.random.normal(0, 8000, n_samples)    # Random noise
    )
    
    # Ensure no negative salaries
    salary = np.maximum(salary, 30000)
    df['annual_salary'] = salary
    
    return df

# Create another dataset - Product Sales Prediction
def create_product_sales_dataset():
    """Create a sample product sales dataset for regression testing."""
    np.random.seed(123)
    n_samples = 800
    
    # Create product features
    data = {
        'price': np.random.uniform(10, 500, n_samples),
        'advertising_spend': np.random.uniform(1000, 50000, n_samples),
        'competitor_price': np.random.uniform(8, 600, n_samples),
        'season': np.random.choice([1, 2, 3, 4], n_samples),  # 1=Spring, 4=Winter
        'product_rating': np.random.uniform(1.0, 5.0, n_samples),
        'num_reviews': np.random.randint(0, 1000, n_samples),
        'inventory_level': np.random.randint(0, 500, n_samples),
        'discount_percentage': np.random.uniform(0, 50, n_samples),
        'website_visits': np.random.randint(100, 10000, n_samples),
        'social_media_mentions': np.random.randint(0, 500, n_samples),
        'is_featured': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'brand_strength': np.random.uniform(1, 10, n_samples),
        'market_share': np.random.uniform(0.1, 15.0, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic sales based on features
    base_sales = 50
    
    sales = (
        base_sales +
        -df['price'] * 0.8 +                    # Higher price = lower sales
        df['advertising_spend'] * 0.02 +        # Advertising impact
        -df['competitor_price'] * 0.3 +         # Lower competitor price = lower sales
        df['season'] * 30 +                     # Seasonal effect
        df['product_rating'] * 150 +            # Rating impact
        df['num_reviews'] * 0.5 +               # Reviews impact
        df['inventory_level'] * 0.2 +           # Inventory availability
        df['discount_percentage'] * 8 +         # Discount impact
        df['website_visits'] * 0.03 +           # Website traffic
        df['social_media_mentions'] * 2 +       # Social media impact
        df['is_featured'] * 200 +               # Featured product bonus
        df['brand_strength'] * 50 +             # Brand strength
        df['market_share'] * 20 +               # Market share impact
        np.random.normal(0, 100, n_samples)     # Random noise
    )
    
    # Ensure no negative sales
    sales = np.maximum(sales, 10)
    df['monthly_sales'] = sales
    
    return df

# Main execution
if __name__ == "__main__":
    print("🔍 Testing Random Forest Regressor with Different Datasets")
    print("=" * 60)
    
    # Test 1: Employee Salary Prediction
    print("\n📊 TEST 1: Employee Salary Prediction")
    print("-" * 40)
    
    df_salary = create_salary_dataset()
    print(f"Dataset created with {len(df_salary)} employees")
    print(f"Target: 'annual_salary'")
    print(f"Features: {len(df_salary.columns)-1}")
    print(f"Dataset preview:")
    print(df_salary.head())
    print()
    
    print("Running Random Forest Regression Analysis...")
    print("=" * 50)
    
    # Run the Random Forest regressor
    results_salary = run_random_forest_regressor(df_salary, 'annual_salary')
    
    print("\n" + "=" * 50)
    print("SALARY PREDICTION ANALYSIS COMPLETE!")
    print("=" * 50)
    
    # Display key results
    print(f"\nKey Results Summary:")
    print(f"• R² Score: {results_salary['r2_score']:.4f}")
    print(f"• Mean Absolute Error: ${results_salary['mae']:,.0f}")
    print(f"• Root Mean Squared Error: ${results_salary['rmse']:,.0f}")
    
    print(f"\nTop 5 Most Important Features:")
    top_features = results_salary['feature_importance'].head()
    for idx, row in top_features.iterrows():
        print(f"  {row['rank']}. {row['feature']}: {row['importance']:.4f}")
    
    # Test 2: Product Sales Prediction
    print("\n\n📈 TEST 2: Product Sales Prediction")
    print("-" * 40)
    
    df_sales = create_product_sales_dataset()
    print(f"Dataset created with {len(df_sales)} products")
    print(f"Target: 'monthly_sales'")
    print(f"Features: {len(df_sales.columns)-1}")
    print(f"Dataset preview:")
    print(df_sales.head())
    print()
    
    print("Running Random Forest Regression Analysis...")
    print("=" * 50)
    
    # Run the Random Forest regressor
    results_sales = run_random_forest_regressor(df_sales, 'monthly_sales')
    
    print("\n" + "=" * 50)
    print("SALES PREDICTION ANALYSIS COMPLETE!")
    print("=" * 50)
    
    # Display key results
    print(f"\nKey Results Summary:")
    print(f"• R² Score: {results_sales['r2_score']:.4f}")
    print(f"• Mean Absolute Error: {results_sales['mae']:,.2f} units")
    print(f"• Root Mean Squared Error: {results_sales['rmse']:,.2f} units")
    
    print(f"\nTop 5 Most Important Features:")
    top_features = results_sales['feature_importance'].head()
    for idx, row in top_features.iterrows():
        print(f"  {row['rank']}. {row['feature']}: {row['importance']:.4f}")
    
    # Compare both models
    print(f"\n\n🔍 MODEL COMPARISON")
    print("=" * 30)
    print(f"Salary Prediction Model:")
    print(f"  • R² Score: {results_salary['r2_score']:.4f}")
    print(f"  • Top Feature: {results_salary['feature_importance'].iloc[0]['feature']}")
    print(f"  • Model Quality: {'Excellent' if results_salary['r2_score'] > 0.8 else 'Good' if results_salary['r2_score'] > 0.6 else 'Moderate'}")
    
    print(f"\nSales Prediction Model:")
    print(f"  • R² Score: {results_sales['r2_score']:.4f}")
    print(f"  • Top Feature: {results_sales['feature_importance'].iloc[0]['feature']}")
    print(f"  • Model Quality: {'Excellent' if results_sales['r2_score'] > 0.8 else 'Good' if results_sales['r2_score'] > 0.6 else 'Moderate'}")
    
    print(f"\n✅ Both regression models have been successfully tested!")
    print(f"📊 Visualizations saved as 'random_forest_regression_analysis.png'")
    print(f"🎯 The regressor function is ready for your own data!")
