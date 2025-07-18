# Feature Importance Evaluator

A comprehensive, model-agnostic tool for evaluating feature importance in machine learning. This tool supports both classification and regression tasks, works with Pandas and Spark DataFrames, and provides automated preprocessing with user visibility.

## Features

- **Model-Agnostic**: Supports multiple ML models (Random Forest, CatBoost, Linear Models, Mutual Information)
- **Dual Task Support**: Works with both classification and regression problems
- **Automated Preprocessing**: Handles missing values, categorical encoding, and feature validation
- **Multiple Importance Methods**: Default (built-in) and permutation importance
- **Consensus Ranking**: Weighted consensus across multiple models
- **Rich Visualizations**: Heatmaps, performance plots, and importance rankings
- **Performance Optimized**: Parallel model training, caching, and smart sampling
- **Spark Compatible**: Works with both Pandas and PySpark DataFrames

## Installation

```bash
# Required dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Optional dependencies
pip install catboost  # For boosted tree models
pip install pyspark   # For Spark DataFrame support
```

## Quick Start

### Basic Usage

```python
from feature_importance_evaluator import FeatureImportanceEvaluator
from preprocessing import preprocess_data

# Step 1: Preprocess your data
processed_df, info = preprocess_data(
    df=your_dataframe,
    target='target_column',
    problem_type='regression'  # or 'classification'
)

# Review preprocessing warnings
for warning in info['warnings']:
    print(f"Warning: {warning}")

# Step 2: Evaluate feature importance
evaluator = FeatureImportanceEvaluator()
results = evaluator.evaluate(
    df=processed_df,
    target='target_column',
    problem_type='regression',
    models=['tree', 'boosted_tree'],  # or 'all'
    importance_methods=['default', 'permutation']
)

# Step 3: View results
print("Top 10 Important Features:")
print(results['consensus_importance'].head(10))
```

### Using with Existing Random Forest Code

The evaluator integrates seamlessly with the existing Random Forest scripts:

```python
# Option 1: Use existing Random Forest directly
from random_forest_regressor import run_random_forest_regressor
rf_results = run_random_forest_regressor(df, 'target')

# Option 2: Use Feature Importance Evaluator for comprehensive analysis
evaluator = FeatureImportanceEvaluator()
results = evaluator.evaluate(df, 'target', 'regression', models=['tree'])
```

## Models Supported

1. **tree** (Random Forest)
   - Best for: Non-linear relationships, robust to outliers
   - Importance: Feature importances based on impurity decrease

2. **boosted_tree** (CatBoost)
   - Best for: High accuracy, handles categoricals natively
   - Importance: Multiple methods including LossFunctionChange

3. **linear** (Logistic/Linear Regression)
   - Best for: Linear relationships, interpretability
   - Importance: Absolute coefficient values

4. **statistical** (Mutual Information)
   - Best for: Model-free importance, non-linear dependencies
   - Importance: Information-theoretic measure

## Preprocessing Features

The `preprocess_data` function handles:

- **Missing Values**:
  - Numeric: Median imputation
  - Categorical: 'unknown' category
  - Target (regression): Median imputation
  - Target (classification): Row removal

- **Categorical Encoding**:
  - Automatic detection of categorical features
  - Label encoding with tracking
  - High cardinality warnings

- **Data Validation**:
  - Class imbalance detection
  - Constant feature detection
  - Missing value warnings
  - Rare class warnings

## Example Scripts

### 1. Regression Example

```python
# See example_usage.py for full code
from example_usage import regression_example

results = regression_example()
# Analyzes house price prediction with multiple models
```

### 2. Classification Example

```python
from example_usage import classification_example

results = classification_example()
# Analyzes customer churn with class imbalance handling
```

### 3. Real Data from S3

```python
import pandas as pd
from preprocessing import preprocess_data
from feature_importance_evaluator import FeatureImportanceEvaluator

# Load your S3 data
df = pd.read_parquet('s3://your-bucket/your-data.parquet')

# Preprocess (handles all categorical encoding automatically)
processed_df, info = preprocess_data(df, 'net_revenue', 'regression')

# Evaluate
evaluator = FeatureImportanceEvaluator()
results = evaluator.evaluate(
    processed_df, 'net_revenue', 'regression',
    models=['tree', 'boosted_tree']
)
```

## Output Structure

The evaluator returns a comprehensive results dictionary:

```python
{
    'models': {
        'tree': {
            'model': <trained model>,
            'metrics': {'r2_score': 0.85, ...},
            'importance': {
                'default': DataFrame,
                'permutation': DataFrame
            }
        },
        ...
    },
    'consensus_importance': DataFrame,  # Weighted consensus ranking
    'visualizations': {
        'importance_heatmap': 'filename.png',
        'consensus_importance': 'filename.png',
        'tree_performance': 'filename.png',
        ...
    },
    'metadata': {
        'problem_type': 'regression',
        'n_features': 15,
        'n_samples': 1000,
        ...
    }
}
```

## Visualizations

The tool generates several visualizations:

1. **Feature Importance Heatmap**: Compares importance across all models
2. **Consensus Importance Bar Chart**: Top features by weighted consensus
3. **Model Performance Plots**: 
   - Classification: Confusion matrix, ROC curves, metrics summary
   - Regression: Actual vs predicted, residuals, metrics summary

## Performance Optimization

- **Caching**: Preprocessed data and results are cached automatically
- **Parallel Training**: Models train in parallel using ProcessPoolExecutor
- **Smart Sampling**: Large datasets sampled for permutation importance
- **Spark Support**: Handles distributed DataFrames with automatic conversion

## Advanced Usage

### Custom Model Parameters

```python
results = evaluator.evaluate(
    df, 'target', 'classification',
    models=['tree', 'boosted_tree'],
    model_params={
        'tree': {'n_estimators': 200, 'max_depth': 15},
        'boosted_tree': {'iterations': 500, 'learning_rate': 0.05}
    }
)
```

### Preprocessing Options

```python
processed_df, info = preprocess_data(
    df, 'target', 'regression',
    scale_features=True,           # Standardize numeric features
    max_cardinality=50,           # Warn for high cardinality
    min_samples_per_class=20      # Minimum class size
)
```

### Using Only Specific Importance Methods

```python
# Just built-in importance (faster)
results = evaluator.evaluate(
    df, 'target', 'regression',
    importance_methods=['default']
)

# Just permutation importance (more robust)
results = evaluator.evaluate(
    df, 'target', 'regression',
    importance_methods=['permutation'],
    sample_size_for_permutation=5000
)
```

## Files in This Repository

- `feature_importance_evaluator.py`: Main evaluator class
- `preprocessing.py`: Data preprocessing module
- `example_usage.py`: Comprehensive examples
- `random_forest_classifier.py`: Standalone Random Forest classifier
- `random_forest_regressor.py`: Standalone Random Forest regressor

## Requirements

- Python 3.7+
- pandas >= 1.0.0
- numpy >= 1.18.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.0.0
- seaborn >= 0.11.0
- catboost >= 0.24 (optional)
- pyspark >= 3.0.0 (optional)

## License

MIT License - see LICENSE file for details.