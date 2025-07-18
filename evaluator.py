"""
Feature Importance Evaluator

A comprehensive, model-agnostic tool for evaluating feature importance in machine learning.
This is the main entry point that coordinates all modules.

Usage:
    from evaluator import FeatureImportanceEvaluator
    from preprocessing import preprocess_data
    
    # Preprocess data
    processed_df, info = preprocess_data(df, 'target', 'regression')
    
    # Evaluate feature importance
    evaluator = FeatureImportanceEvaluator()
    results = evaluator.evaluate(
        df=processed_df,
        target='target',
        problem_type='regression',
        models=['tree', 'boosted_tree'],
        importance_methods=['default', 'permutation']
    )
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from sklearn.model_selection import train_test_split
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import pickle
import hashlib
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.cache_utils import clean_cache_directory

# Import model modules
from models.classification.tree_classifier import TreeClassifier
from models.regression.tree_regressor import TreeRegressor
from models.statistical.mutual_information import MutualInformationEvaluator

# Import visualization
from visualization.plots import FeatureImportancePlotter

# Import importance methods
from importance.model_importance import get_model_importance
from importance.permutation import get_permutation_importance

# Check for optional dependencies
try:
    from models.classification.boosted_tree_classifier import BoostedTreeClassifier
    from models.regression.boosted_tree_regressor import BoostedTreeRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not installed. Install with: pip install catboost")

try:
    from models.classification.linear_classifier import LinearClassifier
    from models.regression.linear_regressor import LinearRegressor
except ImportError:
    warnings.warn("Linear models not available")

# Check for Spark
try:
    from pyspark.sql import DataFrame as SparkDataFrame
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    SparkDataFrame = type(None)


class FeatureImportanceEvaluator:
    """
    Main evaluator class that coordinates all feature importance evaluation.
    """
    
    def __init__(self, cache_dir: str = ".feature_importance_cache", output_dir: str = "output", clean_cache: bool = True):
        """
        Initialize the Feature Importance Evaluator.
        
        Args:
            cache_dir: Directory for caching results
            output_dir: Directory for saving visualization plots
            clean_cache: Whether to clean the cache directory and output directory on initialization
        """
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        
        if clean_cache:
            clean_cache_directory(cache_dir, "feature importance cache")
            # Also clean the output directory to remove old plots
            clean_cache_directory(output_dir, "output plots")
        elif not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Initialize plotter with output directory
        self.plotter = FeatureImportancePlotter(output_dir=output_dir)
        
        # Model configurations
        self.model_configs = {
            'tree': {
                'name': 'Random Forest',
                'available': True
            },
            'boosted_tree': {
                'name': 'CatBoost',
                'available': CATBOOST_AVAILABLE
            },
            'linear': {
                'name': 'Linear Model',
                'available': True
            },
            'statistical': {
                'name': 'Mutual Information',
                'available': True
            }
        }
        
        # Model ranking weights for consensus
        self.model_weights = {
            'boosted_tree': 0.35,
            'tree': 0.30,
            'linear': 0.20,
            'statistical': 0.15
        }
    
    def evaluate(
        self,
        df: Union[pd.DataFrame, SparkDataFrame],
        target: str,
        problem_type: str,
        models: Union[str, List[str]] = 'all',
        importance_methods: List[str] = ['default'],
        model_params: Optional[Dict[str, Dict]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        n_jobs: int = -1,
        sample_size_for_permutation: Optional[int] = 10000
    ) -> Dict[str, Any]:
        """
        Evaluate feature importance using specified models and methods.
        
        Args:
            df: Input dataframe (Pandas or Spark)
            target: Target column name
            problem_type: 'classification' or 'regression'
            models: List of models to use or 'all'
            importance_methods: List of importance methods ['default', 'permutation']
            model_params: Optional custom parameters for models
            test_size: Test set size for evaluation
            random_state: Random seed
            n_jobs: Number of parallel jobs
            sample_size_for_permutation: Sample size for permutation importance
            
        Returns:
            Dictionary containing results for all models and visualizations
        """
        
        # Validate inputs
        if problem_type not in ['classification', 'regression']:
            raise ValueError("problem_type must be 'classification' or 'regression'")
        
        # Convert Spark DataFrame to Pandas if needed
        if SPARK_AVAILABLE and isinstance(df, SparkDataFrame):
            print("Converting Spark DataFrame to Pandas...")
            try:
                df = df.toPandas()
            except Exception as e:
                warnings.warn(f"Failed to convert Spark DataFrame: {e}")
                raise
        
        # Prepare model list
        if models == 'all':
            models = list(self.model_configs.keys())
        elif isinstance(models, str):
            models = [models]
        
        # Filter out unavailable models
        available_models = []
        for model in models:
            if not self.model_configs[model]['available']:
                warnings.warn(f"{self.model_configs[model]['name']} not available. Skipping {model} model.")
                continue
            available_models.append(model)
        
        if not available_models:
            raise ValueError("No models available to run.")
        
        # Prepare data
        X = df.drop(columns=[target])
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if problem_type == 'classification' else None
        )
        
        print(f"\nEvaluating feature importance for {problem_type} task")
        print(f"Dataset shape: {X.shape}")
        print(f"Models to evaluate: {', '.join(available_models)}")
        print(f"Importance methods: {', '.join(importance_methods)}")
        print("-" * 50)
        
        # Train models in parallel
        results = {}
        
        # Use ProcessPoolExecutor for parallel model training
        with ProcessPoolExecutor(max_workers=min(len(available_models), n_jobs if n_jobs > 0 else os.cpu_count())) as executor:
            # Submit all model training tasks
            future_to_model = {
                executor.submit(
                    self._train_and_evaluate_model,
                    model_type=model,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    problem_type=problem_type,
                    importance_methods=importance_methods,
                    model_params=model_params.get(model, {}) if model_params else {},
                    sample_size_for_permutation=sample_size_for_permutation
                ): model
                for model in available_models
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    model_results = future.result()
                    results[model_name] = model_results
                    print(f"✓ Completed evaluation for {model_name}")
                except Exception as exc:
                    print(f"✗ {model_name} generated an exception: {exc}")
                    results[model_name] = {'error': str(exc)}
        
        # Calculate consensus feature importance
        consensus_importance = self._calculate_consensus_importance(results)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        visualizations = self.plotter.generate_all_visualizations(
            results={'models': results},
            consensus_importance=consensus_importance,
            problem_type=problem_type,
            X_test=X_test,
            y_test=y_test
        )
        
        # Prepare final results
        final_results = {
            'models': results,
            'consensus_importance': consensus_importance,
            'visualizations': visualizations,
            'metadata': {
                'problem_type': problem_type,
                'n_features': X.shape[1],
                'n_samples': X.shape[0],
                'feature_names': list(X.columns),
                'models_used': list(results.keys()),
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Cache results
        self._cache_results(final_results)
        
        print("\n" + "="*50)
        print("EVALUATION COMPLETE")
        print("="*50)
        self._print_summary(final_results)
        
        return final_results
    
    def _train_and_evaluate_model(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        problem_type: str,
        importance_methods: List[str],
        model_params: Dict[str, Any],
        sample_size_for_permutation: Optional[int]
    ) -> Dict[str, Any]:
        """Train and evaluate a single model."""
        
        # Get model instance
        model = self._get_model_instance(model_type, problem_type, model_params)
        
        # Handle statistical models separately
        if model_type == 'statistical':
            evaluator = MutualInformationEvaluator()
            importance_df = evaluator.calculate_importance(X_train, y_train, problem_type)
            return {
                'importance': {'default': importance_df},
                'metrics': {'mutual_information': importance_df['importance'].mean()}
            }
        
        # Train model
        model.train(X_train, y_train)
        
        # Make predictions and evaluate
        metrics = model.evaluate(X_test, y_test)
        y_pred = model.predict(X_test)
        
        # Get feature importance
        importance_results = {}
        
        # Default importance
        if 'default' in importance_methods:
            importance_results['default'] = get_model_importance(model, model_type)
        
        # Permutation importance
        if 'permutation' in importance_methods:
            importance_results['permutation'] = get_permutation_importance(
                model, X_test, y_test, problem_type, 
                sample_size_for_permutation, model_type
            )
        
        return {
            'model': model,
            'metrics': metrics,
            'importance': importance_results,
            'y_pred': y_pred,
            'y_test': y_test
        }
    
    def _get_model_instance(
        self, 
        model_type: str, 
        problem_type: str,
        params: Dict[str, Any]
    ):
        """Get model instance based on type and problem."""
        
        if model_type == 'tree':
            if problem_type == 'classification':
                return TreeClassifier(**params)
            else:
                return TreeRegressor(**params)
        elif model_type == 'boosted_tree':
            if problem_type == 'classification':
                return BoostedTreeClassifier(**params)
            else:
                return BoostedTreeRegressor(**params)
        elif model_type == 'linear':
            if problem_type == 'classification':
                return LinearClassifier(**params)
            else:
                return LinearRegressor(**params)
        elif model_type == 'statistical':
            # Statistical model is handled separately in _train_and_evaluate_model
            return None
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _calculate_consensus_importance(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Calculate weighted consensus feature importance across all models."""
        
        # Collect all importance scores
        all_importances = {}
        model_performances = {}
        
        for model_name, model_results in results.items():
            if 'error' in model_results:
                continue
                
            # Get model performance score
            if 'metrics' in model_results:
                metrics = model_results['metrics']
                if 'r2_score' in metrics:
                    performance = metrics['r2_score']
                elif 'accuracy' in metrics:
                    performance = metrics['accuracy']
                else:
                    performance = 0.5
            else:
                performance = 0.5
            
            model_performances[model_name] = max(0, performance)  # Ensure non-negative
            
            # Get importance scores (prefer default over permutation)
            if 'importance' in model_results:
                if 'default' in model_results['importance']:
                    importance_df = model_results['importance']['default']
                elif 'permutation' in model_results['importance']:
                    importance_df = model_results['importance']['permutation']
                else:
                    continue
                
                for _, row in importance_df.iterrows():
                    feature = row['feature']
                    if feature not in all_importances:
                        all_importances[feature] = {}
                    all_importances[feature][model_name] = row['importance']
        
        # Calculate weighted consensus
        consensus_scores = {}
        
        for feature, model_scores in all_importances.items():
            weighted_sum = 0
            weight_sum = 0
            
            for model_name, importance in model_scores.items():
                # Use predefined weights combined with performance
                base_weight = self.model_weights.get(model_name, 0.1)
                performance_weight = model_performances.get(model_name, 0.5)
                combined_weight = base_weight * performance_weight
                
                weighted_sum += importance * combined_weight
                weight_sum += combined_weight
            
            consensus_scores[feature] = weighted_sum / weight_sum if weight_sum > 0 else 0
        
        # Create consensus DataFrame
        consensus_df = pd.DataFrame([
            {'feature': feature, 'importance': score}
            for feature, score in consensus_scores.items()
        ]).sort_values('importance', ascending=False)
        
        # Add rank
        consensus_df['rank'] = range(1, len(consensus_df) + 1)
        
        return consensus_df
    
    def _cache_results(self, results: Dict[str, Any]):
        """Cache results for future use."""
        
        # Create cache key from results metadata
        cache_key = hashlib.md5(
            json.dumps(results['metadata'], sort_keys=True).encode()
        ).hexdigest()
        
        cache_path = os.path.join(self.cache_dir, f'results_{cache_key}.pkl')
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"\nResults cached to: {cache_path}")
        except Exception as e:
            warnings.warn(f"Failed to cache results: {e}")
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print a summary of the evaluation results."""
        
        print("\nCONSENSUS FEATURE IMPORTANCE (Top 10):")
        print("-" * 50)
        consensus = results['consensus_importance'].head(10)
        for _, row in consensus.iterrows():
            print(f"{row['rank']:2d}. {row['feature']:30s} {row['importance']:.4f}")
        
        print("\nMODEL PERFORMANCE SUMMARY:")
        print("-" * 50)
        
        for model_name, model_results in results['models'].items():
            if 'error' in model_results:
                print(f"{model_name}: ERROR - {model_results['error']}")
                continue
                
            print(f"\n{self.model_configs[model_name]['name']}:")
            
            if 'metrics' in model_results:
                metrics = model_results['metrics']
                if results['metadata']['problem_type'] == 'classification':
                    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
                    print(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
                else:
                    print(f"  R² Score: {metrics.get('r2_score', 0):.4f}")
                    print(f"  RMSE: {metrics.get('rmse', 0):.4f}")
        
        print("\nVISUALIZATIONS SAVED:")
        print("-" * 50)
        for viz_name, filename in results['visualizations'].items():
            print(f"  {viz_name}: {filename}") 