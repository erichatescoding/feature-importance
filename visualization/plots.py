"""
Visualization Module for Feature Importance Evaluator

Handles all plotting functionality including:
- Feature importance heatmaps
- Consensus importance bar charts
- Model performance plots (confusion matrices, ROC curves, residual plots)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
from datetime import datetime
import os


class FeatureImportancePlotter:
    """Handles visualization of feature importance results."""
    
    def __init__(self, figsize_base: Tuple[int, int] = (10, 6), output_dir: str = "."):
        """
        Initialize the plotter.
        
        Args:
            figsize_base: Base figure size for plots
            output_dir: Directory to save plots (default: current directory)
        """
        self.figsize_base = figsize_base
        self.output_dir = output_dir
        plt.style.use('default')
        
        # Create output directory if it doesn't exist
        if self.output_dir != ".":
            os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_importance_heatmap(
        self,
        results: Dict[str, Any],
        consensus_importance: pd.DataFrame,
        save_path: str
    ) -> None:
        """
        Create heatmap comparing feature importance across models.
        
        Args:
            results: Model results dictionary
            consensus_importance: Consensus importance DataFrame
            save_path: Path to save the plot
        """
        # Prepare data for heatmap
        features = consensus_importance['feature'].tolist()[:20]  # Top 20 features
        heatmap_data = []
        model_names = []
        
        for model_name, model_results in results.items():
            if 'error' in model_results or 'importance' not in model_results:
                continue
                
            model_display_name = self._get_model_display_name(model_name)
            model_names.append(model_display_name)
            
            # Get importance scores
            if 'default' in model_results['importance']:
                importance_df = model_results['importance']['default']
            else:
                importance_df = list(model_results['importance'].values())[0]
            
            # Create importance dict
            importance_dict = dict(zip(importance_df['feature'], importance_df['importance']))
            
            # Get scores for top features
            scores = [importance_dict.get(feature, 0) for feature in features]
            heatmap_data.append(scores)
        
        if not heatmap_data:
            return
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            np.array(heatmap_data).T,
            xticklabels=model_names,
            yticklabels=features,
            cmap='YlOrRd',
            annot=True,
            fmt='.3f',
            cbar_kws={'label': 'Importance Score'}
        )
        plt.title('Feature Importance Comparison Across Models', fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_consensus_importance(
        self,
        consensus_importance: pd.DataFrame,
        save_path: str,
        top_n: int = 20
    ) -> None:
        """
        Plot consensus feature importance bar chart.
        
        Args:
            consensus_importance: Consensus importance DataFrame
            save_path: Path to save the plot
            top_n: Number of top features to show
        """
        # Select top features
        top_features = consensus_importance.head(top_n)
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        
        bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Consensus Importance Score')
        plt.title('Consensus Feature Importance Ranking', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (idx, row) in enumerate(top_features.iterrows()):
            plt.text(row['importance'] + 0.002, i, f'{row["importance"]:.3f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_model_performance(
        self,
        model_name: str,
        model_results: Dict[str, Any],
        problem_type: str,
        save_path: str
    ) -> None:
        """
        Create performance plots for a single model.
        
        Args:
            model_name: Name of the model
            model_results: Model results dictionary
            problem_type: 'classification' or 'regression'
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        model_display_name = self._get_model_display_name(model_name)
        
        if problem_type == 'classification':
            self._plot_classification_performance(
                model_display_name, model_results, axes
            )
        else:
            self._plot_regression_performance(
                model_display_name, model_results, axes
            )
        
        plt.suptitle(f'{model_display_name} Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_classification_performance(
        self,
        model_display_name: str,
        model_results: Dict[str, Any],
        axes: List[plt.Axes]
    ) -> None:
        """Plot classification performance metrics."""
        y_test = model_results.get('y_test')
        y_pred = model_results.get('y_pred')
        metrics = model_results.get('metrics', {})
        
        # Confusion Matrix
        if y_test is not None and y_pred is not None:
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
            axes[0].set_title(f'{model_display_name} - Confusion Matrix')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('Actual')
        
        # ROC Curve (if available)
        if 'y_proba' in metrics and y_test is not None:
            y_proba = metrics['y_proba']
            if len(np.unique(y_test)) == 2:  # Binary classification
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                axes[1].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
                axes[1].plot([0, 1], [0, 1], 'k--')
                axes[1].set_xlabel('False Positive Rate')
                axes[1].set_ylabel('True Positive Rate')
                axes[1].set_title(f'{model_display_name} - ROC Curve')
                axes[1].legend()
            else:
                axes[1].text(0.5, 0.5, 'ROC Curve\n(Binary Classification Only)', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title(f'{model_display_name} - ROC Curve')
        else:
            axes[1].axis('off')
        
        # Metrics Summary
        axes[2].axis('off')
        summary_text = f"{model_display_name} Metrics\n\n"
        summary_text += f"Accuracy: {metrics.get('accuracy', 0):.4f}\n"
        summary_text += f"Precision: {metrics.get('precision', 0):.4f}\n"
        summary_text += f"Recall: {metrics.get('recall', 0):.4f}\n"
        summary_text += f"F1-Score: {metrics.get('f1_score', 0):.4f}"
        
        axes[2].text(0.1, 0.5, summary_text, transform=axes[2].transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        
    def _plot_regression_performance(
        self,
        model_display_name: str,
        model_results: Dict[str, Any],
        axes: List[plt.Axes]
    ) -> None:
        """Plot regression performance metrics."""
        y_test = model_results.get('y_test')
        y_pred = model_results.get('y_pred')
        metrics = model_results.get('metrics', {})
        
        if y_test is not None and y_pred is not None:
            # Actual vs Predicted
            axes[0].scatter(y_test, y_pred, alpha=0.5)
            axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[0].set_xlabel('Actual')
            axes[0].set_ylabel('Predicted')
            axes[0].set_title(f'{model_display_name} - Actual vs Predicted')
            
            # Residuals Plot
            residuals = y_test - y_pred
            axes[1].scatter(y_pred, residuals, alpha=0.5)
            axes[1].axhline(y=0, color='r', linestyle='--')
            axes[1].set_xlabel('Predicted')
            axes[1].set_ylabel('Residuals')
            axes[1].set_title(f'{model_display_name} - Residuals Plot')
        
        # Metrics Summary
        axes[2].axis('off')
        summary_text = f"{model_display_name} Metrics\n\n"
        summary_text += f"MAE: {metrics.get('mae', 0):.4f}\n"
        summary_text += f"MSE: {metrics.get('mse', 0):.4f}\n"
        summary_text += f"RMSE: {metrics.get('rmse', 0):.4f}\n"
        summary_text += f"R² Score: {metrics.get('r2_score', 0):.4f}"
        
        axes[2].text(0.1, 0.5, summary_text, transform=axes[2].transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen"))
        
    def _get_model_display_name(self, model_name: str) -> str:
        """Get display name for model."""
        display_names = {
            'tree': 'Random Forest',
            'boosted_tree': 'CatBoost',
            'linear': 'Linear Model',
            'statistical': 'Mutual Information'
        }
        return display_names.get(model_name, model_name.title())
        
    def generate_all_visualizations(
        self,
        results: Dict[str, Any],
        consensus_importance: pd.DataFrame,
        problem_type: str,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, str]:
        """
        Generate all visualizations and save to files.
        
        Returns:
            Dictionary mapping visualization types to filenames
        """
        visualizations = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Feature Importance Comparison Heatmap
        heatmap_filename = f'feature_importance_heatmap_{timestamp}.png'
        heatmap_path = os.path.join(self.output_dir, heatmap_filename)
        self.plot_importance_heatmap(results['models'], consensus_importance, heatmap_path)
        visualizations['importance_heatmap'] = heatmap_path
        
        # 2. Individual Model Performance Plots
        for model_name, model_results in results['models'].items():
            if 'error' not in model_results and 'model' in model_results:
                model_filename = f'{model_name}_performance_{timestamp}.png'
                model_path = os.path.join(self.output_dir, model_filename)
                self.plot_model_performance(
                    model_name, model_results, problem_type, model_path
                )
                visualizations[f'{model_name}_performance'] = model_path
        
        # 3. Consensus Feature Importance Bar Chart
        consensus_filename = f'consensus_importance_{timestamp}.png'
        consensus_path = os.path.join(self.output_dir, consensus_filename)
        self.plot_consensus_importance(consensus_importance, consensus_path)
        visualizations['consensus_importance'] = consensus_path
        
        return visualizations 