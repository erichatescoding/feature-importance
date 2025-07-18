"""
Utility functions for models
"""
import os
import shutil


def clean_catboost_info():
    """Clean the catboost_info directory before training."""
    catboost_info_dir = 'catboost_info'
    
    # Remove the directory if it exists
    if os.path.exists(catboost_info_dir):
        try:
            shutil.rmtree(catboost_info_dir)
            print(f"Cleaned existing catboost_info directory")
        except Exception as e:
            print(f"Warning: Could not clean catboost_info: {e}")
    
    return catboost_info_dir 