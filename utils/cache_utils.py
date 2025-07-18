"""
Cache management utilities
"""
import os
import shutil
import warnings


def clean_cache_directory(cache_dir: str, cache_name: str = "cache") -> None:
    """
    Clean a cache directory by removing all its contents.
    
    Args:
        cache_dir: Path to the cache directory
        cache_name: Name of the cache for logging purposes
    """
    if os.path.exists(cache_dir):
        try:
            # Remove all contents but keep the directory
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            print(f"Cleaned {cache_name} directory: {cache_dir}")
        except Exception as e:
            warnings.warn(f"Could not clean {cache_name}: {e}")
    
    # Ensure directory exists
    os.makedirs(cache_dir, exist_ok=True)


def clean_all_caches():
    """Clean preprocessing cache, feature importance cache, and output plots directory."""
    clean_cache_directory(".preprocessing_cache", "preprocessing cache")
    clean_cache_directory(".feature_importance_cache", "feature importance cache")
    clean_cache_directory("output", "output plots") 