"""
Utility functions for tests
"""
import os
import shutil


def clean_output_directory():
    """Clean the test output directory before running tests."""
    # Get the parent directory (project root) and then navigate to test/output
    project_root = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(project_root, 'test', 'output')
    
    # Remove the directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Cleaned existing output directory: {output_dir}")
    
    # Recreate the directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created fresh output directory: {output_dir}")
    
    return output_dir 