#!/usr/bin/env python3
"""
Simple runner script for DRAKES DNA model evaluation.
Usage example for eval_models.py
"""

import os
import sys
from datetime import datetime
from eval_models import ModelEvaluator

def main():
    """Run evaluation with default configuration."""
    
    # Configuration file path
    config_path = "eval_config.yaml"
    
    # Output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = os.environ.get('BASE_PATH')
    if not base_path:
        raise EnvironmentError("BASE_PATH environment variable is not set.")
    output_dir = f"{base_path}/evaluation_results_{timestamp}"
    
    print(f"Starting evaluation with config: {config_path}")
    print(f"Output will be saved to: {output_dir}")
    
    # Create and run evaluator
    evaluator = ModelEvaluator(config_path, output_dir)
    evaluator.run_evaluation()
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Check the following files:")
    print(f"  - {output_dir}/evaluation_summary.csv (main results table)")
    print(f"  - {output_dir}/evaluation_results.json (complete results)")
    print(f"  - {output_dir}/motif_summary.csv (motif analysis)")
    print(f"  - {output_dir}/evaluation_*.log (detailed logs)")

if __name__ == "__main__":
    main() 