#!/usr/bin/env python3
"""
Simple runner script for DRAKES DNA model evaluation.
Usage example for eval_models.py
"""

import os
import sys
import argparse
from datetime import datetime
from eval_models import ModelEvaluator

# python run_evaluation.py /path/to/config.yaml
def main():
    """Run evaluation with specified configuration file."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run DRAKES DNA model evaluation.")
    parser.add_argument('config_path', type=str, help='Path to the configuration YAML file.')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory name. If not specified, uses timestamp-based naming.')
    args = parser.parse_args()
    
    # Output directory with timestamp or custom name
    base_path = os.environ.get('BASE_PATH')
    if not base_path:
        raise EnvironmentError("BASE_PATH environment variable is not set.")
    
    if args.output_dir:
        # Use user-specified output directory name
        output_dir = f"{base_path}/evaluation/{args.output_dir}"
    else:
        # Use timestamp-based naming (default behavior)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{base_path}/evaluation/evaluation_results_{timestamp}"
    
    print(f"Starting evaluation with config: {args.config_path}")
    print(f"Output will be saved to: {output_dir}")
    
    # Create and run evaluator
    evaluator = ModelEvaluator(args.config_path, output_dir)
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