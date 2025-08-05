#!/bin/bash
#SBATCH --job-name=model_comparison_diff_steps
#SBATCH --account=sitanc_lab
#SBATCH --partition gpu,seas_gpu
#SBATCH --gres gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=50G
#SBATCH --time=24:00:00
#SBATCH --output=logs/model_comparison_%j.out
#SBATCH --error=logs/model_comparison_%j.err

echo "========================================"
echo "DRAKES DNA Model Comparison Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load conda and activate environment
source ~/.bashrc
conda activate sedd
module load cuda gcc

# Set up environment
export CUDA_VISIBLE_DEVICES=0

# Set BASE_PATH environment variable
export BASE_PATH="/n/holylabs/LABS/sitanc_lab/Users/mfli/soc-curriculum/DRAKES/data_and_model"

# Change to the project directory
cd /n/holylabs/LABS/sitanc_lab/Users/mfli/soc-curriculum/DRAKES/drakes_dna

# Generate run name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="model_comparison_250805"

echo "========================================"
echo "Configuration:"
echo "  Config file: experiments/marvin/250805/eval_config_250804.yaml"
echo "  Output directory: $OUTPUT_DIR"
echo "  Base path: $BASE_PATH"
echo "========================================"

# Run evaluation
python run_evaluation.py experiments/marvin/250805/eval_config_250804.yaml --output_dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Model comparison evaluation completed successfully!"
    echo "Output directory: $OUTPUT_DIR"
    echo "Results location: $BASE_PATH/evaluation/$OUTPUT_DIR"
else
    echo "ERROR: Model comparison evaluation failed with exit code $EXIT_CODE"
    echo "Check the error logs for details."
fi
echo "========================================"

exit $EXIT_CODE 