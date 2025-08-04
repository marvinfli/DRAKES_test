#!/bin/bash
#SBATCH --job-name=drakes_dna_baseline_sweep
#SBATCH --account=sitanc_lab
#SBATCH --partition gpu,seas_gpu
#SBATCH --gres gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/baseline_%A_%a.out
#SBATCH --error=logs/baseline_%A_%a.err

echo "========================================"
echo "DRAKES DNA Baseline Finetuning + Evaluation Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load conda and activate environment
source ~/.bashrc
conda activate sedd

# Set up environment
export CUDA_VISIBLE_DEVICES=0

# Define seeds for reproducibility
SEEDS=(42 123 456)

# Get seed for this array task
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Generate wandb name with timestamp, baseline identifier, and seed
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WANDB_NAME="baseline_seed${SEED}_${TIMESTAMP}_epochs=500"
echo "WandB Run Name: $WANDB_NAME"
echo "Seed: $SEED"

echo "========================================"
echo "STEP 1: Running Finetuning"
echo "========================================"

# Run finetuning with wandb name and seed
python finetune_reward_bp.py --name "$WANDB_NAME" --seed $SEED

EXIT_CODE=$?

echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Baseline run completed successfully!"
    echo "Run name: $WANDB_NAME"
    echo "Seed: $SEED"
else
    echo "ERROR: Baseline run failed with exit code $EXIT_CODE"
    echo "Run name: $WANDB_NAME"
    echo "Seed: $SEED"
fi
echo "========================================"

exit $EXIT_CODE
