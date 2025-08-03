#!/bin/bash
#SBATCH --job-name=drakes_dna_finetune_eval_epoch500
#SBATCH --account=sitanc_lab
#SBATCH --partition gpu,seas_gpu
#SBATCH --gres gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/finetune_eval_%j.out
#SBATCH --error=logs/finetune_eval_%j.err

echo "========================================"
echo "DRAKES DNA Finetuning + Evaluation Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load conda and activate environment
source ~/.bashrc
conda activate sedd

# Set up environment
export CUDA_VISIBLE_DEVICES=0

# Generate interesting wandb name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WANDB_NAME="dna_conductor_${TIMESTAMP}_epochs=500"
echo "WandB Run Name: $WANDB_NAME"

echo "========================================"
echo "STEP 1: Running Finetuning"
echo "========================================"

# Run finetuning with interesting wandb name and default parameters
python finetune_reward_bp.py --name "$WANDB_NAME" 

if [ $? -ne 0 ]; then
    echo "ERROR: Finetuning failed!"
    exit 1
fi
