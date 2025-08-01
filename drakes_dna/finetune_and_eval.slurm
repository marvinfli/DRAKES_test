#!/bin/bash
#SBATCH --job-name=drakes_dna_finetune_eval
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
WANDB_NAME="dna_conductor_${TIMESTAMP}"
echo "WandB Run Name: $WANDB_NAME"

# Base paths
BASE_PATH="/n/holylabs/LABS/sitanc_lab/Users/mfli/DRAKES_test/data_and_model"
RESULTS_DIR="$BASE_PATH/mdlm/reward_bp_results_final"

echo "========================================"
echo "STEP 1: Running Finetuning"
echo "========================================"

# Run finetuning with interesting wandb name and default parameters
python finetune_reward_bp.py --name "$WANDB_NAME" 

if [ $? -ne 0 ]; then
    echo "ERROR: Finetuning failed!"
    exit 1
fi

echo "========================================"
echo "STEP 2: Processing Finetuned Model"
echo "========================================"

# Determine the run directory (this will be created by the finetuning script)
# Format: alpha{alpha}_accum{num_accum_steps}_bsz{batch_size}_truncate{truncate_steps}_temp{gumbel_temp}_clip{gradnorm_clip}_{name}_{timestamp}
RUN_DIR_NAME=$(ls -td "$RESULTS_DIR"/*/ | head -1)
RUN_DIR=$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "$RUN_DIR_NAME" | head -1)

if [ -z "$RUN_DIR" ]; then
    echo "ERROR: Could not find run directory matching pattern: $RUN_DIR_NAME"
    echo "Available directories in $RESULTS_DIR:"
    ls -la "$RESULTS_DIR/"
    exit 1
fi

echo "Found run directory: $RUN_DIR"

# Find the last saved model (highest epoch number)
LAST_MODEL=$(find "$RUN_DIR" -name "model_*.ckpt" | sort -V | tail -1)

if [ -z "$LAST_MODEL" ]; then
    echo "ERROR: No model checkpoints found in $RUN_DIR"
    exit 1
fi

echo "Using model: $LAST_MODEL"

# Copy the last model to finetuned.ckpt for evaluation
cp "$LAST_MODEL" "$RUN_DIR/finetuned.ckpt"
echo "Copied $LAST_MODEL to $RUN_DIR/finetuned.ckpt"

echo "========================================"
echo "STEP 3: Creating Evaluation Configuration"
echo "========================================"

# Get relative path from base_path
REL_RUN_DIR=$(realpath --relative-to="$BASE_PATH" "$RUN_DIR")
echo "Relative run directory: $REL_RUN_DIR"

# Create custom evaluation config for this run
EVAL_CONFIG="eval_config_${WANDB_NAME}.yaml"
cat > "$EVAL_CONFIG" << EOF
# DRAKES DNA Model Evaluation Configuration
# Generated for run: $WANDB_NAME

# Base paths
base_path: "$BASE_PATH"

# Random seed
seed: 42

# CUDA device
cuda_device: 0

# Sampling parameters
num_sample_batches: 10
num_samples_per_batch: 64

# Likelihood calculation parameters
likelihood_steps: 128

# Model checkpoints to evaluate
models:
  finetuned_model: "$REL_RUN_DIR/finetuned.ckpt"
  pretrained_model: "mdlm/outputs_gosai/pretrained.ckpt"
  zero_alpha_model: "mdlm/reward_bp_results_final/zero_alpha.ckpt"
  cfg_model: "mdlm/outputs_gosai/cfg.ckpt"

# CFG parameters
cfg_weight: 10
cfg_prob: 0.1

# Controlled sampling options (for pretrained model)
use_regular_sampling: true
use_tds_sampling: false
use_cg_sampling: false
use_smc_sampling: false

# TDS sampling parameters
tds_alpha: 0.5
tds_guidance_scale: 1000

# CG sampling parameters
cg_guidance_scale: 300000

# SMC sampling parameters
smc_alpha: 0.5

# Additional evaluation options
save_plots: true
save_sequences: true
EOF

echo "Created evaluation config: $EVAL_CONFIG"

echo "========================================"
echo "STEP 4: Running Evaluation"
echo "========================================"

# Create output directory for evaluation results
EVAL_OUTPUT_DIR="$BASE_PATH/evaluation_results_${WANDB_NAME}"
echo "Evaluation output directory: $EVAL_OUTPUT_DIR"

# Run evaluation with the new model
python eval_models.py "$EVAL_CONFIG" --output-dir "$EVAL_OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed!"
    exit 1
fi

echo "========================================"
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "========================================"
echo ""
echo "Results Summary:"
echo "  • Finetuned model: $RUN_DIR/finetuned.ckpt"
echo "  • Evaluation config: $EVAL_CONFIG"
echo "  • Evaluation results: $EVAL_OUTPUT_DIR"
echo ""
echo "Key files to check:"
echo "  • Training logs: $RUN_DIR/log.txt"
echo "  • Evaluation summary: $EVAL_OUTPUT_DIR/evaluation_summary.csv"
echo "  • Complete results: $EVAL_OUTPUT_DIR/evaluation_results.json"
echo "  • Motif analysis: $EVAL_OUTPUT_DIR/motif_summary.csv"
echo ""
echo "WandB Run: $WANDB_NAME"
echo "========================================" 
