#!/bin/bash
#SBATCH --job-name=exploration_hyperparam_sweep
#SBATCH --account=sitanc_lab
#SBATCH --partition gpu,seas_gpu
#SBATCH --gres gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --array=0-26
#SBATCH --output=logs/hyperparam_sweep_%A_%a.out
#SBATCH --error=logs/hyperparam_sweep_%A_%a.err

echo "========================================"
echo "DRAKES DNA Exploration Hyperparameter Sweep"
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

# Define hyperparameter grids
INITIAL_INVERSE_TEMP_VALUES=(0.05 0.1 0.5)
INITIAL_RANDOM_NOISING_VALUES=(0.1 0.5 0.7)
END_EXPLORATION_STEP_VALUES=(20 50 100)

# Calculate which combination this array task should run
# 3x3x3 = 27 combinations total (indices 0-26)
NUM_TEMP_VALUES=${#INITIAL_INVERSE_TEMP_VALUES[@]}
NUM_NOISE_VALUES=${#INITIAL_RANDOM_NOISING_VALUES[@]}
NUM_STEP_VALUES=${#END_EXPLORATION_STEP_VALUES[@]}

# Convert linear array index to 3D indices
TEMP_IDX=$((SLURM_ARRAY_TASK_ID / (NUM_NOISE_VALUES * NUM_STEP_VALUES)))
NOISE_IDX=$(((SLURM_ARRAY_TASK_ID % (NUM_NOISE_VALUES * NUM_STEP_VALUES)) / NUM_STEP_VALUES))
STEP_IDX=$((SLURM_ARRAY_TASK_ID % NUM_STEP_VALUES))

# Get the actual parameter values
INITIAL_INVERSE_TEMP=${INITIAL_INVERSE_TEMP_VALUES[$TEMP_IDX]}
INITIAL_RANDOM_NOISING=${INITIAL_RANDOM_NOISING_VALUES[$NOISE_IDX]}
END_EXPLORATION_STEP=${END_EXPLORATION_STEP_VALUES[$STEP_IDX]}

# Generate run name with timestamp and parameters
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="sweep_temp${INITIAL_INVERSE_TEMP}_noise${INITIAL_RANDOM_NOISING}_step${END_EXPLORATION_STEP}_${TIMESTAMP}"

echo "========================================"
echo "Hyperparameters for this run:"
echo "  initial_inverse_temp: $INITIAL_INVERSE_TEMP"
echo "  initial_random_noising: $INITIAL_RANDOM_NOISING"
echo "  end_exploration_step: $END_EXPLORATION_STEP"
echo "  run_name: $RUN_NAME"
echo "========================================"

# Run finetuning with the selected hyperparameters
python finetune_reward_bp.py \
    --name "$RUN_NAME" \
    --initial_inverse_temp $INITIAL_INVERSE_TEMP \
    --initial_random_noising $INITIAL_RANDOM_NOISING \
    --end_exploration_step $END_EXPLORATION_STEP \
    --end_exploration_epoch 500

EXIT_CODE=$?

echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Run completed successfully!"
    echo "Run name: $RUN_NAME"
    echo "Parameters: temp=$INITIAL_INVERSE_TEMP, noise=$INITIAL_RANDOM_NOISING, step=$END_EXPLORATION_STEP"
else
    echo "ERROR: Run failed with exit code $EXIT_CODE"
    echo "Run name: $RUN_NAME"
    echo "Parameters: temp=$INITIAL_INVERSE_TEMP, noise=$INITIAL_RANDOM_NOISING, step=$END_EXPLORATION_STEP"
fi
echo "========================================"

exit $EXIT_CODE 