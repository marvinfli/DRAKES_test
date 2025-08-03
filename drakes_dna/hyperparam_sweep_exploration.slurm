#!/bin/bash
#SBATCH --job-name=exploration_hyperparam_sweep
#SBATCH --account=sitanc_lab
#SBATCH --partition gpu,seas_gpu
#SBATCH --gres gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
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
END_EXPLORATION_EPOCH_VALUES=(100 300 500)
INITIAL_INVERSE_TEMP_VALUES=(0.01 0.2)
INITIAL_RANDOM_NOISING_VALUES=(0.1 0.5)

# Calculate which combination this array task should run
# 3x2x2 = 12 combinations total (indices 0-11)
NUM_EPOCH_VALUES=${#END_EXPLORATION_EPOCH_VALUES[@]}
NUM_TEMP_VALUES=${#INITIAL_INVERSE_TEMP_VALUES[@]}
NUM_NOISE_VALUES=${#INITIAL_RANDOM_NOISING_VALUES[@]}

# Convert linear array index to 3D indices
EPOCH_IDX=$((SLURM_ARRAY_TASK_ID / (NUM_TEMP_VALUES * NUM_NOISE_VALUES)))
TEMP_IDX=$(((SLURM_ARRAY_TASK_ID % (NUM_TEMP_VALUES * NUM_NOISE_VALUES)) / NUM_NOISE_VALUES))
NOISE_IDX=$((SLURM_ARRAY_TASK_ID % NUM_NOISE_VALUES))

# Get the actual parameter values
END_EXPLORATION_EPOCH=${END_EXPLORATION_EPOCH_VALUES[$EPOCH_IDX]}
INITIAL_INVERSE_TEMP=${INITIAL_INVERSE_TEMP_VALUES[$TEMP_IDX]}
INITIAL_RANDOM_NOISING=${INITIAL_RANDOM_NOISING_VALUES[$NOISE_IDX]}

# Generate run name with timestamp and parameters
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="sweep_epoch${END_EXPLORATION_EPOCH}_temp${INITIAL_INVERSE_TEMP}_noise${INITIAL_RANDOM_NOISING}_${TIMESTAMP}"

echo "========================================"
echo "Hyperparameters for this run:"
echo "  end_exploration_epoch: $END_EXPLORATION_EPOCH"
echo "  initial_inverse_temp: $INITIAL_INVERSE_TEMP"
echo "  initial_random_noising: $INITIAL_RANDOM_NOISING"
echo "  run_name: $RUN_NAME"
echo "========================================"

# Run finetuning with the selected hyperparameters
python finetune_reward_bp.py \
    --name "$RUN_NAME" \
    --initial_inverse_temp $INITIAL_INVERSE_TEMP \
    --initial_random_noising $INITIAL_RANDOM_NOISING \
    --end_exploration_step 10 \
    --end_exploration_epoch $END_EXPLORATION_EPOCH

EXIT_CODE=$?

echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Run completed successfully!"
    echo "Run name: $RUN_NAME"
    echo "Parameters: epoch=$END_EXPLORATION_EPOCH, temp=$INITIAL_INVERSE_TEMP, noise=$INITIAL_RANDOM_NOISING"
else
    echo "ERROR: Run failed with exit code $EXIT_CODE"
    echo "Run name: $RUN_NAME"
    echo "Parameters: epoch=$END_EXPLORATION_EPOCH, temp=$INITIAL_INVERSE_TEMP, noise=$INITIAL_RANDOM_NOISING"
fi
echo "========================================"

exit $EXIT_CODE 