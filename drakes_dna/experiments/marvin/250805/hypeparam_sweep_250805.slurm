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
END_EXPLORATION_STEP_VALUES=(5 10 15)
END_EXPLORATION_EPOCH_VALUES=(250 500 950)
SEEDS=(42 123 456)

# Fixed hyperparameters
INITIAL_INVERSE_TEMP=0.2
INITIAL_RANDOM_NOISING=0.1

# Calculate which combination this array task should run
# 3x3x3 = 27 combinations total (indices 0-26)
NUM_STEP_VALUES=${#END_EXPLORATION_STEP_VALUES[@]}
NUM_EPOCH_VALUES=${#END_EXPLORATION_EPOCH_VALUES[@]}
NUM_SEEDS=${#SEEDS[@]}

# Convert linear array index to 3D indices
# For 3D array with dimensions [step][epoch][seed]
STEP_IDX=$((SLURM_ARRAY_TASK_ID / (NUM_EPOCH_VALUES * NUM_SEEDS)))
TEMP_IDX=$((SLURM_ARRAY_TASK_ID % (NUM_EPOCH_VALUES * NUM_SEEDS)))
EPOCH_IDX=$((TEMP_IDX / NUM_SEEDS))
SEED_IDX=$((TEMP_IDX % NUM_SEEDS))

# Get the actual parameter values
END_EXPLORATION_STEP=${END_EXPLORATION_STEP_VALUES[$STEP_IDX]}
END_EXPLORATION_EPOCH=${END_EXPLORATION_EPOCH_VALUES[$EPOCH_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Generate run name with timestamp and parameters
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="step${END_EXPLORATION_STEP}_epoch${END_EXPLORATION_EPOCH}_seed${SEED}_${TIMESTAMP}"

echo "========================================"
echo "Hyperparameters for this run:"
echo "  end_exploration_epoch: $END_EXPLORATION_EPOCH"
echo "  end_exploration_step: $END_EXPLORATION_STEP"
echo "  initial_inverse_temp: $INITIAL_INVERSE_TEMP"
echo "  initial_random_noising: $INITIAL_RANDOM_NOISING"
echo "  seed: $SEED"
echo "  run_name: $RUN_NAME"
echo "========================================"

# Run finetuning with the selected hyperparameters
python finetune_reward_bp.py \
    --name "$RUN_NAME" \
    --initial_inverse_temp $INITIAL_INVERSE_TEMP \
    --initial_random_noising $INITIAL_RANDOM_NOISING \
    --end_exploration_step $END_EXPLORATION_STEP \
    --end_exploration_epoch $END_EXPLORATION_EPOCH \
    --seed $SEED

EXIT_CODE=$?

echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Run completed successfully!"
    echo "Run name: $RUN_NAME"
    echo "Parameters: step=$END_EXPLORATION_STEP, epoch=$END_EXPLORATION_EPOCH, seed=$SEED"
else
    echo "ERROR: Run failed with exit code $EXIT_CODE"
    echo "Run name: $RUN_NAME"
    echo "Parameters: step=$END_EXPLORATION_STEP, epoch=$END_EXPLORATION_EPOCH, seed=$SEED"
fi
echo "========================================"

exit $EXIT_CODE 