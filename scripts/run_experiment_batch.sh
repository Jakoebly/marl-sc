#!/bin/bash

# ============================================================================
# SBATCH directives
# ============================================================================

#SBATCH --job-name=marl-training                # Name of the job
#SBATCH --partition=mit_normal                  # Partition
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Number of tasks per node
#SBATCH --cpus-per-task=3                       # CPU cores per task
#SBATCH --mem=16G                               # Memory allocation
#SBATCH --time=10:00:00                         # Maximum walltime (hh:mm:ss)
#SBATCH --chdir=/home/jakobeh/projects/marl-sc  # Working directory
#SBATCH --output=scripts/logs/%x_%A_%a.out      # Standard output
#SBATCH --error=scripts/logs/%x_%A_%a.err       # Standard error
#SBATCH --array=0-2%3                         # 7 configs x 3 runs = 21 tasks (indices 0-20), max 11 concurrent


# ============================================================================
# Parse arguments
# ============================================================================

# Usage: sbatch run_experiment_batch.sh --name <FolderName>
FOLDER_NAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      [[ $# -ge 2 && "$2" != -* ]] || 
        { echo "ERROR: --name requires a value" >&2; exit 1; }
      FOLDER_NAME="$2"; shift 2 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Print the folder name
if [ -n "$FOLDER_NAME" ]; then
  echo "FOLDER_NAME=${FOLDER_NAME}"
else
  echo "FOLDER_NAME=not specified"
fi


# ============================================================================
# Load modules + env
# ============================================================================

# Load the Python distribution, change to the project directory, 
# and activate the virtual environment
module load miniforge/25.11.0-0                 
cd /home/jakobeh/projects/marl-sc              
source ~/projects/marl-sc/.venv/bin/activate   

# Set the Python path, unbuffer the output, and set the Python hash seed
export PYTHONPATH="/home/jakobeh/projects/marl-sc${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0


# ============================================================================
# Map SLURM_ARRAY_TASK_ID to config and run number
# ============================================================================

# Set the number of runs per config
N_RUNS=3

# Use the ID of the current task to compute the config index and run number
ID=${SLURM_ARRAY_TASK_ID}
CONFIG_IDX=$(( ID / N_RUNS ))
RUN_NUMBER=$(( ID % N_RUNS + 1 ))

# Map the config index to run configs
case $CONFIG_IDX in
    0) ENV_CONFIG="env_symmetric_3WH5SKU" ; CONFIG_NAME="SymmetricEnv" ;;
    *) echo "ERROR: Unknown CONFIG_IDX=$CONFIG_IDX"; exit 1 ;;
esac
echo "Task $ID -> Config #${CONFIG_IDX}, Run #${RUN_NUMBER}"


# ============================================================================
# Create temporary configs with overrides
# ============================================================================

# Set environment and algorithm name
ENV_NAME=$ENV_CONFIG
ALGO_NAME="mappo"

# Create temporary config files 
TEMP_ENV_CONFIG=$(mktemp --suffix=.yaml)
TEMP_ALGO_CONFIG=$(mktemp --suffix=.yaml)

# Override values in the environment and algorithm configs
python - <<PY
import yaml

# Set environment and algorithm names
ENV_NAME = "$ENV_NAME"
ALGO_NAME = "$ALGO_NAME"

# --- Environment config ---
with open(f"config_files/environments/{ENV_NAME}.yaml", "r") as f:
    env_cfg = yaml.safe_load(f)

with open("$TEMP_ENV_CONFIG", "w") as f:
    yaml.safe_dump(env_cfg, f, default_flow_style=False, sort_keys=False)

# --- Algorithm config ---
with open(f"config_files/algorithms/{ALGO_NAME}.yaml", "r") as f:
    algo_cfg = yaml.safe_load(f)

with open("$TEMP_ALGO_CONFIG", "w") as f:
    yaml.safe_dump(algo_cfg, f, default_flow_style=False, sort_keys=False)
PY


# ============================================================================
# Start Ray explicitly (with port allocation)
# ============================================================================

# Source the start_ray.sh script to start Ray
RAY_FALLBACK_MEM_MB=32768
RAY_EXTRA_CLEANUP_PATHS="$TEMP_ENV_CONFIG $TEMP_ALGO_CONFIG"
source scripts/lib/start_ray.sh


# ============================================================================
# Run training + evaluation
# ============================================================================

if [ -n "$FOLDER_NAME" ]; then
  STORAGE_DIR="./experiment_outputs/Runs/${FOLDER_NAME}"
else
  STORAGE_DIR="./experiment_outputs/Runs"
fi
EXPERIMENT_NAME="MAPPO_3WH2SKU_${CONFIG_NAME}_Seed${RUN_NUMBER}"
ROOT_SEED=$(( RUN_NUMBER * 100 ))

# Run training
python src/experiments/run_experiment.py \
    --mode single \
    --env-config "$TEMP_ENV_CONFIG" \
    --algorithm-config "$TEMP_ALGO_CONFIG" \
    --storage-dir "${STORAGE_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --wandb-project marl-sc \
    --root-seed ${ROOT_SEED}

# Run evaluation
python src/experiments/run_experiment.py \
    --mode evaluate \
    --storage-dir "${STORAGE_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --visualize \
    --eval-episodes 100 \
    --root-seed ${ROOT_SEED}
