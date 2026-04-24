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
#SBATCH --array=0-29%30                         # 7 configs x 3 runs = 21 tasks (indices 0-20), max 11 concurrent
#SBATCH --exclude=node1620,node1621,node1622,node1623,node1624,node1625,node2704,node2705


# ============================================================================
# Parse arguments
# ============================================================================

# Usage: sbatch run_experiment_batch.sh --name <FolderName> [--wandb]
#
# Options:
#   --name <FolderName>   Subfolder under experiment_outputs/Runs (optional)
#   --wandb               Enable WandB logging with project "marl-sc"
#                         (default: off; no wandb args passed to Python)

WANDB_PROJECT_NAME="marl-sc"

FOLDER_NAME=""
USE_WANDB=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      [[ $# -ge 2 && "$2" != -* ]] || 
        { echo "ERROR: --name requires a value" >&2; exit 1; }
      FOLDER_NAME="$2"; shift 2 ;;
    --wandb)
      USE_WANDB=true; shift ;;
    *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Print the folder name
if [ -n "$FOLDER_NAME" ]; then
  echo "FOLDER_NAME=${FOLDER_NAME}"
else
  echo "FOLDER_NAME=not specified"
fi
echo "USE_WANDB=${USE_WANDB}"


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

# Pin thread pools to the cores SLURM actually granted us.
# Default Python/PyTorch detects *physical* cores on the whole node and
# oversubscribes with other tasks sharing the node, potentially
# resulting in slow runs.
CPUS_THREAD_PIN=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="${CPUS_THREAD_PIN}"
export MKL_NUM_THREADS="${CPUS_THREAD_PIN}"
export OPENBLAS_NUM_THREADS="${CPUS_THREAD_PIN}"
export NUMEXPR_NUM_THREADS="${CPUS_THREAD_PIN}"
export TORCH_NUM_THREADS="${CPUS_THREAD_PIN}"
export RAY_DEFAULT_OMP_NUM_THREADS="${CPUS_THREAD_PIN}"


# ============================================================================
# Map SLURM_ARRAY_TASK_ID to config and run number
# ============================================================================

# Set the number of runs per config
N_RUNS=30

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

# If a prior (killed) run for this (config, run) left a periodic
# checkpoint_<N> behind, resume from the largest N instead of starting
# from iteration 1. The runner truncates training_metrics.yaml to match.
RESUME_FROM=""
RUN_DIR="${STORAGE_DIR}/${EXPERIMENT_NAME}"
if [ -d "$RUN_DIR" ]; then
  LATEST_CHKPT=$(ls -d "${RUN_DIR}"/checkpoint_[0-9]* 2>/dev/null \
    | awk -F'checkpoint_' '{print $2"|"$0}' \
    | sort -n \
    | tail -n1 \
    | cut -d'|' -f2-)
  if [ -n "$LATEST_CHKPT" ] && [ -d "$LATEST_CHKPT" ]; then
    RESUME_FROM="$LATEST_CHKPT"
    echo "[RESUME] Found existing periodic checkpoint: ${RESUME_FROM}"
  fi
fi

# Build training command (with optional --resume-from and --wandb-*)
TRAIN_CMD=(python src/experiments/run_experiment.py
    --mode single
    --env-config "$TEMP_ENV_CONFIG"
    --algorithm-config "$TEMP_ALGO_CONFIG"
    --storage-dir "${STORAGE_DIR}"
    --experiment-name "${EXPERIMENT_NAME}"
    --root-seed ${ROOT_SEED})
if [ -n "$RESUME_FROM" ]; then
  TRAIN_CMD+=(--resume-from "$RESUME_FROM")
fi
if [ "$USE_WANDB" = true ]; then
  TRAIN_CMD+=(--wandb-project "$WANDB_PROJECT_NAME" --wandb-name "$EXPERIMENT_NAME")
fi

"${TRAIN_CMD[@]}"

# Run evaluation
python src/experiments/run_experiment.py \
    --mode evaluate \
    --storage-dir "${STORAGE_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --visualize \
    --eval-episodes 100 \
    --root-seed ${ROOT_SEED}
