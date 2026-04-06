#!/bin/bash

##############################
# SBATCH directives
##############################

#SBATCH --job-name=marl-seed-eval              # Name of the job
#SBATCH --partition=mit_normal                 # Partition
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --ntasks-per-node=1                    # Number of tasks per node
#SBATCH --cpus-per-task=3                      # CPU cores per task
#SBATCH --mem=16G                              # Memory allocation
#SBATCH --time=12:00:00                        # Maximum walltime (hh:mm:ss)
#SBATCH --chdir=/home/jakobeh/projects/marl-sc # Working directory
#SBATCH --output=scripts/logs/%x_%A_%a.out     # Standard output
#SBATCH --error=scripts/logs/%x_%A_%a.err      # Standard error
#SBATCH --array=0-11%12                        # 7 configs x 3 runs = 21 tasks (indices 0-20), max 11 concurrent


##############################
# Parse arguments
##############################

# Usage: sbatch run_seed_evaluation.sh <TuneName> [n_seeds] [num_iterations]
TUNE_NAME=${1:?"ERROR: TuneName is required as first argument"}
N_SEEDS=${2:-3}
NUM_ITERATIONS=${3:-""}

# Set the experiment path
EXPERIMENT_PATH="experiment_outputs/Tuning/${TUNE_NAME}"
if [ ! -d "$EXPERIMENT_PATH" ]; then
  echo "ERROR: Experiment directory not found: ${EXPERIMENT_PATH}"
  exit 1
fi

echo "TUNE_NAME=${TUNE_NAME}"
echo "EXPERIMENT_PATH=${EXPERIMENT_PATH}"
echo "N_SEEDS=${N_SEEDS}"
echo "NUM_ITERATIONS=${NUM_ITERATIONS:-'(use trial default)'}"


##############################
# Load modules + env
##############################

# Load the Python distribution, change to the project directory, 
# and activate the virtual environment
module load miniforge/25.11.0-0
cd /home/jakobeh/projects/marl-sc
source ~/projects/marl-sc/.venv/bin/activate

# Set the Python path, unbuffer the output, and set the Python hash seed
export PYTHONPATH="/home/jakobeh/projects/marl-sc${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0


##############################
# Map SLURM_ARRAY_TASK_ID to trial config and seed
##############################

# Use the ID of the current task to compute the config index and seed number
ID=${SLURM_ARRAY_TASK_ID}
CONFIG_IDX=$(( ID / N_SEEDS ))
SEED_NUMBER=$(( ID % N_SEEDS + 1 ))

# Read trial path from best_trial_results.yaml via inline Python
read -r TRIAL_PATH SHORT_ID < <(python - <<PY
import yaml, sys

yaml_path = "${EXPERIMENT_PATH}/best_trial_results.yaml"
with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

trials = data.get("top_k_trials", [])
idx = ${CONFIG_IDX}
if idx >= len(trials):
    print(
      f"ERROR: CONFIG_IDX={idx} exceeds top_k_trials count ({len(trials)})", file=sys.stderr
    )
    sys.exit(1)

print(trials[idx]["trial_path"], trials[idx]["short_id"])
PY
)

if [ $? -ne 0 ] || [ -z "$TRIAL_PATH" ]; then
  echo "ERROR: Failed to resolve trial path for CONFIG_IDX=${CONFIG_IDX}"
  exit 1
fi

echo "Task $ID -> Trial #${CONFIG_IDX} (${SHORT_ID}), Seed #${SEED_NUMBER}"
echo "  Trial path: ${TRIAL_PATH}"


##############################
# Create temporary configs with overrides
##############################

# Create temporary config files 
TEMP_ENV_CONFIG=$(mktemp --suffix=.yaml)
TEMP_ALGO_CONFIG=$(mktemp --suffix=.yaml)

# Load the trial's environment and algorithm configs and override the num_iterations
python - <<PY
import yaml, sys
from pathlib import Path

trial_path = Path("${TRIAL_PATH}")
env_cfg_path = trial_path / "env_config.yaml"
algo_cfg_path = trial_path / "algorithm_config.yaml"

if not env_cfg_path.exists():
    print(f"ERROR: {env_cfg_path} not found", file=sys.stderr)
    sys.exit(1)
if not algo_cfg_path.exists():
    print(f"ERROR: {algo_cfg_path} not found", file=sys.stderr)
    sys.exit(1)

# Load environment config (no modifications needed)
with open(env_cfg_path, "r") as f:
    env_cfg = yaml.safe_load(f)
with open("${TEMP_ENV_CONFIG}", "w") as f:
    yaml.safe_dump(env_cfg, f, default_flow_style=False, sort_keys=False)

# Load algorithm config and optionally override num_iterations
with open(algo_cfg_path, "r") as f:
    algo_cfg = yaml.safe_load(f)

num_iterations = "${NUM_ITERATIONS}".strip()
if num_iterations:
    algo_cfg["algorithm"]["shared"]["num_iterations"] = int(num_iterations)

with open("${TEMP_ALGO_CONFIG}", "w") as f:
    yaml.safe_dump(algo_cfg, f, default_flow_style=False, sort_keys=False)
PY

if [ $? -ne 0 ]; then
  echo "ERROR: Failed to create temporary config files"
  rm -f "$TEMP_ENV_CONFIG" "$TEMP_ALGO_CONFIG"
  exit 1
fi


##############################
# Start Ray explicitly (with port allocation)
##############################

# Make Ray not accidentally attach somewhere else
unset RAY_ADDRESS

# Get the number of CPUs from Slurm
CPUS=${SLURM_CPUS_PER_TASK:-1}

# Get the memory from Slurm
RAY_MEMORY_BYTES=$(( (${SLURM_MEM_PER_NODE:-32768} - 2048) * 1024 * 1024 ))

# Determine array size and number of tasks
MAX_TASK_ID=${SLURM_ARRAY_TASK_MAX:-${SLURM_ARRAY_TASK_ID}}
N_TASKS=$((MAX_TASK_ID + 1))

# Define the port range
BASE_PORT=20000
MAX_PORT=65535
AVAILABLE=$((MAX_PORT - BASE_PORT + 1))

# Define the preferred block size (i.e., number of ports) per task
PREFERRED_BLOCK_SIZE=200

# Reserve the first 20 ports in the block for fixed components and leave the rest for workers
RESERVED_WITHIN_BLOCK=20

# Define the minimum block size such that the worker range is non-empty
MIN_BLOCK_SIZE=$((RESERVED_WITHIN_BLOCK + 1))

# Cap the preferred block size to the maximum possible if necessary
BLOCK_SIZE=$PREFERRED_BLOCK_SIZE
MAX_BLOCK_SIZE=$((AVAILABLE / N_TASKS))
if [ $MAX_BLOCK_SIZE -lt $BLOCK_SIZE ]; then
  BLOCK_SIZE=$MAX_BLOCK_SIZE
fi

# Sanity check if the block size is too small
if [ $BLOCK_SIZE -lt $MIN_BLOCK_SIZE ]; then
  echo "ERROR: Array too large to allocate ports safely."
  echo "N_TASKS=${N_TASKS}, AVAILABLE_PORTS=${AVAILABLE}, computed BLOCK_SIZE=${BLOCK_SIZE} (< ${MIN_BLOCK_SIZE})"
  exit 1
fi

# Define the job width (i.e., number of ports per job)
ARRAY_WIDTH=$((N_TASKS * BLOCK_SIZE)) # total number of ports for all jobs in the array
ARRAY_SLOTS=$((AVAILABLE / ARRAY_WIDTH)) # number of arrays with the same size as the current array that can fit in the available port range

# Sanity check if the number of array slots is at least 1
if [ $ARRAY_SLOTS -lt 1 ]; then
  echo "ERROR: Not enough port space for even one array slot."
  exit 1
fi

# Get the array slot for the current array
ARRAY_SLOT=$((SLURM_JOB_ID % ARRAY_SLOTS))

# Compute the first port for the current task
TASK_ID=${SLURM_ARRAY_TASK_ID}
P=$((BASE_PORT + ARRAY_SLOT * ARRAY_WIDTH + TASK_ID * BLOCK_SIZE))

# Set the ports for the Ray components of the current task
RAY_GCS_PORT=$((P + 0))
RAY_NODE_MANAGER_PORT=$((P + 1))
RAY_OBJECT_MANAGER_PORT=$((P + 2))
RAY_MIN_WORKER_PORT=$((P + RESERVED_WITHIN_BLOCK))
RAY_MAX_WORKER_PORT=$((P + BLOCK_SIZE - 1))

# Sanity check if the maximum worker port is within the available port range
if [ $RAY_MAX_WORKER_PORT -gt $MAX_PORT ]; then
  echo "ERROR: Port calculation overflowed: ${RAY_MAX_WORKER_PORT} > ${MAX_PORT}"
  exit 1
fi

# Per-task Ray temp dir to avoid session/state collisions on shared nodes
RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RAY_TMPDIR"

# Cleanup function for temp config files
cleanup() {
  rm -f "$TEMP_ENV_CONFIG" "$TEMP_ALGO_CONFIG" >/dev/null 2>&1 || true
  rm -rf "$RAY_TMPDIR" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Force the current task's driver to connect to the current task's head
export RAY_ADDRESS="127.0.0.1:${RAY_GCS_PORT}"

# Start Ray explicitly with ports and number of CPUs
ray start --head \
  --port="${RAY_GCS_PORT}" \
  --node-manager-port="${RAY_NODE_MANAGER_PORT}" \
  --object-manager-port="${RAY_OBJECT_MANAGER_PORT}" \
  --min-worker-port="${RAY_MIN_WORKER_PORT}" \
  --max-worker-port="${RAY_MAX_WORKER_PORT}" \
  --num-cpus="${CPUS}" \
  --memory="${RAY_MEMORY_BYTES}" \
  --temp-dir="${RAY_TMPDIR}" \
  --include-dashboard=false \
  --disable-usage-stats \
  || { echo "ERROR: ray start failed"; exit 1; }
echo "Ray started successfully"


##############################
# Run training + evaluation
##############################

# Set output directory and experiment name
STORAGE_DIR="${EXPERIMENT_PATH}/seed_evaluation"
EXPERIMENT_NAME="${SHORT_ID}_seed${SEED_NUMBER}"

# Set the root seed as a deterministic but well-separated number (e.g., 100, 200, 300, ...)
ROOT_SEED=$(( SEED_NUMBER * 100 ))

# Run training
python src/experiments/run_experiment.py \
    --mode single \
    --env-config "$TEMP_ENV_CONFIG" \
    --algorithm-config "$TEMP_ALGO_CONFIG" \
    --storage-dir "${STORAGE_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --wandb-project marl-sc \
    --root-seed ${ROOT_SEED}

# Run evaluation on the trained checkpoint
python src/experiments/run_experiment.py \
    --mode evaluate \
    --storage-dir "${STORAGE_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --visualize \
    --eval-episodes 100 \
    --root-seed ${ROOT_SEED}


##############################
# Cleanup
##############################
