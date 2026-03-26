#!/bin/bash

##############################
# SBATCH directives
##############################

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
#SBATCH --array=0-11%12                         # 7 configs x 3 runs = 21 tasks (indices 0-20), max 11 concurrent


##############################
# Parse arguments
##############################

# Usage: sbatch run_experiment_batch.sh [ArrayName]
ARRAY_NAME=${1:-""}

# Print the array name
if [ -n "$ARRAY_NAME" ]; then
  echo "ARRAY_NAME=${ARRAY_NAME}"
else
  echo "ARRAY_NAME=not specified"
fi


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
# Map SLURM_ARRAY_TASK_ID to config and run number
##############################

# Set the number of runs per config
N_RUNS=3

# Use the ID of the current task to compute the config index and run number
ID=${SLURM_ARRAY_TASK_ID}
CONFIG_IDX=$(( ID / N_RUNS ))
RUN_NUMBER=$(( ID % N_RUNS + 1 ))

# Map the config index to run configs
case $CONFIG_IDX in
    0) BETA="None" ;;
    1) BETA=0.3 ;;
    2) BETA=0.5 ;;
    3) BETA=0.7 ;;
    *) echo "ERROR: Unknown CONFIG_IDX=$CONFIG_IDX"; exit 1 ;;
esac
echo "Task $ID -> Config #${CONFIG_IDX}, Run #${RUN_NUMBER}"
echo "  beta=$BETA"


##############################
# Create temporary configs with overrides
##############################

# Set environment and algorithm name
ENV_NAME="env_simplified_symmetric"
ALGO_NAME="ippo"

# Create temporary config files 
TEMP_ENV_CONFIG=$(mktemp --suffix=.yaml)
TEMP_ALGO_CONFIG=$(mktemp --suffix=.yaml)

# Override values in the environment and algorithm configs
python - <<PY
import yaml

# Set environment and algorithm names
ENV_NAME = "$ENV_NAME"
ALGO_NAME = "$ALGO_NAME"

_raw_beta = "$BETA".strip()
if _raw_beta.lower() in ("none", "null", ""):
    beta = None
else:
    beta = float(_raw_beta)

# --- Environment config ---
with open(f"config_files/environments/{ENV_NAME}.yaml", "r") as f:
    env_cfg = yaml.safe_load(f)

with open("$TEMP_ENV_CONFIG", "w") as f:
    yaml.safe_dump(env_cfg, f, default_flow_style=False, sort_keys=False)

# --- Algorithm config ---
with open(f"config_files/algorithms/{ALGO_NAME}.yaml", "r") as f:
    algo_cfg = yaml.safe_load(f)

algo_cfg["algorithm"]["algorithm_specific"]["hysteretic_beta"] = beta

with open("$TEMP_ALGO_CONFIG", "w") as f:
    yaml.safe_dump(algo_cfg, f, default_flow_style=False, sort_keys=False)
PY


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
  echo "Fix: reduce array concurrency per node (e.g., use --exclusive) or lower BASE_PORT / change policy."
  exit 1
fi

# Define the job width (i.e., number of ports per job)
ARRAY_WIDTH=$((N_TASKS * BLOCK_SIZE)) # total number of ports for all jobs in the array
ARRAY_SLOTS=$((AVAILABLE / ARRAY_WIDTH)) # number of arrays with the same size as the current array that can fit in the available port range

# Sanity check if the number of array slots is at least 1
if [ $ARRAY_SLOTS -lt 1 ]; then
  echo "ERROR: Not enough port space for even one array slot (this should not happen if BLOCK_SIZE check passed)."
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
if [ -n "$ARRAY_NAME" ]; then
  STORAGE_DIR="./experiment_outputs/Phase1/${ARRAY_NAME}"
else
  STORAGE_DIR="./experiment_outputs/Phase1/WorkingConfig_Phase1.10"  
fi
EXPERIMENT_NAME="IPPO_3WH2SKU_SimplifiedEnv_Beta${BETA}_Run${RUN_NUMBER}"

# Run training
python src/experiments/run_experiment.py \
    --mode single \
    --env-config "$TEMP_ENV_CONFIG" \
    --algorithm-config "$TEMP_ALGO_CONFIG" \
    --storage-dir "${STORAGE_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --wandb-project marl-sc \
    --root-seed 42

# Run evaluation
python src/experiments/run_experiment.py \
    --mode evaluate \
    --storage-dir "${STORAGE_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --visualize \
    --root-seed 42


##############################
# Cleanup
##############################