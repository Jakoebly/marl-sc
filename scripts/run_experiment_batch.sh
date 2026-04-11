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
#SBATCH --array=0-8%9                         # 7 configs x 3 runs = 21 tasks (indices 0-20), max 11 concurrent


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

# # Map the config index to run configs
case $CONFIG_IDX in
    0) ENV_CONFIG="env_pilot_sku_hetero"         ; CONFIG_NAME="SKUHetero" ;;
    1) ENV_CONFIG="env_pilot_sku_hetero_balanced" ; CONFIG_NAME="SKUHeteroBalanced" ;;
    2) ENV_CONFIG="env_pilot_demand_hetero"       ; CONFIG_NAME="DemandHetero" ;;
    *) echo "ERROR: Unknown CONFIG_IDX=$CONFIG_IDX"; exit 1 ;;
esac
echo "Task $ID -> Config #${CONFIG_IDX}, Run #${RUN_NUMBER}"


# ============================================================================
# Create temporary configs with overrides
# ============================================================================

# Set environment and algorithm name
ENV_NAME=$ENV_CONFIG
ALGO_NAME="mappo_best_3WH2SKU"

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
RAY_METRICS_EXPORT_PORT=$((P + 3))
RAY_DASHBOARD_AGENT_GRPC_PORT=$((P + 4))
RAY_DASHBOARD_AGENT_HTTP_PORT=$((P + 5))
RAY_RUNTIME_ENV_AGENT_PORT=$((P + 6))
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

# Disable Ray's application-level OOM killer to prevent the kill-restart
# death spiral; rely on SLURM's cgroup memory enforcement instead
export RAY_memory_monitor_refresh_ms=0
export PYTHONWARNINGS="ignore::DeprecationWarning"

# Give Ray more time to start up to avoid premature termination due to heavy loads
export RAY_raylet_start_wait_time_s=300
sleep $(( RANDOM % 45 ))

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
  --metrics-export-port="${RAY_METRICS_EXPORT_PORT}" \
  --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
  --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_HTTP_PORT}" \
  --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}" \
  || { echo "ERROR: ray start failed"; exit 1; }
echo "Ray started successfully"


# ============================================================================
# Run training + evaluation
# ============================================================================

if [ -n "$FOLDER_NAME" ]; then
  STORAGE_DIR="./experiment_outputs/Runs/${FOLDER_NAME}"
else
  STORAGE_DIR="./experiment_outputs/Runs/Pilot"
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
