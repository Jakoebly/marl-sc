#!/bin/bash

##############################
# SBATCH directives
##############################

#SBATCH --job-name=marl-training                # Name of the job
#SBATCH --partition=mit_normal                  # Partition
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Number of tasks per node 
#SBATCH --cpus-per-task=8                       # CPU cores per task
#SBATCH --mem=32G                               # Memory allocation
#SBATCH --time=12:00:00                         # Maximum walltime (hh:mm:ss)
#SBATCH --chdir=/home/jakobeh/projects/marl-sc  # Working directory
#SBATCH --output=scripts/logs/%x_%A_%a.out      # Standard output
#SBATCH --error=scripts/logs/%x_%A_%a.err       # Standard error
#SBATCH --array=0-2%3                          # Array for 10 jobs (indices 0-9) with 1 job at once per node


##############################
# Parse arguments
##############################

ARRAY_NAME=${1:-}
if [ -n "$ARRAY_NAME" ]; then
  echo "ARRAY_NAME=${ARRAY_NAME}"
else
  echo "ARRAY_NAME=not specified"
fi


##############################
# Load modules + env
##############################

module load miniforge/25.11.0-0                 # Load the Python distribution
cd /home/jakobeh/projects/marl-sc               # Change to the project directory
source ~/projects/marl-sc/.venv/bin/activate    # Activate the virtual environment

export PYTHONPATH="/home/jakobeh/projects/marl-sc${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1


##############################
# Map SLURM_ARRAY_TASK_ID -> run parameters
##############################

# #### INDEX ARITHMETICS ####
# # Define possible values for max quantity and entropy coefficient
# MAX_QTYS=(20 30 40)
# LEARNING_RATES=(0.0003 0.001)
# VF_CLIP_PARAMS=(30000 100000 10000000000)

# # Get the number of possible values for max quantity and entropy coefficient
# N_QTYS=${#MAX_QTYS[@]}
# N_LRS=${#LEARNING_RATES[@]}
# N_VFCLIPS=${#VF_CLIP_PARAMS[@]}

# # Get the ID of the current task for indexing the MAX_QTYS and ENTROPY_COEFFS arrays
# ID=${SLURM_ARRAY_TASK_ID}
# QTY_IDX=$(( ID % N_QTYS ))
# LR_IDX=$(( (ID / N_QTYS) % N_LRS ))
# VFC_IDX=$(( (ID / ( N_QTYS*N_LRS )) ))

# # Get the max quantity and entropy coefficient for this task
# MAX_QTY=${MAX_QTYS[$QTY_IDX]}
# LEARNING_RATE=${LEARNING_RATES[$LR_IDX]}
# VF_CLIP_PARAM=${VF_CLIP_PARAMS[$VFC_IDX]}

# echo "Task $ID -> max_qty=$MAX_QTY, learning_rate=$LEARNING_RATE, vf_clip_param=$VF_CLIP_PARAM"

#### CASE LOOKUP ####

ID=${SLURM_ARRAY_TASK_ID}

case $ID in
    0) HIDDEN_SIZES=[64] ;;
    1) HIDDEN_SIZES=[128] ;;
    2) HIDDEN_SIZES=[] ;;
esac

echo "Task $ID -> hidden_sizes=$HIDDEN_SIZES"

##############################
# Create temporary config with max quantity and entropy coefficient overrides
##############################

# Set environment and algorithm name
ENV_NAME="env_simplified_single"
ALGO_NAME="ippo"

# Create temporary config files
TEMP_ENV_CONFIG=$(mktemp --suffix=.yaml)
TEMP_ALGO_CONFIG=$(mktemp --suffix=.yaml)

python - <<PY
import yaml

# Set environment and algorithm names
ENV_NAME = "$ENV_NAME"
ALGO_NAME = "$ALGO_NAME"

# Set run parameters
hidden_sizes = $HIDDEN_SIZES

# --- Algorithm config ---
with open(f"config_files/algorithms/{ALGO_NAME}.yaml", "r") as f:
    algo_cfg = yaml.safe_load(f)

algo_cfg["algorithm"]["algorithm_specific"]["networks"]["actor"]["config"]["hidden_sizes"] = hidden_sizes
algo_cfg["algorithm"]["algorithm_specific"]["networks"]["critic"]["config"]["hidden_sizes"] = hidden_sizes

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

# Cleanup function for Ray temp dir
cleanup() {
  rm -f "$TEMP_ENV_CONFIG" "$TEMP_ALGO_CONFIG" >/dev/null 2>&1 || true
  rm -rf "$RAY_TMPDIR" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Force the current task's driver to connect to the current task's head
export RAY_ADDRESS="127.0.0.1:${RAY_GCS_PORT}"

# Print the current task's ports and Ray components
echo "Array tasks:     ${N_TASKS} (max id ${MAX_TASK_ID})"
echo "Port base/range: ${BASE_PORT}-${MAX_PORT} (available ${AVAILABLE})"
echo "Block size:      ${BLOCK_SIZE} ports per task (workers: $((BLOCK_SIZE - RESERVED_WITHIN_BLOCK)))"
echo "Job slots:       ${ARRAY_SLOTS} (using slot ${ARRAY_SLOT})"
echo "Ray head:        ${RAY_ADDRESS}"
echo "Node mgr port:   ${RAY_NODE_MANAGER_PORT}"
echo "Object mgr port: ${RAY_OBJECT_MANAGER_PORT}"
echo "Worker ports:    ${RAY_MIN_WORKER_PORT}-${RAY_MAX_WORKER_PORT}"
echo "Ray temp dir:    ${RAY_TMPDIR}"

# Start Ray explicitly with ports and number of CPUs
ray start --head \
  --port="${RAY_GCS_PORT}" \
  --node-manager-port="${RAY_NODE_MANAGER_PORT}" \
  --object-manager-port="${RAY_OBJECT_MANAGER_PORT}" \
  --min-worker-port="${RAY_MIN_WORKER_PORT}" \
  --max-worker-port="${RAY_MAX_WORKER_PORT}" \
  --num-cpus="${CPUS}" \
  --temp-dir="${RAY_TMPDIR}" \
  --include-dashboard=false \
  --disable-usage-stats \
  || { echo "ERROR: ray start failed"; exit 1; }


##############################
# Run training + evaluation
##############################

# Set output directory and experiment name
if [ -n "$ARRAY_NAME" ]; then
  OUTPUT_DIR="./experiment_outputs/${ARRAY_NAME}"
else
  OUTPUT_DIR="./experiment_outputs/WorkingConfig_Phase1.4"  
fi

if [ "$HIDDEN_SIZES" = "[64]" ]; then
  HIDDEN_SIZES_LABEL="64"
elif [ "$HIDDEN_SIZES" = "[128]" ]; then
  HIDDEN_SIZES_LABEL="128"
else
  HIDDEN_SIZES_LABEL="None"
fi

EXPERIMENT_NAME="IPPO_Single_1WH_1SKUS_Agent_PSFalse_LogStdFloor-2_NN${HIDDEN_SIZES_LABEL}"


python src/experiments/run_experiment.py \
    --mode single \
    --env-config "$TEMP_ENV_CONFIG" \
    --algorithm-config "$TEMP_ALGO_CONFIG" \
    --output-dir "${OUTPUT_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --wandb-project marl-sc \
    --root-seed 42

python src/experiments/run_experiment.py \
    --mode evaluate \
    --output-dir "${OUTPUT_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --visualize \
    --root-seed 42


##############################
# Cleanup
##############################