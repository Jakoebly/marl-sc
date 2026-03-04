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
#SBATCH --array=0-5%5                           # Array for 5 jobs (indices 0-4) with 1 job at once per node


##############################
# Parse arguments
##############################

ARRAY_NAME=${1:?"Usage: sbatch run_experiment_array.sh <ArrayName>"}
echo "ARRAY_NAME=${ARRAY_NAME}"


##############################
# Load modules + env
##############################

module load miniforge/25.11.0-0                 # Load the Python distribution
cd /home/jakobeh/projects/marl-sc               # Change to the project directory
source ~/projects/marl-sc/.venv/bin/activate    # Activate the virtual environment

export PYTHONPATH="/home/jakobeh/projects/marl-sc${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1


##############################
# Map SLURM_ARRAY_TASK_ID -> holding_cost
##############################

# Define possible values for holding cost
HOLDING_COSTS=(0.1 0.3 0.6 1.0 2.0)
N_HOLDING_COSTS=${#HOLDING_COSTS[@]}

# Get the holding cost for this task
ID=${SLURM_ARRAY_TASK_ID}
HOLDING_COST_IDX=$(( ID % N_HOLDING_COSTS ))
HOLDING_COST=${HOLDING_COSTS[$HOLDING_COST_IDX]}

echo "Task $ID -> holding_cost=$HOLDING_COST (index $HOLDING_COST_IDX of ${N_HOLDING_COSTS})"


##############################
# Create temporary config with holding_cost override
##############################

TEMP_CONFIG=$(mktemp --suffix=.yaml)

python - <<PY
import yaml

HOLDING_COST = $HOLDING_COST
TEMP_CONFIG  = "$TEMP_CONFIG"

with open("config_files/environments/env_2EU_1US.yaml", "r") as f:
    config = yaml.safe_load(f)

env = config["environment"]
if "cost_structure" not in env:
    env["cost_structure"] = {}
env["cost_structure"]["holding_cost"] = HOLDING_COST

with open(TEMP_CONFIG, "w") as f:
    yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
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
  rm -f "$TEMP_CONFIG" >/dev/null 2>&1 || true
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
# Run training
##############################

python src/experiments/run_experiment.py \
    --mode single \
    --env-config "$TEMP_CONFIG" \
    --algorithm-config config_files/algorithms/ippo.yaml \
    --output-dir "./experiment_outputs/${ARRAY_NAME}" \
    --experiment-name "IPPO_Single_3WH_3SKUS_Team_PSFalse_HC${HOLDING_COST}" \
    --wandb-project marl-sc \
    --root-seed 42

python src/experiments/run_experiment.py \
  --mode evaluate \
  --output-dir "./experiment_outputs/${ARRAY_NAME}" \
  --experiment-name "IPPO_Single_3WH_3SKUS_Team_PSFalse_HC${HOLDING_COST}" \
  --visualize \
  --root-seed 42


##############################f
# Cleanup
##############################