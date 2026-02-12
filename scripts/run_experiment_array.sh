#!/bin/bash

##############################
# SBATCH directives
##############################

#SBATCH --job-name=marl-training                # Name of the job
#SBATCH --partition=mit_normal                  # Partition
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Number of tasks per node 
#SBATCH --cpus-per-task=5                       # CPU cores per task
#SBATCH --mem=32G                               # Memory allocation
#SBATCH --time=12:00:00                         # Maximum walltime (hh:mm:ss)
#SBATCH --chdir=/home/jakobeh/projects/marl-sc  # Working directory
#SBATCH --output=scripts/logs/%x_%A_%a.out      # Standard output
#SBATCH --error=scripts/logs/%x_%A_%a.err       # Standard error
#SBATCH --array=0-24%5                          # Array for 25 jobs (indices 0-24) with 1 job at once per node


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
# Map SLURM_ARRAY_TASK_ID -> (n_warehouses, n_skus)
##############################

# Define possible values for warehouses and SKUs
WAREHOUSE_VALUES=(3 5 10 15)
SKU_VALUES=(3 5 10 20 50 100)

# Compute the number of possible tasks
N_WHVALS=${#WAREHOUSE_VALUES[@]}
N_SKUVALS=${#SKU_VALUES[@]}
N_TASKS=$((N_WHVALS * N_SKUVALS))

# Get the warehouse and SKU values for this task
ID=${SLURM_ARRAY_TASK_ID}
SKU_IDX=$(( ID / N_WHVALS ))
WH_IDX=$(( ID % N_WHVALS ))
N_SKUS=${SKU_VALUES[$SKU_IDX]}
N_WAREHOUSES=${WAREHOUSE_VALUES[$WH_IDX]}
N_REGIONS=$N_WAREHOUSES

echo "Task $ID -> n_warehouses=$N_WAREHOUSES, n_skus=$N_SKUS (n_regions=$N_REGIONS)"


##############################
# Create temporary config with dimension overrides
##############################

TEMP_CONFIG=$(mktemp --suffix=.yaml)

python - <<PY
import yaml

N_WAREHOUSES = $N_WAREHOUSES
N_SKUS       = $N_SKUS
N_REGIONS    = $N_REGIONS
TEMP_CONFIG  = "$TEMP_CONFIG"

with open("config_files/environments/base_env.yaml", "r") as f:
    config = yaml.safe_load(f)

env = config["environment"]
env["n_warehouses"] = N_WAREHOUSES
env["n_skus"]       = N_SKUS
env["n_regions"]    = N_REGIONS

with open(TEMP_CONFIG, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
PY


##############################
# Start Ray explicitly
##############################

# Make Ray not accidentally attach somewhere else
unset RAY_ADDRESS

# Get the number of CPUs from Slurm
CPUS=${SLURM_CPUS_PER_TASK:-1}
echo "Starting Ray with ${CPUS} CPUs"

# Allocate a unique port block per task
BASE_PORT=30000
BLOCK_SIZE=200   # room for worker ports
JOB_OFFSET=$(( (SLURM_JOB_ID % 100) * BLOCK_SIZE ))
TASK_OFFSET=$(( SLURM_ARRAY_TASK_ID * BLOCK_SIZE ))
P=$(( BASE_PORT + JOB_OFFSET + TASK_OFFSET ))

RAY_GCS_PORT=$((P + 0))
RAY_NODE_MANAGER_PORT=$((P + 1))
RAY_OBJECT_MANAGER_PORT=$((P + 2))
RAY_MIN_WORKER_PORT=$((P + 20))
RAY_MAX_WORKER_PORT=$((P + 199))

# Per-task Ray temp dir to avoid session/state collisions on shared nodes
RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RAY_TMPDIR"

# Cleanup function for Ray temp dir
cleanup() {
  rm -f "$TEMP_CONFIG" >/dev/null 2>&1 || true
  rm -rf "$RAY_TMPDIR" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Force this task's driver to connect to this task's head
export RAY_ADDRESS="127.0.0.1:${RAY_GCS_PORT}"
echo "Ray head:        ${RAY_ADDRESS}"
echo "Node mgr port:   ${RAY_NODE_MANAGER_PORT}"
echo "Object mgr port: ${RAY_OBJECT_MANAGER_PORT}"
echo "Worker ports:    ${RAY_MIN_WORKER_PORT}-${RAY_MAX_WORKER_PORT}"
echo "Ray temp dir:    ${RAY_TMPDIR}"

# Start Ray explicitly with ONLY those CPUs
ray start --head \
  --port="${RAY_GCS_PORT}" \
  --node-manager-port="${RAY_NODE_MANAGER_PORT}" \
  --object-manager-port="${RAY_OBJECT_MANAGER_PORT}" \
  --min-worker-port="${RAY_MIN_WORKER_PORT}" \
  --max-worker-port="${RAY_MAX_WORKER_PORT}" \
  --num-cpus="${CPUS}" \
  --temp-dir="${RAY_TMPDIR}" \
  --include-dashboard=false \
  --disable-usage-stats


##############################
# Run training
##############################

python src/experiments/run_experiment.py \
    --mode single \
    --env-config "$TEMP_CONFIG" \
    --algorithm-config config_files/algorithms/mappo.yaml \
    --output-dir "./experiment_outputs/${ARRAY_NAME}" \
    --wandb-project marl-sc \
    --root-seed 42


##############################
# Cleanup
##############################