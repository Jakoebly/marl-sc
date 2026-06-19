#!/bin/bash

# ============================================================================
# SBATCH directives
# ============================================================================

#SBATCH --job-name=marl-eval                                # Name of the job
##SBATCH --partition=mit_normal                              # Partition MIT      <-- ADJUST THIS!
#SBATCH --clusters=cm4                                      # Cluster LRZ        <-- ADJUST THIS!
#SBATCH --partition=cm4_tiny                                # Partition LRZ      <-- ADJUST THIS!
#SBATCH --nodes=1                                           # Number of nodes
#SBATCH --ntasks-per-node=1                                 # Number of tasks per node 
#SBATCH --cpus-per-task=8                                   # CPU cores per task
#SBATCH --mem=32G                                           # Memory allocation
#SBATCH --time=12:00:00                                     # Maximum walltime 
##SBATCH --chdir=/home/jakobeh/projects/marl-sc             # Working directory MIT Jakobeh   <-- ADJUST THIS!
##SBATCH --chdir=/home/lumechs/projects/marl-sc             # Working directory MIT Lumechs   <-- ADJUST THIS!
#SBATCH --chdir=/dss/dsshome1/07/go56luf2/projects/marl-sc  # Working directory LRZ Go56luf2  <-- ADJUST THIS!
#SBATCH --output=scripts/logs/%x_%j.out                     # Standard output
#SBATCH --error=scripts/logs/%x_%j.err                      # Standard error

# Set the cluster (one of "lrz", "mit-jakobeh", "mit-lumechs")
CLUSTER="lrz"


# ============================================================================
# Parse arguments
# ============================================================================

# Usage: 
#   sbatch run_evaluation.sh --name <ExperimentName|TrainablePrefix> [--checkpoint-number N]
#
# Options:
#   --name <ExperimentName|TrainablePrefix>   Experiment or tune name (required).
#   --checkpoint-number <N>                   Checkpoint number to evaluate (default: auto: best > final > latest).

# Set defaults
EXPERIMENT_NAME=""
CHECKPOINT_NUMBER=""

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      [[ $# -ge 2 && "$2" != -* ]] || 
        { echo "ERROR: --name requires a value" >&2; exit 1; }
      EXPERIMENT_NAME="$2"; shift 2 ;;
    --checkpoint-number)
      [[ $# -ge 2 && "$2" != -* ]] || 
        { echo "ERROR: --checkpoint-number requires a value" >&2; exit 1; }
      [[ "$2" =~ ^[0-9]+$ ]]       || 
        { echo "ERROR: --checkpoint-number must be a non-negative integer, got: $2" >&2; exit 1; }
      CHECKPOINT_NUMBER="$2"; shift 2 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Validate args
# Check if the experiment name is provided
if [ -z "$EXPERIMENT_NAME" ]; then
  echo "Usage: sbatch run_evaluation.sh --name <ExperimentName|TrainablePrefix> [--checkpoint-number N]"
  exit 1
fi

# Print the experiment name and checkpoint number
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
if [ -n "${CHECKPOINT_NUMBER}" ]; then
    echo "CHECKPOINT_NUMBER=${CHECKPOINT_NUMBER}"
else
    echo "CHECKPOINT_NUMBER=(auto: best > final > latest)"
fi


# ============================================================================
# Load modules + env
# ============================================================================

# Load the Python distribution, change to the project directory, 
# and activate the virtual environment based on the cluster
if [ "$CLUSTER" == "lrz" ]; then
    PROJECT_ROOT="/dss/dsshome1/07/go56luf2/projects/marl-sc"
    module load miniforge3/24.3.0-0
    cd "$PROJECT_ROOT"
    source activate ~/projects/marl-sc/.venv
elif [ "$CLUSTER" == "mit-jakobeh" ]; then
    PROJECT_ROOT="/home/jakobeh/projects/marl-sc"
    module load miniforge/25.11.0-0                 
    cd "$PROJECT_ROOT"              
    source ~/projects/marl-sc/.venv/bin/activate    
elif [ "$CLUSTER" == "mit-lumechs" ]; then
    PROJECT_ROOT="/home/lumechs/projects/marl-sc"
    module load miniforge/25.11.0-0                 
    cd "$PROJECT_ROOT"              
    source ~/projects/marl-sc/.venv/bin/activate    
fi

# Set the Python path, unbuffer the output, and set the Python hash seed
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0

# Get the number of CPUs from Slurm
CPUS=${SLURM_CPUS_PER_TASK:-1}
echo "Starting Ray with ${CPUS} CPUs"

# Clean up stale Ray state and ensure we use a fresh local cluster
unset RAY_ADDRESS
ray stop --force 2>/dev/null || true

# Start Ray explicitly with ONLY those CPUs
ray start --head \
  --num-cpus="${CPUS}" \
  --include-dashboard=false \
  --disable-usage-stats


# ============================================================================
# Run evaluation with visualization
# ============================================================================

STORAGE_DIR="./experiment_outputs/Tuning/IPPO_Tune_3WH_5SKUS_Symmetric_Optuna_FIFO"

# Assemble evaluation command
EVAL_CMD=(
    python src/experiments/run_experiment.py
    --mode evaluate
    --storage-dir "${STORAGE_DIR}"
    --experiment-name "${EXPERIMENT_NAME}"
    --eval-episodes 100
    --visualize
    --root-seed 42
)

if [ -n "${CHECKPOINT_NUMBER}" ]; then
    EVAL_CMD+=(--checkpoint-number "${CHECKPOINT_NUMBER}")
fi

# Run evaluation
"${EVAL_CMD[@]}"






