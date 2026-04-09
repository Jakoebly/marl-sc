#!/bin/bash

##############################
# SBATCH directives
##############################

#SBATCH --job-name=marl-eval                    # Name of the job
#SBATCH --partition=mit_normal                  # Partition
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Number of tasks per node 
#SBATCH --cpus-per-task=8                       # CPU cores per task
#SBATCH --mem=32G                               # Memory allocation
#SBATCH --time=12:00:00                         # Maximum walltime (hh:mm:ss)
#SBATCH --chdir=/home/jakobeh/projects/marl-sc  # Working directory
#SBATCH --output=scripts/logs/%x_%j.out         # Standard output
#SBATCH --error=scripts/logs/%x_%j.err          # Standard error


##############################
# Parse arguments
##############################

# Usage: sbatch run_evaluation.sh --name <ExperimentName|TrainablePrefix> [--checkpoint-number N]
EXPERIMENT_NAME=""
CHECKPOINT_NUMBER=""
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

# Validate that name is provided
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


##############################
# Load modules + env
##############################

# Load the Python distribution, change to the project directory, 
# and activate the virtual environment
module load miniforge/25.11.0-0                 
cd /home/jakobeh/projects/marl-sc               
source ~/projects/marl-sc/.venv/bin/activate    

# Set the Python path and unbuffer the output
export PYTHONPATH="/home/jakobeh/projects/marl-sc${PYTHONPATH:+:$PYTHONPATH}"
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


##############################
# Run evaluation with visualization
##############################

# Assemble evaluation command
EVAL_CMD=(
    python src/experiments/run_experiment.py
    --mode evaluate
    --experiment-name "${EXPERIMENT_NAME}"
    --visualize
    --root-seed 42
)

if [ -n "${CHECKPOINT_NUMBER}" ]; then
    EVAL_CMD+=(--checkpoint-number "${CHECKPOINT_NUMBER}")
fi

# Run evaluation
"${EVAL_CMD[@]}"





