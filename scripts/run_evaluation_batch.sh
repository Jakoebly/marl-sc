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

# Usage: sbatch run_evaluation_batch.sh <ParentFolder> [CheckpointNumber]
PARENT_DIR=${1:?"Usage: sbatch run_evaluation_batch.sh <ParentFolder> [CheckpointNumber]"}
CHECKPOINT_NUMBER=${2:-""}

# Print the parent directory and checkpoint number
echo "PARENT_DIR=${PARENT_DIR}"
if [ -n "${CHECKPOINT_NUMBER}" ]; then
    echo "CHECKPOINT_NUMBER=${CHECKPOINT_NUMBER}"
    CHECKPOINT_DIR="checkpoint_${CHECKPOINT_NUMBER}"
else
    echo "CHECKPOINT_NUMBER=best (default)"
    CHECKPOINT_DIR="checkpoint_best"
fi

# If parent directory is not found and the name has no path separators, try under 
# experiment_outputs 
if [ ! -d "${PARENT_DIR}" ]; then
    if [[ "${PARENT_DIR}" != */* ]] && [ -d "experiment_outputs/${PARENT_DIR}" ]; then
        PARENT_DIR="experiment_outputs/${PARENT_DIR}"
        echo "Resolved parent directory to: ${PARENT_DIR}"
    else
        echo "[ERROR] Parent directory does not exist: ${PARENT_DIR}"
        exit 1
    fi
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
# Loop over subfolders and run evaluation
##############################

count=0
skipped=0

# Loop over each subfolder in the parent directory and run evaluation
for subdir in "${PARENT_DIR}"/*/; do
    # Skip if not a directory
    [ -d "${subdir}" ] || continue 
    name=$(basename "${subdir}")
    # Skip if no checkpoint directory exists
    if [ ! -d "${subdir}/${CHECKPOINT_DIR}" ]; then
        echo "[SKIP] ${name}: no ${CHECKPOINT_DIR} found"
        ((skipped++)) || true
        continue
    fi

    echo ""
    echo "=========================================="
    echo "[EVAL] Running evaluation for: ${name}"
    echo "=========================================="

    # Assemble evaluation command
    EVAL_CMD=(
        python src/experiments/run_experiment.py
        --mode evaluate
        --storage-dir "${PARENT_DIR}"
        --experiment-name "${name}"
        --visualize
        --root-seed 42
    )

    # Add checkpoint number if provided
    if [ -n "${CHECKPOINT_NUMBER}" ]; then
        EVAL_CMD+=(--checkpoint-number "${CHECKPOINT_NUMBER}")
    fi

    # Run evaluation
    if "${EVAL_CMD[@]}"; then
        ((count++)) || true
        echo "[OK] Completed: ${name}"
    else
        echo "[FAIL] Evaluation failed for: ${name}"
    fi
done

echo ""
echo "=========================================="
echo "Batch evaluation finished: ${count} completed, ${skipped} skipped"
echo "=========================================="

# If no evaluations were completed and no evaluations were skipped, exit with error
if [ "${count}" -eq 0 ] && [ "${skipped}" -eq 0 ]; then
    echo "[WARN] No experiment subfolders found in ${PARENT_DIR}"
    exit 1
fi
