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
# Example: sbatch run_evaluation_batch.sh WorkingConfig_Phase1.4
# Example: sbatch run_evaluation_batch.sh experiment_outputs/WorkingConfig_Phase1.4 50

PARENT_DIR=${1:?"Usage: sbatch run_evaluation_batch.sh <ParentFolder> [CheckpointNumber]

  ParentFolder:     Folder containing experiment subfolders. Can be a bare name (e.g. WorkingConfig_Phase1.4)
                    which is resolved to experiment_outputs/<name>, or a full path.
  CheckpointNumber: Optional. E.g. 50 for checkpoint_50. Default: checkpoint_final

  Loops over each subfolder, runs manual evaluation with --visualize for each."}
CHECKPOINT_NUMBER=${2:-""}

echo "PARENT_DIR=${PARENT_DIR}"
if [ -n "${CHECKPOINT_NUMBER}" ]; then
    echo "CHECKPOINT_NUMBER=${CHECKPOINT_NUMBER}"
    CHECKPOINT_DIR="checkpoint_${CHECKPOINT_NUMBER}"
else
    echo "CHECKPOINT_NUMBER=final (default)"
    CHECKPOINT_DIR="checkpoint_final"
fi

# Resolve path: if not found and name has no path separators, try under experiment_outputs
# (e.g. WorkingConfig_Phase1.4 -> experiment_outputs/WorkingConfig_Phase1.4)
if [ ! -d "${PARENT_DIR}" ]; then
    if [[ "${PARENT_DIR}" != */* ]] && [ -d "experiment_outputs/${PARENT_DIR}" ]; then
        PARENT_DIR="experiment_outputs/${PARENT_DIR}"
        echo "Resolved to: ${PARENT_DIR}"
    else
        echo "[ERROR] Parent directory does not exist: ${PARENT_DIR}"
        exit 1
    fi
fi

##############################
# Load modules + env
##############################

module load miniforge/25.11.0-0                 # Load the Python distribution
cd /home/jakobeh/projects/marl-sc              # Change to the project directory
source ~/projects/marl-sc/.venv/bin/activate    # Activate the virtual environment

export PYTHONPATH="/home/jakobeh/projects/marl-sc${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# How many CPUs Slurm actually gave you
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

# --output-dir is the base for find_experiment_dir (parent of experiment subfolders)
count=0
skipped=0

for subdir in "${PARENT_DIR}"/*/; do
    [ -d "${subdir}" ] || continue
    name=$(basename "${subdir}")
    # Skip if no checkpoint
    if [ ! -d "${subdir}/${CHECKPOINT_DIR}" ]; then
        echo "[SKIP] ${name}: no ${CHECKPOINT_DIR} found"
        ((skipped++)) || true
        continue
    fi
    echo ""
    echo "=========================================="
    echo "[EVAL] Running evaluation for: ${name}"
    echo "=========================================="

    EVAL_CMD=(
        python src/experiments/run_experiment.py
        --mode evaluate
        --output-dir "${PARENT_DIR}"
        --experiment-name "${name}"
        --visualize
        --root-seed 42
    )

    if [ -n "${CHECKPOINT_NUMBER}" ]; then
        EVAL_CMD+=(--checkpoint-number "${CHECKPOINT_NUMBER}")
    fi

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

if [ "${count}" -eq 0 ] && [ "${skipped}" -eq 0 ]; then
    echo "[WARN] No experiment subfolders found in ${PARENT_DIR}"
    exit 1
fi
