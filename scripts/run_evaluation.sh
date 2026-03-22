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

EXPERIMENT_NAME=${1:?"Usage: sbatch run_evaluation.sh <ExperimentName> [CheckpointNumber]"}
CHECKPOINT_NUMBER=${2:-""}
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
if [ -n "${CHECKPOINT_NUMBER}" ]; then
    echo "CHECKPOINT_NUMBER=${CHECKPOINT_NUMBER}"
else
    echo "CHECKPOINT_NUMBER=final (default)"
fi

##############################
# Load modules + env
##############################

module load miniforge/25.11.0-0                 # Load the Python distribution
cd /home/jakobeh/projects/marl-sc               # Change to the project directory
source ~/projects/marl-sc/.venv/bin/activate    # Activate the virtual environment

export PYTHONPATH="/home/jakobeh/projects/marl-sc${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0

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
# Run evaluation with visualization
##############################

# EVAL_CMD=(
#     python src/experiments/run_experiment.py
#     --mode evaluate
#     --experiment-name "${EXPERIMENT_NAME}"
#     --visualize
#     --root-seed 42
# )

# if [ -n "${CHECKPOINT_NUMBER}" ]; then
#     EVAL_CMD+=(--checkpoint-number "${CHECKPOINT_NUMBER}")
# fi

# "${EVAL_CMD[@]}"

python src/experiments/run_experiment.py \
    --mode evaluate \
    --checkpoint-dir "/home/jakobeh/projects/marl-sc/experiment_outputs/Tuning/IPPO_Tune_3WH_2SKUS_Optuna_ASHA_SimplifiedEnv/trainable_a8329ab4_291_actor_obs_type=local,clip_param=0.2000,critic_obs_type=local,entropy_coeff=0.0100,gamma=0.9900,grad_clip=10_2026-03-21_20-16-01/checkpoint_000000" \
    --eval-episodes 50 \
    --visualize \
    --root-seed 42




