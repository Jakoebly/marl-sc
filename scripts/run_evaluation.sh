#!/bin/bash

##############################
# SBATCH directives
##############################

#SBATCH --job-name=marl-eval                    # Name of the job
#SBATCH --partition=mit_normal                  # Partition
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Number of tasks per node 
#SBATCH --cpus-per-task=5                       # CPU cores per task
#SBATCH --mem=32G                               # Memory allocation
#SBATCH --time=12:00:00                         # Maximum walltime (hh:mm:ss)
#SBATCH --chdir=/home/jakobeh/projects/marl-sc  # Working directory
#SBATCH --output=scripts/logs/%x_%j.out         # Standard output
#SBATCH --error=scripts/logs/%x_%j.err          # Standard error

##############################
# Parse arguments
##############################

CHECKPOINT_DIR=${1:?"Usage: sbatch run_evaluation.sh <checkpoint-dir>"}
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"

##############################
# Load modules + env
##############################

module load miniforge/25.11.0-0                 # Load the Python distribution
cd /home/jakobeh/projects/marl-sc               # Change to the project directory
source ~/projects/marl-sc/.venv/bin/activate    # Activate the virtual environment

export PYTHONPATH="/home/jakobeh/projects/marl-sc${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# How many CPUs Slurm actually gave you
CPUS=${SLURM_CPUS_PER_TASK:-1}

echo "Starting Ray with ${CPUS} CPUs"

# Start Ray explicitly with ONLY those CPUs
ray start --head \
  --num-cpus="${CPUS}" \
  --include-dashboard=false \
  --disable-usage-stats

##############################
# Run evaluation with visualization
##############################

python src/experiments/run_experiment.py \
    --mode evaluate \
    --env-config config_files/environments/base_env.yaml \
    --algorithm-config config_files/algorithms/mappo.yaml \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --visualize \
    --output-dir ./experiment_outputs \
    --root-seed 42
