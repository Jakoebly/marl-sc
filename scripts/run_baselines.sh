#!/bin/bash

##############################
# SBATCH directives
##############################

#SBATCH --job-name=marl-baselines                # Name of the job
#SBATCH --partition=mit_normal                   # Partition
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks-per-node=1                      # Number of tasks per node
#SBATCH --cpus-per-task=2                        # CPU cores per task
#SBATCH --mem=8G                                 # Memory allocation
#SBATCH --time=01:00:00                          # Maximum walltime (hh:mm:ss)
#SBATCH --chdir=/home/jakobeh/projects/marl-sc   # Working directory
#SBATCH --output=scripts/logs/%x_%j.out          # Standard output
#SBATCH --error=scripts/logs/%x_%j.err           # Standard error

##############################
# Load modules + env
##############################

module load miniforge/25.11.0-0                  # Load the Python distribution
cd /home/jakobeh/projects/marl-sc                # Change to the project directory
source ~/projects/marl-sc/.venv/bin/activate     # Activae the virtual environment

export PYTHONPATH="/home/jakobeh/projects/marl-sc${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

##############################
# Run baselines (no Ray needed)
##############################

EXPERIMENT_NAME="BASELINES_Single_3WH_2SKUS_SingleAgent_Scale1e-2_HCOST3"

python src/experiments/run_baselines.py \
    --env-config ./config_files/environments/env_simplified_symmetric.yaml \
    --output-dir ./experiment_outputs/WorkingConfig_Phase1.2.2 \
    --experiment-name "${EXPERIMENT_NAME}" \
    --num-episodes 10 \
    --root-seed 42
