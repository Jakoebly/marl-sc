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
#SBATCH --output=scripts/logs/%x_%j.out         # Standard output
#SBATCH --error=scripts/logs/%x_%j.err          # Standard error

##############################
# Load modules + env
##############################

module load miniforge/25.11.0-0                 # Load the Python distribution
source ~/projects/marl-sc/.venv/bin/activate    # Activate the virtual environment

##############################
# Run training
##############################

python src/experiments/run_experiment.py \
    --mode single \
    --env-config config_files/environments/base_env.yaml \
    --algorithm-config config_files/algorithms/ippo.yaml \
    --output-dir ./experiment_outputs \
    --wandb-project marl-sc \
    --root-seed 42
