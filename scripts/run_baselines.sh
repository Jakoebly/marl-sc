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

# Load the Python distribution, change to the project directory, 
# and activate the virtual environment
module load miniforge/25.11.0-0                 
cd /home/jakobeh/projects/marl-sc               
source ~/projects/marl-sc/.venv/bin/activate    

# Set the Python path and unbuffer the output
export PYTHONPATH="/home/jakobeh/projects/marl-sc${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1


##############################
# Run baselines (no Ray needed)
##############################

# Set the storage directory and experiment name
STORAGE_DIR="./experiment_outputs/Runs/Pilot"
EXPERIMENT_NAME="BASELINE_3WH1SKU_DemandHetero"

# Run baselines
python src/experiments/run_baselines.py \
    --env-config ./config_files/environments/env_pilot_demand_hetero.yaml \
    --storage-dir "${STORAGE_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --num-episodes 100 \
    --root-seed 100
