#!/bin/bash

# ============================================================================
# SBATCH directives
# ============================================================================

#SBATCH --job-name=marl-tune                    # Name of the job
#SBATCH --partition=mit_normal                  # Partition
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Number of tasks per node 
#SBATCH --cpus-per-task=96                      # CPU cores per task
#SBATCH --mem=256G                              # Memory allocation
#SBATCH --time=12:00:00                         # Maximum walltime (hh:mm:ss)
#SBATCH --chdir=/home/jakobeh/projects/marl-sc  # Working directory
#SBATCH --output=scripts/logs/%x_%j.out         # Standard output
#SBATCH --error=scripts/logs/%x_%j.err          # Standard error


# ============================================================================
# Load modules + env
# ============================================================================

# Load the Python distribution, change to the project directory, 
# and activate the virtual environment
module load miniforge/25.11.0-0                 
cd /home/jakobeh/projects/marl-sc               
source ~/projects/marl-sc/.venv/bin/activate    

# Set the Python path, unbuffer the output, and set the Python hash seed
export PYTHONPATH="/home/jakobeh/projects/marl-sc${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0


# ============================================================================
# Start Ray explicitly (with port allocation)
# ============================================================================

# Source the start_ray.sh script to start Ray
RAY_PREFERRED_BLOCK_SIZE=1000
RAY_FALLBACK_MEM_MB=65536
source scripts/lib/start_ray.sh


# ============================================================================
# Run tune experiment
# ============================================================================

# Set output directory and experiment name
STORAGE_DIR="/home/jakobeh/projects/marl-sc/experiment_outputs/Tuning"
EXPERIMENT_NAME="MAPPO_Tune_3WH_5SKUS_Symmetric_Optuna_FIFO"

# Run tune experiment.
# python src/experiments/run_experiment.py \
#     --mode tune \
#     --env-config ./config_files/environments/env_symmetric_3WH5SKU.yaml \
#     --algorithm-config ./config_files/algorithms/mappo.yaml \
#     --tune-config ./config_files/experiments/tune_config.yaml \
#     --num-samples 1000 \
#     --storage-dir "${STORAGE_DIR}" \
#     --experiment-name "${EXPERIMENT_NAME}" \
#     --wandb-project marl-sc \
#     --root-seed 42 \
#     --eval-seed 123

# Resume existing tune experiment
python src/experiments/run_experiment.py \
  --mode tune \
  --resume-from "${EXPERIMENT_NAME}" \
  --storage-dir "${STORAGE_DIR}" \
  --wandb-project marl-sc \
  --eval-seed 123
