#!/bin/bash

# ============================================================================
# SBATCH directives
# ============================================================================

#SBATCH --job-name=marl-tune                                # Name of the job
##SBATCH --partition=mit_normal                              # Partition MIT      <-- ADJUST THIS!
#SBATCH --clusters=cm4                                      # Cluster LRZ        <-- ADJUST THIS!
#SBATCH --partition=cm4_tiny                                # Partition LRZ      <-- ADJUST THIS!
#SBATCH --nodes=1                                           # Number of nodes
#SBATCH --ntasks-per-node=1                                 # Number of tasks per node 
#SBATCH --cpus-per-task=96                                  # CPU cores per task
#SBATCH --mem=256G                                          # Memory allocation
#SBATCH --time=24:00:00                                     # Maximum walltime 
##SBATCH --chdir=/home/jakobeh/projects/marl-sc             # Working directory MIT Jakobeh   <-- ADJUST THIS!
##SBATCH --chdir=/home/lumechs/projects/marl-sc             # Working directory MIT Lumechs   <-- ADJUST THIS!
#SBATCH --chdir=/dss/dsshome1/07/go56luf2/projects/marl-sc  # Working directory LRZ Go56luf2  <-- ADJUST THIS!
#SBATCH --output=scripts/logs/%x_%j.out                     # Standard output
#SBATCH --error=scripts/logs/%x_%j.err                      # Standard error

# Set the cluster (one of "lrz", "mit-jakobeh", "mit-lumechs")
CLUSTER="lrz"


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
STORAGE_DIR="${PROJECT_ROOT}/experiment_outputs/Tuning"
EXPERIMENT_NAME="MAPPO_PS_Tune_3WH_5SKUS_SKUHetero_A4_Optuna_FIFO"

# Run tune experiment.
# python src/experiments/run_experiment.py \
#     --mode tune \
#     --env-config ./config_files/environments/env_demand_hetero_3WH5SKU.yaml \
#     --algorithm-config ./config_files/algorithms/algorithms_E1/mappo_ps_demand_hetero_E1.yaml \
#     --tune-config ./config_files/experiments/experiments_E2/tune_config_demand_hetero_E2.yaml \
#     --num-samples 100 \
#     --storage-dir "${STORAGE_DIR}" \
#     --experiment-name "${EXPERIMENT_NAME}" \
#     --root-seed 42 \
#     --eval-seed 123

# Resume existing tune experiment
python src/experiments/run_experiment.py \
  --mode tune \
  --resume-from "${EXPERIMENT_NAME}" \
  --storage-dir "${STORAGE_DIR}" \
  --wandb-project marl-sc \
  --eval-seed 123


