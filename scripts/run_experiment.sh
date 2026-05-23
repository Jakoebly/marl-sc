#!/bin/bash

# ============================================================================
# SBATCH directives
# ============================================================================

#SBATCH --job-name=marl-training                            # Name of the job
##SBATCH --partition=mit_normal                              # Partition MIT      <-- ADJUST THIS!
##SBATCH --clusters=cm4                                      # Cluster LRZ        <-- ADJUST THIS!
##SBATCH --partition=cm4_tiny                                # Partition LRZ      <-- ADJUST THIS!
#SBATCH --clusters=serial                                   # Cluster LRZ        <-- ADJUST THIS!
#SBATCH --partition=serial_std                              # Partition LRZ      <-- ADJUST THIS!
#SBATCH --nodes=1                                           # Number of nodes
#SBATCH --ntasks-per-node=1                                 # Number of tasks per node 
#SBATCH --cpus-per-task=5                                   # CPU cores per task
#SBATCH --mem=32G                                           # Memory allocation
#SBATCH --time=10:00:00                                     # Maximum walltime
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
source scripts/lib/start_ray.sh


# ============================================================================
# Run training
# ============================================================================

# Set output directory and experiment name
STORAGE_DIR="./experiment_outputs/Runs/E1/MAPPO_PS_HeteroTransfer_E1.1"
EXPERIMENT_NAME="MAPPO_PS_Visual_3WH_5SKUS_DemandHetero_E1.1"

# Run training
python src/experiments/run_experiment.py \
    --mode single \
    --env-config ./config_files/environments/env_demand_hetero_3WH5SKU.yaml \
    --algorithm-config ./config_files/algorithms/mappo_ps_E0.yaml \
    --storage-dir "${STORAGE_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --wandb-project marl-sc \
    --root-seed 42

# Run evaluation
python src/experiments/run_experiment.py \
    --mode evaluate \
    --storage-dir "${STORAGE_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --eval-episodes 100 \
    --visualize \
    --root-seed 42
