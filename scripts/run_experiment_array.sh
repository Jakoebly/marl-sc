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
#SBATCH --output=scripts/logs/%x_%A_%a.out      # Standard output
#SBATCH --error=scripts/logs/%x_%A_%a.err       # Standard error
#SBATCH --array=0-5                             # Array for 6 jobs (indices 0-5)

##############################
# Load modules + env
##############################

module load miniforge/25.11.0-0                 # Load the Python distribution
cd /home/jakobeh/projects/marl-sc               # Change to the project directory
source ~/projects/marl-sc/.venv/bin/activate    # Activate the virtual environment

export PYTHONPATH="/home/jakobeh/projects/marl-sc${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Define configs as arrays
WAREHOUSES=(3 5 7 3 5 7)
SKUS=(5 5 5 10 10 10)

echo "POSSIBLE WAREHOUSES: ${WAREHOUSES}"
echo "POSSIBLE SKUS: ${SKUS}"

# Get config for this array task
ID=$SLURM_ARRAY_TASK_ID
N_WAREHOUSES=${WAREHOUSES[$ID]}
N_SKUS=${SKUS[$ID]}

# Create temporary config file
TEMP_CONFIG=$(mktemp)
python -c "
import yaml
with open('config_files/environments/base_env.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['environment']['n_warehouses'] = $N_WAREHOUSES
config['environment']['n_skus'] = $N_SKUS
with open('$TEMP_CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
"

echo "TEMP_CONFIG: ${TEMP_CONFIG}"
cat ${TEMP_CONFIG}

# How many CPUs Slurm actually gave you
CPUS=${SLURM_CPUS_PER_TASK:-1}

echo "Starting Ray with ${CPUS} CPUs"

# Start Ray explicitly with ONLY those CPUs
ray start --head \
  --num-cpus="${CPUS}" \
  --include-dashboard=false \
  --disable-usage-stats

##############################
# Run training
##############################

python src/experiments/run_experiment.py \
    --mode single \
    --env-config "$TEMP_CONFIG" \
    --algorithm-config config_files/algorithms/ippo.yaml \
    --output-dir ./experiment_outputs \
    --wandb-project marl-sc \
    --root-seed 42

# Cleanup
rm "$TEMP_CONFIG"