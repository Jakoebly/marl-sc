#!/bin/bash

# ============================================================================
# SBATCH directives
# ============================================================================

#SBATCH --job-name=marl-seed-eval-seq           # Name of the job
#SBATCH --partition=mit_normal                  # Partition
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Number of tasks per node
#SBATCH --cpus-per-task=3                       # CPU cores per task
#SBATCH --mem=32G                               # Memory allocation
#SBATCH --time=12:00:00                         # Maximum walltime (hh:mm:ss)
#SBATCH --chdir=/home/jakobeh/projects/marl-sc  # Working directory
#SBATCH --output=scripts/logs/%x_%A_%a.out      # Standard output
#SBATCH --error=scripts/logs/%x_%A_%a.err       # Standard error


# ============================================================================
# Parse arguments
# ============================================================================

# Usage:
#   	Single mode:
#       sbatch scripts/run_seed_evaluation_sequential.sh \
#           --mode tune \
#           --tune-name "IPPO_Tune_3WH5SKU_Optuna" \
#           [--n-seeds 5] [--top-k 10] [--eval-episodes 100] [--wandb]
#
#   	Tune mode:
#       sbatch scripts/run_seed_evaluation_sequential.sh \
#         --mode single \
#         --env-config ./config_files/environments/env_symmetric_3WH5SKU.yaml \
#         --algorithm-config ./config_files/algorithms/ippo.yaml \
#         --experiment-name "IPPO_3WH5SKU_Final" \
#         [--n-seeds 5] [--eval-episodes 100] [--num-iterations N] [--wandb]
#
# Options:
#   --eval-seed <N>  Root seed for the final evaluation; shared by all
#                    seed runs so paired comparisons see identical eval
#                    episodes (default: 123, distinct from per-seed training
#                    seeds 100,200,... so the benchmark is held out)
#   --wandb          Enable WandB logging with project "marl-sc"
#                    (default: off; no wandb args passed to Python)

SEED_EVAL_MODE=""
ENV_CONFIG=""
ALGO_CONFIG=""
EXPERIMENT_NAME=""
TUNE_NAME=""
N_SEEDS=5
TOP_K=10
EVAL_EPISODES=100
NUM_ITERATIONS=""
USE_WANDB=false

WANDB_PROJECT_NAME="marl-sc"
STORAGE_DIR="./experiment_outputs/Runs"
EVAL_SEED=123

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --mode requires a value" >&2; exit 1; }
      SEED_EVAL_MODE="$2"; shift 2 ;;
    --env-config)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --env-config requires a value" >&2; exit 1; }
      ENV_CONFIG="$2"; shift 2 ;;
    --algorithm-config)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --algorithm-config requires a value" >&2; exit 1; }
      ALGO_CONFIG="$2"; shift 2 ;;
    --experiment-name)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --experiment-name requires a value" >&2; exit 1; }
      EXPERIMENT_NAME="$2"; shift 2 ;;
    --tune-name)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --tune-name requires a value" >&2; exit 1; }
      TUNE_NAME="$2"; shift 2 ;;
    --n-seeds)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --n-seeds requires a value" >&2; exit 1; }
      N_SEEDS="$2"; shift 2 ;;
    --top-k)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --top-k requires a value" >&2; exit 1; }
      TOP_K="$2"; shift 2 ;;
    --eval-episodes)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --eval-episodes requires a value" >&2; exit 1; }
      EVAL_EPISODES="$2"; shift 2 ;;
    --num-iterations)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --num-iterations requires a value" >&2; exit 1; }
      NUM_ITERATIONS="$2"; shift 2 ;;
    --eval-seed)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --eval-seed requires a value" >&2; exit 1; }
      EVAL_SEED="$2"; shift 2 ;;
    --wandb)
      USE_WANDB=true; shift ;;
    *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
  esac
done
if [ -z "$SEED_EVAL_MODE" ]; then
  echo "ERROR: --mode is required (single or tune)" >&2
  exit 1
fi

# Print the arguments
echo "SEED_EVAL_MODE=${SEED_EVAL_MODE}"
echo "N_SEEDS=${N_SEEDS}"
echo "EVAL_EPISODES=${EVAL_EPISODES}"
echo "EVAL_SEED=${EVAL_SEED}"

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
source scripts/lib/start_ray.sh


# ============================================================================
# Build and run the Python command
# ============================================================================

# Build the Python command
CMD=(
  python src/experiments/run_experiment.py
  --mode seed-eval
  --n-seeds "$N_SEEDS"
  --eval-episodes "$EVAL_EPISODES"
  --eval-seed "$EVAL_SEED"
)

if [ "$SEED_EVAL_MODE" = "tune" ]; then
  CMD+=(--tune-name "$TUNE_NAME" --top-k "$TOP_K")
elif [ "$SEED_EVAL_MODE" = "single" ]; then
  CMD+=(--env-config "$ENV_CONFIG" --algorithm-config "$ALGO_CONFIG")
  CMD+=(--experiment-name "$EXPERIMENT_NAME" --storage-dir "$STORAGE_DIR")
else
  echo "ERROR: --mode must be 'single' or 'tune' (got: ${SEED_EVAL_MODE})" >&2
  exit 1
fi

if [ -n "$NUM_ITERATIONS" ]; then
  CMD+=(--num-iterations "$NUM_ITERATIONS")
fi
if [ "$USE_WANDB" = true ]; then
  CMD+=(--wandb-project "$WANDB_PROJECT_NAME")
fi

# Print the command and run it
echo "Running: ${CMD[*]}"
"${CMD[@]}"
