#!/bin/bash

# ============================================================================
# SBATCH directives
# ============================================================================

#SBATCH --job-name=marl-baseline-single          # Name of the job
#SBATCH --partition=mit_normal                   # Partition
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks-per-node=1                      # Number of tasks per node
#SBATCH --cpus-per-task=2                        # CPU cores per task
#SBATCH --mem=8G                                 # Memory allocation
#SBATCH --time=06:00:00                          # Maximum walltime
#SBATCH --chdir=/home/jakobeh/projects/marl-sc   # Working directory
#SBATCH --output=scripts/logs/%x_%j.out          # Standard output
#SBATCH --error=scripts/logs/%x_%j.err           # Standard error


# ============================================================================
# Parse arguments
# ============================================================================

# Usage:
#   sbatch ./scripts/run_single_baseline.sh --name <Name> --baseline <key> --root-seed <N> [opts]
#
# Options:
#   --name <Name>           Experiment / output folder name (required).
#   --baseline <key>        Single baseline key, e.g. random | constant |
#                           bs_newsvendor | bs_adaptive | bs_optimized |
#                           bs_independent (required).
#   --root-seed <N>         Per-run root seed, e.g. 100 (required).
#   --storage-dir <dir>     Parent directory for the experiment folder
#                           (default: experiment_outputs/Runs).
#   --env-config <path>     Path to the environment config YAML
#                           (default: ./config_files/environments/env_symmetric_3WH5SKU.yaml).
#   --eval-episodes <N>     Final eval episodes (default: 100).
#   --eval-seed <N>         Held-out eval root seed (default: 123).
#
# Output layout (always nested under the baseline's display name):
#   <storage_dir>/<Name>/<DisplayName>/eval_results_final.yaml
#   <storage_dir>/<Name>/visualizations/<DisplayName>/sweep.png
#   <storage_dir>/<Name>/visualizations/<DisplayName>/<DisplayName>_Best_*/

NAME=""
BASELINE=""
ROOT_SEED=""
STORAGE_DIR="experiment_outputs/Runs"
ENV_CONFIG="./config_files/environments/env_symmetric_3WH5SKU.yaml"
EVAL_EPISODES=100
EVAL_SEED=123

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --name requires a value" >&2; exit 1; }
      NAME="$2"; shift 2 ;;
    --baseline)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --baseline requires a value" >&2; exit 1; }
      BASELINE="$2"; shift 2 ;;
    --root-seed)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --root-seed requires a value" >&2; exit 1; }
      ROOT_SEED="$2"; shift 2 ;;
    --storage-dir)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --storage-dir requires a value" >&2; exit 1; }
      STORAGE_DIR="$2"; shift 2 ;;
    --env-config)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --env-config requires a value" >&2; exit 1; }
      ENV_CONFIG="$2"; shift 2 ;;
    --eval-episodes)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --eval-episodes requires a value" >&2; exit 1; }
      EVAL_EPISODES="$2"; shift 2 ;;
    --eval-seed)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --eval-seed requires a value" >&2; exit 1; }
      EVAL_SEED="$2"; shift 2 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$NAME" ] || [ -z "$BASELINE" ] || [ -z "$ROOT_SEED" ]; then
  echo "ERROR: --name, --baseline, and --root-seed are required." >&2
  exit 1
fi


# ============================================================================
# Setup
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

# Pin thread pools to the cores SLURM granted us.
CPUS_THREAD_PIN=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="${CPUS_THREAD_PIN}"
export MKL_NUM_THREADS="${CPUS_THREAD_PIN}"
export OPENBLAS_NUM_THREADS="${CPUS_THREAD_PIN}"
export NUMEXPR_NUM_THREADS="${CPUS_THREAD_PIN}"
export TORCH_NUM_THREADS="${CPUS_THREAD_PIN}"
export RAY_DEFAULT_OMP_NUM_THREADS="${CPUS_THREAD_PIN}"


# ============================================================================
# Run the single baseline with visualization
# ============================================================================

echo "Running single baseline: ${BASELINE} | name=${NAME} | root_seed=${ROOT_SEED}"

python src/experiments/run_baselines.py \
  --mode single \
  --baselines "${BASELINE}" \
  --root-seed "${ROOT_SEED}" \
  --env-config "${ENV_CONFIG}" \
  --storage-dir "${STORAGE_DIR}" \
  --experiment-name "${NAME}" \
  --eval-episodes "${EVAL_EPISODES}" \
  --eval-seed "${EVAL_SEED}" \
  --visualize
