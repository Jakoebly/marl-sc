#!/bin/bash

# ============================================================================
# SBATCH directives
# ============================================================================

#SBATCH --job-name=marl-seed-eval               # Name of the job
#SBATCH --partition=mit_normal                  # Partition
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Number of tasks per node
#SBATCH --cpus-per-task=3                       # CPU cores per task
#SBATCH --mem=96G                               # Memory allocation
#SBATCH --time=12:00:00                         # Maximum walltime (hh:mm:ss)
#SBATCH --chdir=/home/jakobeh/projects/marl-sc  # Working directory
#SBATCH --output=scripts/logs/%x_%A_%a.out      # Standard output
#SBATCH --error=scripts/logs/%x_%A_%a.err       # Standard error


# ============================================================================
# Parse arguments
# ============================================================================

# Usage (run directly on login node — not via sbatch):
#   ./scripts/run_seed_evaluation.sh --mode <single|tune> --name <Name> [opts]
#
# Options:
#   --mode <single|tune>    Seed-eval mode (required)
#   --name <Name>           Experiment or tune name (required)
#   --n-seeds <N>           Number of seeds (default: 5)
#   --top-k <K>             Top-K trials to evaluate (tune mode, default: 10)
#   --num-iterations <N>    Override training iterations
#   --eval-episodes <N>     Final eval episodes (default: 100)
#   --phase <worker|aggregate>  Internal, set automatically by the launcher
#
# When --phase is omitted the script acts as a launcher. It computes the
# SLURM array size and submits a worker array job + a dependent aggregate job,
# both pointing back at this same script with the appropriate --phase.

MODE=""
NAME=""
N_SEEDS=5
TOP_K=10
NUM_ITERATIONS=""
EVAL_EPISODES=100
PHASE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --mode requires a value" >&2; exit 1; }
      MODE="$2"; shift 2 ;;
    --name)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --name requires a value" >&2; exit 1; }
      NAME="$2"; shift 2 ;;
    --n-seeds)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --n-seeds requires a value" >&2; exit 1; }
      N_SEEDS="$2"; shift 2 ;;
    --top-k)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --top-k requires a value" >&2; exit 1; }
      TOP_K="$2"; shift 2 ;;
    --num-iterations)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --num-iterations requires a value" >&2; exit 1; }
      NUM_ITERATIONS="$2"; shift 2 ;;
    --eval-episodes)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --eval-episodes requires a value" >&2; exit 1; }
      EVAL_EPISODES="$2"; shift 2 ;;
    --phase)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --phase requires a value" >&2; exit 1; }
      PHASE="$2"; shift 2 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$MODE" ] || [ -z "$NAME" ]; then
  echo "Usage: ./scripts/run_seed_evaluation.sh --mode <single|tune> --name <Name> [options]"
  exit 1
fi
if [[ "$MODE" != "single" && "$MODE" != "tune" ]]; then
  echo "ERROR: --mode must be 'single' or 'tune' (got: ${MODE})" >&2
  exit 1
fi


# ============================================================================
# LAUNCHER: no --phase was provided and we should launch the jobs
# ============================================================================

# If no phase was provided, launch the jobs
if [ -z "$PHASE" ]; then

  # Compute the number of configs
  if [ "$MODE" = "tune" ]; then
    N_CONFIGS=$(python3 -c "
import yaml, sys
path = 'experiment_outputs/Tuning/${NAME}/best_trial_results.yaml'
try:
    with open(path) as f:
        data = yaml.safe_load(f)
    print(min(${TOP_K}, len(data.get('top_k_trials', []))))
except FileNotFoundError:
    print(f'ERROR: {path} not found', file=sys.stderr)
    sys.exit(1)
")
    if [ $? -ne 0 ] || [ -z "$N_CONFIGS" ]; then
      echo "ERROR: Failed to determine number of configs from best_trial_results.yaml"
      exit 1
    fi
  else
    N_CONFIGS=1
  fi

  # Determine the array size and the maximum concurrent tasks
  ARRAY_SIZE=$(( N_CONFIGS * N_SEEDS - 1 ))
  TOTAL_TASKS=$(( ARRAY_SIZE + 1 ))
  MAX_CONCURRENT=$(( TOTAL_TASKS < 17 ? TOTAL_TASKS : 17 ))

  # Collect all arguments to forward to the worker and aggregate phases
  FORWARD_ARGS="--mode ${MODE} --name ${NAME} --n-seeds ${N_SEEDS} --top-k ${TOP_K} --eval-episodes ${EVAL_EPISODES}"
  if [ -n "$NUM_ITERATIONS" ]; then
    FORWARD_ARGS="${FORWARD_ARGS} --num-iterations ${NUM_ITERATIONS}"
  fi
  echo "Launching seed evaluation: mode=${MODE}, name=${NAME}, seeds=${N_SEEDS}, configs=${N_CONFIGS}"
  echo "Total tasks: $(( ARRAY_SIZE + 1 )) (${N_CONFIGS} configs x ${N_SEEDS} seeds)"

  # Submit worker array job
  WORKER_JOB=$(sbatch --parsable \
    --array=0-${ARRAY_SIZE}%${MAX_CONCURRENT} \
    "$0" --phase worker ${FORWARD_ARGS})
  echo "Workers submitted: job ${WORKER_JOB} ($(( ARRAY_SIZE + 1 )) tasks, max ${MAX_CONCURRENT} concurrent)"

  # Submit aggregate job (depends on ALL worker tasks completing)
  AGG_JOB=$(sbatch --parsable \
    --dependency=afterok:${WORKER_JOB} \
    --cpus-per-task=1 --mem=4G --time=00:30:00 \
    "$0" --phase aggregate ${FORWARD_ARGS})
  echo "Aggregate submitted: job ${AGG_JOB} (runs after ${WORKER_JOB})"

  exit 0
fi


# ============================================================================
# SHARED SETUP: shared settings for worker and aggregate phases
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
# AGGREGATE PHASE: aggregate the results from all workers
# ============================================================================

if [ "$PHASE" = "aggregate" ]; then
  echo "Running seed evaluation aggregation for: ${NAME} (mode=${MODE})"

  python -c "
from src.experiments.utils.seed_evaluation import aggregate_and_plot_seed_evaluation
aggregate_and_plot_seed_evaluation(mode='${MODE}', name='${NAME}')
"
  exit $?
fi


# ============================================================================
# WORKER PHASE: each worker trains and evaluates one (config, seed) pair
# ============================================================================

# If the phase is not worker, exit with an error
if [ "$PHASE" != "worker" ]; then
  echo "ERROR: Unknown --phase: ${PHASE}" >&2
  exit 1
fi

# --------------------------------------------------
# Map SLURM_ARRAY_TASK_ID to config and run number
# --------------------------------------------------

# Use the ID of the current task to compute the config index and seed index
ID=${SLURM_ARRAY_TASK_ID}
CONFIG_IDX=$(( ID / N_SEEDS ))
SEED_IDX=$(( ID % N_SEEDS + 1 ))

# Set the root seed as the seed index multiplied by 100 to ensure spaced out seeds
ROOT_SEED=$(( SEED_IDX * 100 ))


# --------------------------------------------------
# Resolve paths and directories
# --------------------------------------------------

# ---------- Single mode ----------
if [ "$MODE" = "single" ]; then

  # Safeguard against multiple configs in single mode
  if [ $CONFIG_IDX -ne 0 ]; then
    echo "ERROR: Single mode supports 1 config (CONFIG_IDX=${CONFIG_IDX} != 0)"
    exit 1
  fi

  # Get the experiment path from the runs directory with the given name
  EXPERIMENT_PATH=$(python -c "
import sys
from src.experiments.utils.experiment_utils import find_experiment_dir
try:
    d = find_experiment_dir('experiment_outputs/Runs', '${NAME}')
    print(str(d))
except (FileNotFoundError, ValueError) as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
")
  if [ $? -ne 0 ] || [ -z "$EXPERIMENT_PATH" ]; then
    echo "ERROR: Failed to find experiment '${NAME}' under experiment_outputs/Runs/"
    exit 1
  fi

  # Set the config path and name and the storage directory explicitly
  CONFIG_PATH="${EXPERIMENT_PATH}"
  CONFIG_NAME="${NAME}"
  STORAGE_DIR="${EXPERIMENT_PATH}/seed_evaluation"
fi

# ---------- Tune mode ----------
elif [ "$MODE" = "tune" ]; then

  # Get the experiment path from the tuning directory with the given name
  EXPERIMENT_PATH="experiment_outputs/Tuning/${NAME}"
  if [ ! -d "$EXPERIMENT_PATH" ]; then
    echo "ERROR: Tune experiment directory not found: ${EXPERIMENT_PATH}"
    exit 1
  fi

  # Get the config path and name for the current CONFIG_IDX from the best trial results
  PYTHON_OUTPUT=$(python -c "
import yaml, sys
with open('${EXPERIMENT_PATH}/best_trial_results.yaml') as f:
    data = yaml.safe_load(f)
trials = data.get('top_k_trials', [])
idx = ${CONFIG_IDX}
if idx >= len(trials):
    print(f'ERROR: CONFIG_IDX={idx} >= len(top_k_trials)={len(trials)}', file=sys.stderr)
    sys.exit(1)
trial = trials[idx]
rank = trial['rank']
short_id = trial['short_id']
trial_path = trial['trial_path']
config_name = f'{rank:02d}_{short_id}'
print(f'{trial_path} {config_name}')
")
  if [ $? -ne 0 ] || [ -z "$PYTHON_OUTPUT" ]; then
    echo "ERROR: Failed to resolve trial config for CONFIG_IDX=${CONFIG_IDX}"
    exit 1
  fi
  read -r CONFIG_PATH CONFIG_NAME <<< "$PYTHON_OUTPUT"

# Set the storage directory to the seed evaluation directory with the config name
  STORAGE_DIR="${EXPERIMENT_PATH}/seed_evaluation/${CONFIG_NAME}"

# Set the experiment name for both single and tune modes as the config name plus the root seed
EXPERIMENT_NAME="${CONFIG_NAME}_Seed${ROOT_SEED}"

echo "Task ${ID} -> Config #${CONFIG_IDX} (${CONFIG_NAME}), Seed #${SEED_IDX} (root_seed=${ROOT_SEED})"
echo "  Config path:  ${CONFIG_PATH}"
echo "  Storage dir:  ${STORAGE_DIR}"
echo "  Experiment:   ${EXPERIMENT_NAME}"


# --------------------------------------------------
# Create temporary configs with overrides
# --------------------------------------------------

# Create temporary config files 
TEMP_ENV_CONFIG=$(mktemp --suffix=.yaml)
TEMP_ALGO_CONFIG=$(mktemp --suffix=.yaml)

# Override values in the environment and algorithm configs
# with the values from the config files
python - <<PY
import yaml, sys
from pathlib import Path

# Get the env and algorithm config paths from the config path
config_path = Path("${CONFIG_PATH}")
env_cfg_path = config_path / "env_config.yaml"
algo_cfg_path = config_path / "algorithm_config.yaml"

# Check if the env and algorithm config files exist
if not env_cfg_path.exists():
    print(f"ERROR: {env_cfg_path} not found", file=sys.stderr)
    sys.exit(1)
if not algo_cfg_path.exists():
    print(f"ERROR: {algo_cfg_path} not found", file=sys.stderr)
    sys.exit(1)

# Load the env config and save it to the temporary config file
with open(env_cfg_path) as f:
    env_cfg = yaml.safe_load(f)
with open("${TEMP_ENV_CONFIG}", "w") as f:
    yaml.safe_dump(env_cfg, f, default_flow_style=False, sort_keys=False)

# Load the algorithm config
with open(algo_cfg_path) as f:
    algo_cfg = yaml.safe_load(f)

# Override the num_iterations value if provided
num_iterations = "${NUM_ITERATIONS}".strip()
if num_iterations:
    algo_cfg["algorithm"]["shared"]["num_iterations"] = int(num_iterations)

# Set the number of env runners and envs per env runner to 0 and 1 respectively	
algo_cfg["algorithm"]["shared"]["num_env_runners"] = 0
algo_cfg["algorithm"]["shared"]["num_envs_per_env_runner"] = 1
algo_cfg["algorithm"]["shared"]["evaluation_parallel_to_training"] = False

# Save the algorithm config to the temporary config file
with open("${TEMP_ALGO_CONFIG}", "w") as f:
    yaml.safe_dump(algo_cfg, f, default_flow_style=False, sort_keys=False)
PY

# Check if the temporary config files were created successfully
if [ $? -ne 0 ]; then
  echo "ERROR: Failed to create temporary config files"
  exit 1
fi


# --------------------------------------------------
# Start Ray explicitly (with port allocation)
# --------------------------------------------------

# Source the start_ray.sh script to start Ray
RAY_FALLBACK_MEM_MB=16384
RAY_EXTRA_CLEANUP_PATHS="$TEMP_ENV_CONFIG $TEMP_ALGO_CONFIG"
source scripts/lib/start_ray.sh


# ============================================================================
# Run training + evaluation
# ============================================================================

# Run training
python src/experiments/run_experiment.py \
    --mode single \
    --env-config "$TEMP_ENV_CONFIG" \
    --algorithm-config "$TEMP_ALGO_CONFIG" \
    --storage-dir "${STORAGE_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --root-seed ${ROOT_SEED}
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
  echo "ERROR: Training failed with exit code ${TRAIN_EXIT}"
  exit $TRAIN_EXIT
fi

# Run evaluation	
python src/experiments/run_experiment.py \
    --mode evaluate \
    --storage-dir "${STORAGE_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --eval-episodes ${EVAL_EPISODES} \
    --root-seed 42
