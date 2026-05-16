#!/bin/bash

# ============================================================================
# SBATCH directives
# ============================================================================

#SBATCH --job-name=marl-seed-eval               # Name of the job
#SBATCH --partition=mit_normal                  # Partition
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Number of tasks per node
#SBATCH --cpus-per-task=3                       # CPU cores per task
#SBATCH --mem=32G                               # Memory allocation
#SBATCH --time=04:00:00                         # Max walltime per worker (resume-on-restart)
##SBATCH --chdir=/home/jakobeh/projects/marl-sc  # Working directory
#SBATCH --chdir=/home/lumechs/projects/marl-sc  # Working directory
#SBATCH --output=scripts/logs/%x_%A_%a.out      # Standard output
#SBATCH --error=scripts/logs/%x_%A_%a.err       # Standard error
#SBATCH --exclude=node1620,node1621,node1622,node1623,node1624,node1625,node2704,node2705


# ============================================================================
# Parse arguments
# ============================================================================

# Usage (run directly on login node — not via sbatch):
#   ./scripts/run_seed_evaluation.sh --mode <single|tune> --name <Name> [opts]
#
# Options:
#   --mode <single|tune>      Seed-eval mode (required)
#   --name <Name>             Experiment or tune name (required).
#                             Existing-run sub-case (no explicit configs):
#                             resolved against <storage-dir>/ via
#                             find_experiment_dir (supports prefix match).
#                             New-config sub-case (with --env-config +
#                             --algorithm-config): used as the literal folder
#                             name under <storage-dir>/, created on demand.
#   --storage-dir <path>      Base directory for runs (single mode) or tune
#                             experiments (tune mode). Defaults to
#                             experiment_outputs/Runs (single) or
#                             experiment_outputs/Tuning (tune). When provided
#                             explicitly, the directory must already exist.
#   --env-config <path>       Single-mode only. Path to an env config YAML to
#                             seed-evaluate a fresh config (no prior run
#                             required). Must be passed together with
#                             --algorithm-config. When omitted, the saved
#                             env_config.yaml in the resolved run directory
#                             is used.
#   --algorithm-config <path> Single-mode only. Path to an algorithm config
#                             YAML. Must be passed together with --env-config.
#                             When omitted, the saved algorithm_config.yaml in
#                             the resolved run directory is used.
#   --n-seeds <N>             Number of seeds (default: 5)
#   --top-k <K>               Top-K trials to evaluate (tune mode, default: 10)
#   --num-iterations <N>      Override training iterations
#   --final-eval-episodes <N> Evaluation episodes for both intra-training and 
#                             final evaluation (default: 100)
#   --intra-eval-episodes <N> Evaluation episodes for intra-training 
#                             evaluations (default: 50)
#   --eval-interval <N>       Intra-training eval interval (default: 1)
#   --eval-seed <N>           Root seed for the final evaluation; shared by
#                             all (config, seed) pairs so paired comparisons
#                             across seed runs see identical eval episodes
#                             (default: 123, distinct from per-seed training
#                             seeds 100,200,... so the benchmark is held out)
#   --checkpoint-freq <N>     Override algorithm.shared.checkpoint_freq per
#                             worker so resume granularity is fine enough for
#                             the 6h walltime cap (default: 25)
#   --wandb                   Enable WandB logging with project "marl-sc"
#                             (default: off; no wandb args passed to Python)
#   --phase <worker|aggregate>Internal, set automatically by the launcher
#   --heal-round <N>          Internal, self-heal recursion depth (default: 0)
#   --max-heal-rounds <N>     Stop self-healing after N rounds (default: 2)
#
# When --phase is omitted the script acts as a launcher. It computes the
# SLURM array size and submits a worker array job + a dependent aggregate job,
# both pointing back at this same script with the appropriate --phase.
#
# The aggregate phase is self-healing: if some (config, seed) pairs are
# still missing eval_results_final.yaml, it re-submits only those array
# indices and chains a new aggregate behind them. Workers auto-resume from
# the latest periodic checkpoint when one is present.

WANDB_PROJECT_NAME="marl-sc"

MODE=""
NAME=""
STORAGE_DIR=""
ENV_CONFIG=""
ALGO_CONFIG=""
N_SEEDS=10
TOP_K=10
NUM_ITERATIONS=""
FINAL_EVAL_EPISODES=100
INTRA_EVAL_EPISODES=50
EVAL_INTERVAL=1
EVAL_SEED=123
CHECKPOINT_FREQ=25
PHASE=""
HEAL_ROUND=0
MAX_HEAL_ROUNDS=2
USE_WANDB=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --mode requires a value" >&2; exit 1; }
      MODE="$2"; shift 2 ;;
    --name)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --name requires a value" >&2; exit 1; }
      NAME="$2"; shift 2 ;;
    --storage-dir)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --storage-dir requires a value" >&2; exit 1; }
      STORAGE_DIR="$2"; shift 2 ;;
    --env-config)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --env-config requires a value" >&2; exit 1; }
      ENV_CONFIG="$2"; shift 2 ;;
    --algorithm-config)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --algorithm-config requires a value" >&2; exit 1; }
      ALGO_CONFIG="$2"; shift 2 ;;
    --n-seeds)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --n-seeds requires a value" >&2; exit 1; }
      N_SEEDS="$2"; shift 2 ;;
    --top-k)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --top-k requires a value" >&2; exit 1; }
      TOP_K="$2"; shift 2 ;;
    --num-iterations)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --num-iterations requires a value" >&2; exit 1; }
      NUM_ITERATIONS="$2"; shift 2 ;;
    --final-eval-episodes)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --final-eval-episodes requires a value" >&2; exit 1; }
      FINAL_EVAL_EPISODES="$2"; shift 2 ;;
    --intra-eval-episodes)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --intra-eval-episodes requires a value" >&2; exit 1; }
      INTRA_EVAL_EPISODES="$2"; shift 2 ;;
    --eval-interval)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --eval-interval requires a value" >&2; exit 1; }
      EVAL_INTERVAL="$2"; shift 2 ;;
    --eval-seed)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --eval-seed requires a value" >&2; exit 1; }
      EVAL_SEED="$2"; shift 2 ;;
    --checkpoint-freq)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --checkpoint-freq requires a value" >&2; exit 1; }
      CHECKPOINT_FREQ="$2"; shift 2 ;;
    --phase)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --phase requires a value" >&2; exit 1; }
      PHASE="$2"; shift 2 ;;
    --heal-round)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --heal-round requires a value" >&2; exit 1; }
      HEAL_ROUND="$2"; shift 2 ;;
    --max-heal-rounds)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --max-heal-rounds requires a value" >&2; exit 1; }
      MAX_HEAL_ROUNDS="$2"; shift 2 ;;
    --wandb)
      USE_WANDB=true; shift ;;
    *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Validate args
if [ -z "$MODE" ]; then
  echo "ERROR: --mode is required (single or tune)" >&2
  exit 1
fi
if [[ "$MODE" != "single" && "$MODE" != "tune" ]]; then
  echo "ERROR: --mode must be 'single' or 'tune' (got: ${MODE})" >&2
  exit 1
fi
if [ -n "$ENV_CONFIG" ] && [ -z "$ALGO_CONFIG" ]; then
  echo "ERROR: --env-config requires --algorithm-config" >&2
  exit 1
fi
if [ -z "$ENV_CONFIG" ] && [ -n "$ALGO_CONFIG" ]; then
  echo "ERROR: --algorithm-config requires --env-config" >&2
  exit 1
fi
if [ "$MODE" = "tune" ] && { [ -n "$ENV_CONFIG" ] || [ -n "$ALGO_CONFIG" ]; }; then
  echo "ERROR: --env-config / --algorithm-config are not valid in tune mode" >&2
  exit 1
fi
if [ -z "$NAME" ]; then
  echo "ERROR: --name is required for seed evaluation" >&2
  exit 1
fi

# Default to the canonical storage directory when --storage-dir was not provided.
if [ -z "$STORAGE_DIR" ]; then
  if [ "$MODE" = "single" ]; then
    STORAGE_DIR="experiment_outputs/Runs"
  else
    STORAGE_DIR="experiment_outputs/Tuning"
  fi
elif [ ! -d "$STORAGE_DIR" ]; then
  echo "ERROR: --storage-dir does not exist: ${STORAGE_DIR}" >&2
  exit 1
fi


# ============================================================================
# LAUNCHER: no --phase was provided and we should launch the jobs
# ============================================================================

# If no phase was provided, launch the jobs
if [ -z "$PHASE" ]; then

  # If in tune mode compute the number of configs based on the length of the top-k trials
  if [ "$MODE" = "tune" ]; then
    N_CONFIGS=$(python3 -c "
import sys
from pathlib import Path
import yaml
from src.experiments.utils.experiment_utils import find_experiment_dir
try:
    experiment_dir = find_experiment_dir('${STORAGE_DIR}', '${NAME}')
    if not experiment_dir.exists():
        raise FileNotFoundError(f'Experiment directory not found: {experiment_dir}')
    path = experiment_dir / 'best_trial_results.yaml'
    with open(path) as f:
        data = yaml.safe_load(f)
    print(min(${TOP_K}, len(data.get('top_k_trials', []))))
except (FileNotFoundError, ValueError) as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
")
    if [ $? -ne 0 ] || [ -z "$N_CONFIGS" ]; then
      echo "ERROR: Failed to determine number of configs from best_trial_results.yaml"
      exit 1
    fi
  
  # If in single mode, set the number of configs to 1
  else
    N_CONFIGS=1

    # If explicit env/algorithm configs are provided, make sure they exist
    if [ -n "$ENV_CONFIG" ] && [ -n "$ALGO_CONFIG" ]; then
      if [ ! -f "$ENV_CONFIG" ]; then
        echo "ERROR: --env-config not found: ${ENV_CONFIG}" >&2
        exit 1
      fi
      if [ ! -f "$ALGO_CONFIG" ]; then
        echo "ERROR: --algorithm-config not found: ${ALGO_CONFIG}" >&2
        exit 1
      fi
    
    # If no explicit env/algorithm configs are provided, find the experiment directory
    # and make sure the saved env/algorithm configs can be resolved
    else
      EXPERIMENT_DIR=$(python3 -c "
import sys
from src.experiments.utils.experiment_utils import find_experiment_dir
try:
    experiment_dir = find_experiment_dir('${STORAGE_DIR}', '${NAME}')
    print(str(experiment_dir))
except (FileNotFoundError, ValueError) as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
")
      if [ $? -ne 0 ] || [ -z "$EXPERIMENT_DIR" ]; then
        echo "ERROR: No run directory matching '${NAME}' found under ${STORAGE_DIR}/." >&2
        echo "  Either point --name at an existing run, or pass --env-config +" >&2
        echo "  --algorithm-config to seed-evaluate a fresh config." >&2
        exit 1
      fi
      if ! python3 -c "
import os
import sys
from pathlib import Path
from src.experiments.utils.experiment_utils import resolve_saved_config
try:
    d = Path('${EXPERIMENT_DIR}')
    resolve_saved_config(d, 'env_config.yaml')
    resolve_saved_config(d, 'algorithm_config.yaml')
except FileNotFoundError as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"; then
        echo "ERROR: Resolved run dir '${EXPERIMENT_DIR}' is missing env_config.yaml" >&2
        echo "  or algorithm_config.yaml. Either pass --env-config + --algorithm-config" >&2
        echo "  to seed-evaluate a fresh config, or use a run with saved configs." >&2
        exit 1
      fi

    fi
  fi

  # Determine the array size and the maximum concurrent tasks
  ARRAY_SIZE=$(( N_CONFIGS * N_SEEDS - 1 ))
  TOTAL_TASKS=$(( ARRAY_SIZE + 1 ))
  MAX_CONCURRENT=$(( TOTAL_TASKS < 20 ? TOTAL_TASKS : 20 ))

  # Collect all arguments to forward to the worker and aggregate phases
  FORWARD_ARGS="--mode ${MODE} --name ${NAME} --storage-dir ${STORAGE_DIR} --n-seeds ${N_SEEDS} --top-k ${TOP_K} --final-eval-episodes ${FINAL_EVAL_EPISODES} --intra-eval-episodes ${INTRA_EVAL_EPISODES} --eval-interval ${EVAL_INTERVAL} --eval-seed ${EVAL_SEED} --checkpoint-freq ${CHECKPOINT_FREQ} --max-heal-rounds ${MAX_HEAL_ROUNDS}"
  if [ -n "$NUM_ITERATIONS" ]; then
    FORWARD_ARGS="${FORWARD_ARGS} --num-iterations ${NUM_ITERATIONS}"
  fi
  if [ -n "$ENV_CONFIG" ]; then
    FORWARD_ARGS="${FORWARD_ARGS} --env-config ${ENV_CONFIG}"
  fi
  if [ -n "$ALGO_CONFIG" ]; then
    FORWARD_ARGS="${FORWARD_ARGS} --algorithm-config ${ALGO_CONFIG}"
  fi
  if [ "$USE_WANDB" = true ]; then
    FORWARD_ARGS="${FORWARD_ARGS} --wandb"
  fi
  echo "Launching seed evaluation: mode=${MODE}, name=${NAME}, configs=${N_CONFIGS}, n_seeds=${N_SEEDS}"
  echo "Total tasks: ${TOTAL_TASKS} (${N_CONFIGS} configs x ${N_SEEDS} seeds)"

  # Submit worker array job
  WORKER_JOB=$(sbatch --parsable \
    --array=0-${ARRAY_SIZE}%${MAX_CONCURRENT} \
    "$0" --phase worker ${FORWARD_ARGS})
  echo "Workers submitted: job ${WORKER_JOB} ($(( ARRAY_SIZE + 1 )) tasks, max ${MAX_CONCURRENT} concurrent)"

  # Submit aggregate job
  AGG_JOB=$(sbatch --parsable \
    --dependency=afterany:${WORKER_JOB} \
    --cpus-per-task=1 --mem=4G --time=00:30:00 \
    "$0" --phase aggregate ${FORWARD_ARGS} --heal-round 0)
  echo "Aggregate submitted: job ${AGG_JOB} (runs after ${WORKER_JOB})"

  exit 0
fi


# ============================================================================
# SHARED SETUP: shared settings for worker and aggregate phases
# ============================================================================

# Load the Python distribution, change to the project directory, 
# and activate the virtual environment
# PROJECT_ROOT="/home/jakobeh/projects/marl-sc"
PROJECT_ROOT="/home/lumechs/projects/marl-sc"
module load miniforge/25.11.0-0
cd "$PROJECT_ROOT"
source ~/projects/marl-sc/.venv/bin/activate

# Set the Python path, unbuffer the output, and set the Python hash seed
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

# Pin thread pools to the cores SLURM actually granted us.
# Default Python/PyTorch detects *physical* cores on the whole node and
# oversubscribes with other tasks sharing the node, potentially
# resulting in slow runs.
CPUS_THREAD_PIN=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="${CPUS_THREAD_PIN}"
export MKL_NUM_THREADS="${CPUS_THREAD_PIN}"
export OPENBLAS_NUM_THREADS="${CPUS_THREAD_PIN}"
export NUMEXPR_NUM_THREADS="${CPUS_THREAD_PIN}"
export TORCH_NUM_THREADS="${CPUS_THREAD_PIN}"
export RAY_DEFAULT_OMP_NUM_THREADS="${CPUS_THREAD_PIN}"


# ============================================================================
# AGGREGATE PHASE: aggregate the results (self-healing)
# ============================================================================

if [ "$PHASE" = "aggregate" ]; then
  echo "Running seed evaluation aggregation for: ${NAME} (mode=${MODE}, heal_round=${HEAL_ROUND})"

  # ---------- Step 1 ----------
  # Detect any (config, seed) pairs whose eval_results_final.yaml is
  # missing or malformed. These are the tasks we need to re-submit.
  # Use a sentinel prefix so we can safely strip any import-time noise that
  # leaks to stdout from Ray/PyTorch/etc.
  MISSING_RAW=$(python -c "
from src.experiments.utils.seed_evaluation import find_missing_seed_evaluation_tasks
missing = find_missing_seed_evaluation_tasks(
    mode='${MODE}', 
    name='${NAME}', 
    n_seeds=${N_SEEDS}, 
    top_k=${TOP_K},
    storage_dir='${STORAGE_DIR}',
)
print('__MISSING_TASKS__:' + ','.join(str(t) for t in missing))
")
  MISSING_RC=$?
  if [ $MISSING_RC -ne 0 ]; then
    echo "ERROR: Could not scan for missing seed-eval tasks (exit ${MISSING_RC})"
    echo "python output was:"
    echo "$MISSING_RAW"
    exit $MISSING_RC
  fi
  MISSING_STR=$(echo "$MISSING_RAW" | grep '^__MISSING_TASKS__:' | tail -n1 | sed 's/^__MISSING_TASKS__://')

  # ---------- Step 2 ----------
  # If work is still pending and we haven't exhausted the heal
  # budget, re-submit JUST the missing array indices and chain a new
  # aggregate behind them.
  if [ -n "$MISSING_STR" ]; then
    # Count the number of missing tasks
    MISSING_COUNT=$(echo "$MISSING_STR" | awk -F, '{print NF}')
    if [ "$HEAL_ROUND" -lt "$MAX_HEAL_ROUNDS" ]; then
      NEXT_ROUND=$(( HEAL_ROUND + 1 ))
      MAX_CONCURRENT=$(( MISSING_COUNT < 20 ? MISSING_COUNT : 20 ))
      echo "[HEAL ${NEXT_ROUND}/${MAX_HEAL_ROUNDS}] ${MISSING_COUNT} task(s) missing: ${MISSING_STR}"

      # Build the forward arguments for the worker and aggregate jobs
      HEAL_FORWARD="--mode ${MODE} --name ${NAME} --storage-dir ${STORAGE_DIR} --n-seeds ${N_SEEDS} --top-k ${TOP_K} --final-eval-episodes ${FINAL_EVAL_EPISODES} --intra-eval-episodes ${INTRA_EVAL_EPISODES} --eval-interval ${EVAL_INTERVAL} --eval-seed ${EVAL_SEED} --checkpoint-freq ${CHECKPOINT_FREQ} --max-heal-rounds ${MAX_HEAL_ROUNDS}"
      if [ -n "$NUM_ITERATIONS" ]; then
        HEAL_FORWARD="${HEAL_FORWARD} --num-iterations ${NUM_ITERATIONS}"
      fi
      if [ -n "$ENV_CONFIG" ]; then
        HEAL_FORWARD="${HEAL_FORWARD} --env-config ${ENV_CONFIG}"
      fi
      if [ -n "$ALGO_CONFIG" ]; then
        HEAL_FORWARD="${HEAL_FORWARD} --algorithm-config ${ALGO_CONFIG}"
      fi
      if [ "$USE_WANDB" = true ]; then
        HEAL_FORWARD="${HEAL_FORWARD} --wandb"
      fi

      # Submit the missing worker jobs
      HEAL_WORKER_JOB=$(sbatch --parsable \
        --array=${MISSING_STR}%${MAX_CONCURRENT} \
        "$0" --phase worker ${HEAL_FORWARD})
      echo "[HEAL] Re-submitted workers: job ${HEAL_WORKER_JOB}"

      # Submit a new aggregate job
      HEAL_AGG_JOB=$(sbatch --parsable \
        --dependency=afterany:${HEAL_WORKER_JOB} \
        --cpus-per-task=1 --mem=4G --time=00:30:00 \
        "$0" --phase aggregate ${HEAL_FORWARD} --heal-round ${NEXT_ROUND})
      echo "[HEAL] Chained aggregate: job ${HEAL_AGG_JOB}"

      # Deliberately exit 0 since the next aggregate job will do the final plotting
      exit 0
    
    # If we have exhausted the heal budget, proceed to the final aggregation
    else
      echo "[HEAL] Exhausted ${MAX_HEAL_ROUNDS} heal rounds but ${MISSING_COUNT} task(s) still missing: ${MISSING_STR}"
      echo "[HEAL] Proceeding to aggregate with incomplete results."
    fi
  fi

  # ---------- Step 3 ----------
  # Final aggregation + plots
  python -c "
from src.experiments.utils.seed_evaluation import aggregate_and_plot_seed_evaluation
aggregate_and_plot_seed_evaluation(
    mode='${MODE}', 
    name='${NAME}',
    storage_dir='${STORAGE_DIR}',
)
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

  # If new environment and algorithm configs are provided, create the experiment path
  if [ -n "$ENV_CONFIG" ] && [ -n "$ALGO_CONFIG" ]; then
    EXPERIMENT_DIR="${STORAGE_DIR}/${NAME}"
    mkdir -p "$EXPERIMENT_DIR"
    ENV_CFG_FILE="$ENV_CONFIG"
    ALGO_CFG_FILE="$ALGO_CONFIG"
  
  # If an existing run name is provided, find its experiment dir
  else
    EXPERIMENT_DIR=$(python -c "
import sys
from src.experiments.utils.experiment_utils import find_experiment_dir
try:
    experiment_dir = find_experiment_dir('${STORAGE_DIR}', '${NAME}')
    print(str(experiment_dir))
except (FileNotFoundError, ValueError) as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
")
    if [ $? -ne 0 ] || [ -z "$EXPERIMENT_DIR" ]; then
      echo "ERROR: Failed to find experiment '${NAME}' under ${STORAGE_DIR}/"
      exit 1
    fi
    ENV_CFG_FILE="${EXPERIMENT_DIR}/env_config.yaml"
    ALGO_CFG_FILE="${EXPERIMENT_DIR}/algorithm_config.yaml"
  fi

  # Set the config path and name and the storage directory explicitly
  EXPERIMENT_NAME="${NAME}"
  SEED_DIR="${EXPERIMENT_DIR}/seed_evaluation"

# ---------- Tune mode ----------
elif [ "$MODE" = "tune" ]; then

  # Get the experiment directory from the storage_dir and name
    EXPERIMENT_DIR=$(python -c "
import sys
from pathlib import Path
from src.experiments.utils.experiment_utils import find_experiment_dir
try:
    experiment_dir = find_experiment_dir('${STORAGE_DIR}', '${NAME}')
    print(experiment_dir)
except (FileNotFoundError, ValueError) as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
")
  if [ $? -ne 0 ] || [ -z "$EXPERIMENT_DIR" ]; then
    echo "ERROR: Tune experiment not found for '${NAME}' under ${STORAGE_DIR}/"
    exit 1
  fi

  # Get the config path and name for the current CONFIG_IDX from the best trial results
  PYTHON_OUTPUT=$(python -c "
import yaml, sys
with open('${EXPERIMENT_DIR}/best_trial_results.yaml') as f:
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
  read -r TRIAL_DIR EXPERIMENT_NAME <<< "$PYTHON_OUTPUT"

  # Tune-mode configs always live next to the trial directory
  ENV_CFG_FILE="${TRIAL_DIR}/env_config.yaml"
  ALGO_CFG_FILE="${TRIAL_DIR}/algorithm_config.yaml"

  # Set the seed evaluation directory
  SEED_DIR="${EXPERIMENT_DIR}/seed_evaluation/${EXPERIMENT_NAME}"
fi

# Set the seed experiment name as the experiment name plus the root seed
SEED_EXPERIMENT_NAME="${EXPERIMENT_NAME}_Seed${ROOT_SEED}"

echo "Task ${ID} -> Config #${CONFIG_IDX} (${EXPERIMENT_NAME}), Seed #${SEED_IDX} (root_seed=${ROOT_SEED})"
echo "  Env config:   ${ENV_CFG_FILE}"
echo "  Algo config:  ${ALGO_CFG_FILE}"
echo "  Storage dir:  ${SEED_DIR}"
echo "  Experiment:   ${SEED_EXPERIMENT_NAME}"


# --------------------------------------------------
# Detect resumable checkpoint
# --------------------------------------------------

# If a prior (killed) run for this (config, seed) left a periodic checkpoint_<N> behind, 
# resume from the largest N instead of starting from iteration 1.
RESUME_FROM=""
SEED_EXPERIMENT_DIR="${SEED_DIR}/${SEED_EXPERIMENT_NAME}"
if [ -d "$SEED_EXPERIMENT_DIR" ]; then
  LATEST_CHKPT=$(ls -d "${SEED_EXPERIMENT_DIR}"/checkpoint_[0-9]* 2>/dev/null \
    | awk -F'checkpoint_' '{print $2"|"$0}' \
    | sort -n \
    | tail -n1 \
    | cut -d'|' -f2-)
  if [ -n "$LATEST_CHKPT" ] && [ -d "$LATEST_CHKPT" ]; then
    RESUME_FROM="$LATEST_CHKPT"
    echo "[RESUME] Found existing periodic checkpoint: ${RESUME_FROM}"
  fi
fi


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

# Resolve env/algorithm config paths (file paths, set by the worker phase)
env_cfg_path = Path("${ENV_CFG_FILE}")
algo_cfg_path = Path("${ALGO_CFG_FILE}")

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

# Override the checkpoint frequency if provided
checkpoint_freq = int("${CHECKPOINT_FREQ}")
if checkpoint_freq > 0:
    algo_cfg["algorithm"]["shared"]["checkpoint_freq"] = checkpoint_freq

# Override the eval interval and eval episodes if provided 
eval_interval = int("${EVAL_INTERVAL}")
eval_episodes = int("${INTRA_EVAL_EPISODES}")
if eval_interval is not None:
    algo_cfg["algorithm"]["shared"]["eval_interval"] = eval_interval
if eval_episodes is not None:
    algo_cfg["algorithm"]["shared"]["num_eval_episodes"] = eval_episodes

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


# --------------------------------------------------
# Run training + evaluation
# --------------------------------------------------

# Build training command
TRAIN_CMD=(python src/experiments/run_experiment.py
    --mode single
    --env-config "$TEMP_ENV_CONFIG"
    --algorithm-config "$TEMP_ALGO_CONFIG"
    --storage-dir "${SEED_DIR}"
    --experiment-name "${SEED_EXPERIMENT_NAME}"
    --root-seed ${ROOT_SEED}
    --eval-seed ${EVAL_SEED})
if [ -n "$RESUME_FROM" ]; then
  TRAIN_CMD+=(--resume-from "$RESUME_FROM")
fi
if [ "$USE_WANDB" = true ]; then
  TRAIN_CMD+=(--wandb-project "$WANDB_PROJECT_NAME" --wandb-name "$SEED_EXPERIMENT_NAME")
fi

# Run training
"${TRAIN_CMD[@]}"
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
  echo "ERROR: Training failed with exit code ${TRAIN_EXIT}"
  exit $TRAIN_EXIT
fi

# Run evaluation
python src/experiments/run_experiment.py \
    --mode evaluate \
    --storage-dir "${SEED_DIR}" \
    --experiment-name "${SEED_EXPERIMENT_NAME}" \
    --eval-episodes ${FINAL_EVAL_EPISODES} \
    --root-seed ${EVAL_SEED}
