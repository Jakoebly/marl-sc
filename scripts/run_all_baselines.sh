#!/bin/bash

# ============================================================================
# SBATCH directives
# ============================================================================

#SBATCH --job-name=marl-baselines                # Name of the job
#SBATCH --partition=mit_normal                   # Partition
#SBATCH --nodes=1                                # Number of nodes
#SBATCH --ntasks-per-node=1                      # Number of tasks per node
#SBATCH --cpus-per-task=2                        # CPU cores per task
#SBATCH --mem=8G                                 # Memory allocation
#SBATCH --time=06:00:00                          # Maximum walltime per worker
#SBATCH --chdir=/home/jakobeh/projects/marl-sc   # Working directory
#SBATCH --output=scripts/logs/%x_%A_%a.out       # Standard output
#SBATCH --error=scripts/logs/%x_%A_%a.err        # Standard error


# ============================================================================
# Parse arguments
# ============================================================================

# Usage (run directly on login node -- not via sbatch):
#   ./scripts/run_all_baselines.sh --name <Name> [opts]
#
# Options:
#   --name <Name>           Experiment / output folder name (required).
#   --storage-dir <dir>     Parent directory for the experiment folder
#                           (default: experiment_outputs/Runs)
#   --n-seeds <N>           Number of seeds (default: 5)
#   --eval-episodes <N>     Final eval episodes (default: 100)
#   --eval-seed <N>         Held-out eval root seed shared by all
#                           (baseline, seed) pairs and (config, seed) RL
#                           pairs (default: 123)
#   --baselines <list>      Space-separated subset of baselines (default: all)
#   --phase <worker|aggregate>  Internal, set automatically by the launcher
#   --heal-round <N>        Internal, self-heal recursion depth (default: 0)
#   --max-heal-rounds <N>   Stop self-healing after N rounds (default: 2)
#
# When --phase is omitted the script acts as a launcher. It computes the
# SLURM array size and submits a worker array job + a dependent aggregate job,
# both pointing back at this same script with the appropriate --phase.

# The aggregate phase scans for missing pairs (via find_missing_baseline_tasks)
# and either re-submits them (self-heal) or runs the final aggregator
# (aggregate_and_plot_seed_evaluation in baseline mode) which writes
# seed_evaluation_summary.yaml and seed_eval_curves.png.

# The aggregate phase is self-healing: if some (baseline, seed) pairs are
# still missing, it re-submits only those array indices and chains a new
# aggregate behind them.

NAME=""
STORAGE_DIR="experiment_outputs/Runs"
ENV_CONFIG="./config_files/environments/env_symmetric_3WH5SKU.yaml"
N_SEEDS=5
EVAL_EPISODES=100
EVAL_SEED=123
BASELINES=""
PHASE=""
HEAL_ROUND=0
MAX_HEAL_ROUNDS=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --name requires a value" >&2; exit 1; }
      NAME="$2"; shift 2 ;;
    --storage-dir)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --storage-dir requires a value" >&2; exit 1; }
      STORAGE_DIR="$2"; shift 2 ;;
    --n-seeds)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --n-seeds requires a value" >&2; exit 1; }
      N_SEEDS="$2"; shift 2 ;;
    --eval-episodes)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --eval-episodes requires a value" >&2; exit 1; }
      EVAL_EPISODES="$2"; shift 2 ;;
    --eval-seed)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --eval-seed requires a value" >&2; exit 1; }
      EVAL_SEED="$2"; shift 2 ;;
    --baselines)
      # Slurp space-separated baseline names until the next --flag or EOF.
      shift
      BASELINES=""
      while [[ $# -gt 0 && "$1" != --* ]]; do
        BASELINES="${BASELINES} $1"
        shift
      done
      BASELINES="$(echo $BASELINES | xargs)" ;;
    --phase)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --phase requires a value" >&2; exit 1; }
      PHASE="$2"; shift 2 ;;
    --heal-round)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --heal-round requires a value" >&2; exit 1; }
      HEAL_ROUND="$2"; shift 2 ;;
    --max-heal-rounds)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --max-heal-rounds requires a value" >&2; exit 1; }
      MAX_HEAL_ROUNDS="$2"; shift 2 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "$NAME" ]; then
  echo "ERROR: --name is required for baseline seed-evaluation." >&2
  exit 1
fi

# Default to the canonical baseline list when --baselines was not provided.
if [ -z "$BASELINES" ]; then
  BASELINES="$(python -c "
from src.experiments.run_baselines import BASELINE_NAMES
print(' '.join(BASELINE_NAMES))
")"
  if [ -z "$BASELINES" ]; then
    echo "ERROR: Failed to enumerate baselines from run_baselines.py"
    exit 1
  fi
fi

# Convert the baselines string into a bash array and compute the number of baselines
read -r -a BASELINES_ARR <<< "$BASELINES"
N_BASELINES=${#BASELINES_ARR[@]}

# Set the experiment name and directories
EXPERIMENT_NAME="${NAME}"
EXPERIMENT_DIR="${STORAGE_DIR}/${EXPERIMENT_NAME}"
SEED_DIR="${EXPERIMENT_DIR}/seed_evaluation"


# ============================================================================
# LAUNCHER: no --phase was provided and we should launch the jobs
# ============================================================================

# If no phase was provided, launch the jobs
if [ -z "$PHASE" ]; then

  # Determine the array size and the maximum concurrent tasks
  ARRAY_SIZE=$(( N_BASELINES * N_SEEDS - 1 ))
  TOTAL_TASKS=$(( ARRAY_SIZE + 1 ))
  MAX_CONCURRENT=$(( TOTAL_TASKS < 17 ? TOTAL_TASKS : 17 ))

  # Collect all arguments to forward to the worker and aggregate phases
  FORWARD_ARGS="--name ${NAME} --storage-dir ${STORAGE_DIR} --n-seeds ${N_SEEDS} --eval-episodes ${EVAL_EPISODES} --eval-seed ${EVAL_SEED} --max-heal-rounds ${MAX_HEAL_ROUNDS} --baselines ${BASELINES}"

  echo "Launching baseline seed evaluation: name=${NAME}, n_baselines=${N_BASELINES}, n_seeds=${N_SEEDS}"
  echo "Total tasks: ${TOTAL_TASKS} (${N_BASELINES} baselines x ${N_SEEDS} seeds)"
  echo "Baselines:   ${BASELINES}"

  # Submit worker array job
  WORKER_JOB=$(sbatch --parsable \
    --array=0-${ARRAY_SIZE}%${MAX_CONCURRENT} \
    "$0" --phase worker ${FORWARD_ARGS})
  echo "Workers submitted: job ${WORKER_JOB} (${TOTAL_TASKS} tasks, max ${MAX_CONCURRENT} concurrent)"

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
module load miniforge/25.11.0-0
cd /home/jakobeh/projects/marl-sc
source ~/projects/marl-sc/.venv/bin/activate

# Set the Python path, unbuffer the output, and set the Python hash seed
export PYTHONPATH="/home/jakobeh/projects/marl-sc${PYTHONPATH:+:$PYTHONPATH}"
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
  echo "Running baseline seed-eval aggregation for: ${NAME} (heal_round=${HEAL_ROUND})"

  # ---------- Step 1 ----------
  # Detect any (baseline, seed) pairs whose eval_results_final.yaml is
  # missing or malformed. Use a sentinel-prefixed line so we can grep
  # it out of import-time noise from Ray/PyTorch/etc.
  MISSING_RAW=$(python -c "
from src.experiments.run_baselines import find_missing_baseline_tasks
baselines = '${BASELINES}'.split() or None
missing = find_missing_baseline_tasks(
    storage_dir='${SEED_DIR}',
    n_seeds=${N_SEEDS},
    baselines=baselines,
)
print('__MISSING_TASKS__:' + ','.join(str(t) for t in missing))
")
  MISSING_RC=$?
  if [ $MISSING_RC -ne 0 ]; then
    echo "ERROR: Could not scan for missing baseline tasks (exit ${MISSING_RC})"
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
      MAX_CONCURRENT=$(( MISSING_COUNT < 17 ? MISSING_COUNT : 17 ))
      echo "[HEAL ${NEXT_ROUND}/${MAX_HEAL_ROUNDS}] ${MISSING_COUNT} task(s) missing: ${MISSING_STR}"

      # Build the forward arguments for the worker and aggregate jobs
      HEAL_FORWARD="--name ${NAME} --storage-dir ${STORAGE_DIR} --n-seeds ${N_SEEDS} --eval-episodes ${EVAL_EPISODES} --eval-seed ${EVAL_SEED} --max-heal-rounds ${MAX_HEAL_ROUNDS} --baselines ${BASELINES}"

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
  # Final aggregation + plotting (writes seed_evaluation_summary.yaml and
  # seed_eval_curves.png under <EXPERIMENT_DIR>/seed_evaluation/...).
  python -c "
from src.experiments.utils.seed_evaluation import aggregate_and_plot_seed_evaluation
aggregate_and_plot_seed_evaluation(mode='baseline', name='${EXPERIMENT_NAME}', storage_dir='${STORAGE_DIR}')
"
  exit $?
fi


# ============================================================================
# WORKER PHASE: each worker runs one (baseline, seed) pair
# ============================================================================

# If the phase is not worker, exit with an error
if [ "$PHASE" != "worker" ]; then
  echo "ERROR: Unknown --phase: ${PHASE}" >&2
  exit 1
fi

# --------------------------------------------------
# Map SLURM_ARRAY_TASK_ID to baseline and seed index
# --------------------------------------------------

# Use the ID of the current task to compute the baseline and seed index
ID=${SLURM_ARRAY_TASK_ID}
BASELINE_IDX=$(( ID / N_SEEDS ))
SEED_IDX=$(( ID % N_SEEDS + 1 ))

# Set the root seed as the seed index multiplied by 100 to ensure spaced out seeds
ROOT_SEED=$(( SEED_IDX * 100 ))

# Safeguard against out-of-bounds baseline index
if [ $BASELINE_IDX -ge $N_BASELINES ]; then
  echo "ERROR: BASELINE_IDX=${BASELINE_IDX} >= N_BASELINES=${N_BASELINES} (id=${ID})"
  exit 1
fi

# Set the baseline name and get its display name
BASELINE_NAME="${BASELINES_ARR[$BASELINE_IDX]}"
BASELINE_DISPLAY_NAME="$(
python - <<PY
from src.experiments.run_baselines import BASELINE_DISPLAY_NAMES
print(BASELINE_DISPLAY_NAMES["${BASELINE_NAME}"])
PY
)"

# Set the seed evaluation directory and the per-seed experiment directory
SEED_BASELINE_DIR="${SEED_DIR}/${BASELINE_DISPLAY_NAME}"
SEED_EXPERIMENT_NAME="${BASELINE_DISPLAY_NAME}_Seed${ROOT_SEED}"
SEED_EXPERIMENT_DIR="${SEED_BASELINE_DIR}/${SEED_EXPERIMENT_NAME}"

echo "Task ${ID} -> Baseline #${BASELINE_IDX} (${BASELINE_DISPLAY_NAME}), Seed #${SEED_IDX} (root_seed=${ROOT_SEED})"


# --------------------------------------------------
# Run the baseline (no visualization, fixed layout under seed_evaluation/)
# --------------------------------------------------

# We bypass the run_baselines.py CLI here because --mode single auto-nests
# results under <DisplayName>/, which would conflict with the seed-eval layout.
# Instead, we call run_single_baseline() directly with an explicit experiment_dir.
python -c "
from pathlib import Path

from src.config.loader import load_environment_config
from src.experiments.run_baselines import run_single_baseline

env_config = load_environment_config('${ENV_CONFIG}')
run_single_baseline(
    baseline='${BASELINE_NAME}',
    env_config=env_config,
    root_seed=${ROOT_SEED},
    eval_seed=${EVAL_SEED},
    eval_episodes=${EVAL_EPISODES},
    experiment_dir=Path('${SEED_EXPERIMENT_DIR}'),
)
"
