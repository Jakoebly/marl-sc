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
#   ./scripts/run_baselines.sh --name <Name> --env-config <path> [opts]
#
# Options:
#   --name <Name>           Experiment / output folder name (required).
#                           Convention: BASELINE_<n_warehouses>WH<n_skus>SKU_<env_class>
#   --env-config <path>     Environment config YAML (required)
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
# SLURM array size (N_BASELINES * N_SEEDS) and submits a worker array job
# plus a dependent aggregate job, both pointing back at this same script
# with the appropriate --phase.
#
# Each worker runs ONE (baseline, root_seed) pair via
# `run_baselines.py --mode single`, writing:
#   <storage-dir>/<name>/seed_evaluation/<DisplayName>_Seed<root_seed>/eval_results_best.yaml
#
# The aggregate phase scans for missing pairs (via find_missing_baseline_tasks)
# and either re-submits them (self-heal) or runs the final aggregator
# (aggregate_seed_evaluation) which writes seed_evaluation_summary.yaml.

NAME=""
ENV_CONFIG=""
STORAGE_DIR="experiment_outputs/Runs"
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
    --env-config)
      [[ $# -ge 2 && "$2" != -* ]] || { echo "ERROR: --env-config requires a value" >&2; exit 1; }
      ENV_CONFIG="$2"; shift 2 ;;
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
  echo "Usage: ./scripts/run_baselines.sh --name <Name> --env-config <path> [options]"
  exit 1
fi
if [ -z "$ENV_CONFIG" ]; then
  echo "ERROR: --env-config is required"
  exit 1
fi


# ============================================================================
# Resolve seed-eval directory (shared by launcher / worker / aggregate)
# ============================================================================

# The seed-eval directory is always derived from --storage-dir and --name so
# that baseline outputs land in their own BASELINE_* folder, structurally
# identical to (but separate from) RL run folders.
SEED_EVAL_DIR="${STORAGE_DIR}/${NAME}/seed_evaluation"


# ============================================================================
# Resolve baselines and array size
# ============================================================================

# Default to the canonical baseline list when --baselines was not provided.
# Imported directly from the registry in run_baselines.py so the .sh and
# .py stay in sync.
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

# Convert the baselines string into a bash array for index lookups in the
# worker phase. This array MUST stay aligned with how run_baselines.py
# registers baselines and with find_missing_baseline_tasks's task-id mapping.
read -r -a BASELINES_ARR <<< "$BASELINES"
N_BASELINES=${#BASELINES_ARR[@]}


# ============================================================================
# LAUNCHER: no --phase was provided and we should launch the jobs
# ============================================================================

if [ -z "$PHASE" ]; then

  # Compute the array size
  ARRAY_SIZE=$(( N_BASELINES * N_SEEDS - 1 ))
  TOTAL_TASKS=$(( ARRAY_SIZE + 1 ))
  MAX_CONCURRENT=$(( TOTAL_TASKS < 17 ? TOTAL_TASKS : 17 ))

  # Collect all arguments to forward to the worker and aggregate phases
  FORWARD_ARGS="--name ${NAME} --env-config ${ENV_CONFIG} --storage-dir ${STORAGE_DIR} --n-seeds ${N_SEEDS} --eval-episodes ${EVAL_EPISODES} --eval-seed ${EVAL_SEED} --max-heal-rounds ${MAX_HEAL_ROUNDS} --baselines ${BASELINES}"

  echo "Launching baseline seed-eval: name=${NAME}, n_baselines=${N_BASELINES}, n_seeds=${N_SEEDS}"
  echo "Total tasks: ${TOTAL_TASKS} (${N_BASELINES} baselines x ${N_SEEDS} seeds)"
  echo "Baselines:   ${BASELINES}"
  echo "Output dir:  ${SEED_EVAL_DIR}"

  # Submit worker array job
  WORKER_JOB=$(sbatch --parsable \
    --array=0-${ARRAY_SIZE}%${MAX_CONCURRENT} \
    "$0" --phase worker ${FORWARD_ARGS})
  echo "Workers submitted: job ${WORKER_JOB} (${TOTAL_TASKS} tasks, max ${MAX_CONCURRENT} concurrent)"

  # Aggregate depends on 'afterany' (not 'afterok') so a single failed worker
  # does not stall the pipeline; the aggregate phase is self-healing and
  # re-submits only the unfinished (baseline, seed) pairs.
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

module load miniforge/25.11.0-0
cd /home/jakobeh/projects/marl-sc
source ~/projects/marl-sc/.venv/bin/activate

export PYTHONPATH="/home/jakobeh/projects/marl-sc${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

# Pin thread pools to the cores SLURM granted us so two co-tenanted tasks
# do not oversubscribe the node and slow each other down.
CPUS_THREAD_PIN=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS="${CPUS_THREAD_PIN}"
export MKL_NUM_THREADS="${CPUS_THREAD_PIN}"
export OPENBLAS_NUM_THREADS="${CPUS_THREAD_PIN}"
export NUMEXPR_NUM_THREADS="${CPUS_THREAD_PIN}"


# ============================================================================
# AGGREGATE PHASE: aggregate the results (self-healing)
# ============================================================================

if [ "$PHASE" = "aggregate" ]; then
  echo "Running baseline seed-eval aggregation: name=${NAME}, heal_round=${HEAL_ROUND}"

  # ---------- Step 1 ----------
  # Detect any (baseline, seed) pairs whose eval_results_best.yaml is
  # missing or malformed. Use a sentinel-prefixed line so we can grep
  # it out of import-time noise from Ray/PyTorch/etc.
  MISSING_RAW=$(python -c "
from src.experiments.run_baselines import find_missing_baseline_tasks
baselines = '${BASELINES}'.split() or None
missing = find_missing_baseline_tasks(
    seed_eval_dir='${SEED_EVAL_DIR}',
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
  # If work is still pending and we have heal budget left, re-submit JUST
  # the missing array indices and chain a new aggregate behind them.
  if [ -n "$MISSING_STR" ]; then
    MISSING_COUNT=$(echo "$MISSING_STR" | awk -F, '{print NF}')
    if [ "$HEAL_ROUND" -lt "$MAX_HEAL_ROUNDS" ]; then
      NEXT_ROUND=$(( HEAL_ROUND + 1 ))
      MAX_CONCURRENT=$(( MISSING_COUNT < 17 ? MISSING_COUNT : 17 ))
      echo "[HEAL ${NEXT_ROUND}/${MAX_HEAL_ROUNDS}] ${MISSING_COUNT} task(s) missing: ${MISSING_STR}"

      HEAL_FORWARD="--name ${NAME} --env-config ${ENV_CONFIG} --storage-dir ${STORAGE_DIR} --n-seeds ${N_SEEDS} --eval-episodes ${EVAL_EPISODES} --eval-seed ${EVAL_SEED} --max-heal-rounds ${MAX_HEAL_ROUNDS} --baselines ${BASELINES}"

      HEAL_WORKER_JOB=$(sbatch --parsable \
        --array=${MISSING_STR}%${MAX_CONCURRENT} \
        "$0" --phase worker ${HEAL_FORWARD})
      echo "[HEAL] Re-submitted workers: job ${HEAL_WORKER_JOB}"

      HEAL_AGG_JOB=$(sbatch --parsable \
        --dependency=afterany:${HEAL_WORKER_JOB} \
        --cpus-per-task=1 --mem=4G --time=00:30:00 \
        "$0" --phase aggregate ${HEAL_FORWARD} --heal-round ${NEXT_ROUND})
      echo "[HEAL] Chained aggregate: job ${HEAL_AGG_JOB}"

      # Exit 0 so SLURM does not flag the chain as failed; the next
      # aggregate job will perform the final aggregation.
      exit 0
    else
      echo "[HEAL] Exhausted ${MAX_HEAL_ROUNDS} heal rounds but ${MISSING_COUNT} task(s) still missing: ${MISSING_STR}"
      echo "[HEAL] Proceeding to aggregate with incomplete results."
    fi
  fi

  # ---------- Step 3 ----------
  # Final aggregation: produces seed_evaluation_summary.yaml in SEED_EVAL_DIR.
  python -c "
from src.experiments.utils.seed_evaluation import aggregate_seed_evaluation
aggregate_seed_evaluation('${SEED_EVAL_DIR}')
"
  exit $?
fi


# ============================================================================
# WORKER PHASE: each worker runs one (baseline, seed) pair
# ============================================================================

if [ "$PHASE" != "worker" ]; then
  echo "ERROR: Unknown --phase: ${PHASE}" >&2
  exit 1
fi

# --------------------------------------------------
# Map SLURM_ARRAY_TASK_ID to (baseline, seed)
# --------------------------------------------------

# Mirrors baseline_task_layout() in run_baselines.py:
#   task_id    = baseline_idx * n_seeds + (seed_idx - 1)
#   seed_idx   = task_id % n_seeds + 1
#   root_seed  = seed_idx * 100
ID=${SLURM_ARRAY_TASK_ID}
BASELINE_IDX=$(( ID / N_SEEDS ))
SEED_IDX=$(( ID % N_SEEDS + 1 ))
ROOT_SEED=$(( SEED_IDX * 100 ))

if [ $BASELINE_IDX -ge $N_BASELINES ]; then
  echo "ERROR: BASELINE_IDX=${BASELINE_IDX} >= N_BASELINES=${N_BASELINES} (id=${ID})"
  exit 1
fi

BASELINE_NAME="${BASELINES_ARR[$BASELINE_IDX]}"

echo "Task ${ID} -> Baseline #${BASELINE_IDX} (${BASELINE_NAME}), Seed #${SEED_IDX} (root_seed=${ROOT_SEED})"
echo "  env_config:   ${ENV_CONFIG}"
echo "  seed_eval:    ${SEED_EVAL_DIR}"
echo "  eval_seed:    ${EVAL_SEED}"
echo "  eval_eps:     ${EVAL_EPISODES}"

# --------------------------------------------------
# Run the worker
# --------------------------------------------------

python src/experiments/run_baselines.py \
  --mode single \
  --baseline "${BASELINE_NAME}" \
  --root-seed ${ROOT_SEED} \
  --env-config "${ENV_CONFIG}" \
  --storage-dir "${STORAGE_DIR}" \
  --experiment-name "${NAME}" \
  --eval-seed ${EVAL_SEED} \
  --eval-episodes ${EVAL_EPISODES}
