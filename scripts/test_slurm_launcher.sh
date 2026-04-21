#!/bin/bash

# ============================================================================
# Lightweight test for the launcher → worker array → aggregate pattern.
#
# Usage (run directly on login node — NOT via sbatch):
#   ./scripts/test_slurm_launcher.sh --n-seeds 3 --n-configs 2
#
# This submits a SLURM array of (n_configs * n_seeds) trivial worker tasks
# that each just sleep a few seconds and write a file, then a dependent
# aggregate task that reads them all. Check scripts/logs/ for output.
# ============================================================================

#SBATCH --job-name=test-launcher
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/jakobeh/projects/marl-sc
#SBATCH --output=scripts/logs/%x_%A_%a.out
#SBATCH --error=scripts/logs/%x_%A_%a.err


# ============================================================================
# Parse arguments
# ============================================================================

N_SEEDS=3
N_CONFIGS=2
PHASE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-seeds)   N_SEEDS="$2"; shift 2 ;;
    --n-configs) N_CONFIGS="$2"; shift 2 ;;
    --phase)     PHASE="$2"; shift 2 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
  esac
done


# ============================================================================
# LAUNCHER — no --phase means we are on the login node
# ============================================================================

if [ -z "$PHASE" ]; then
  ARRAY_SIZE=$(( N_CONFIGS * N_SEEDS - 1 ))

  echo "=== TEST LAUNCHER ==="
  echo "  n_configs=${N_CONFIGS}, n_seeds=${N_SEEDS}"
  echo "  Total tasks: $(( ARRAY_SIZE + 1 ))"

  mkdir -p scripts/logs
  WORK_DIR=$(mktemp -d /tmp/test_launcher_XXXXXX)
  echo "  Work dir: ${WORK_DIR}"

  FORWARD="--n-seeds ${N_SEEDS} --n-configs ${N_CONFIGS}"

  WORKER_JOB=$(sbatch --parsable \
    --array=0-${ARRAY_SIZE} \
    --export=ALL,WORK_DIR="${WORK_DIR}" \
    "$0" --phase worker ${FORWARD})
  echo "  Workers submitted: job ${WORKER_JOB}"

  AGG_JOB=$(sbatch --parsable \
    --dependency=afterok:${WORKER_JOB} \
    --export=ALL,WORK_DIR="${WORK_DIR}" \
    --cpus-per-task=1 --mem=1G --time=00:05:00 \
    "$0" --phase aggregate ${FORWARD})
  echo "  Aggregate submitted: job ${AGG_JOB} (depends on ${WORKER_JOB})"

  echo "=== Done — monitor with: squeue --me ==="
  exit 0
fi


# ============================================================================
# PHASE: worker
# ============================================================================

if [ "$PHASE" = "worker" ]; then
  ID=${SLURM_ARRAY_TASK_ID}
  CONFIG_IDX=$(( ID / N_SEEDS ))
  SEED_IDX=$(( ID % N_SEEDS + 1 ))
  ROOT_SEED=$(( SEED_IDX * 100 ))

  echo "[WORKER ${ID}] config=${CONFIG_IDX} seed=${SEED_IDX} root_seed=${ROOT_SEED}"
  echo "  Hostname: $(hostname)"
  echo "  SLURM_JOB_ID: ${SLURM_JOB_ID}"
  echo "  WORK_DIR: ${WORK_DIR}"

  sleep $(( RANDOM % 5 + 2 ))

  mkdir -p "${WORK_DIR}"
  echo "config=${CONFIG_IDX} seed=${ROOT_SEED} mean_return=0.$(( RANDOM % 1000 ))" \
    > "${WORK_DIR}/result_${ID}.txt"

  echo "[WORKER ${ID}] Done."
  exit 0
fi


# ============================================================================
# PHASE: aggregate
# ============================================================================

if [ "$PHASE" = "aggregate" ]; then
  echo "[AGGREGATE] Collecting results from: ${WORK_DIR}"
  echo ""

  if [ ! -d "${WORK_DIR}" ]; then
    echo "ERROR: WORK_DIR does not exist: ${WORK_DIR}"
    exit 1
  fi

  echo "--- All worker results ---"
  for f in "${WORK_DIR}"/result_*.txt; do
    echo "  $(cat "$f")"
  done

  TOTAL=$(ls -1 "${WORK_DIR}"/result_*.txt 2>/dev/null | wc -l)
  EXPECTED=$(( N_CONFIGS * N_SEEDS ))
  echo ""
  echo "Found ${TOTAL} / ${EXPECTED} result files."

  if [ "$TOTAL" -eq "$EXPECTED" ]; then
    echo "SUCCESS: All workers completed."
  else
    echo "WARNING: Some workers may have failed."
  fi

  rm -rf "${WORK_DIR}"
  echo "[AGGREGATE] Done."
  exit 0
fi

echo "ERROR: Unknown --phase: ${PHASE}" >&2
exit 1
