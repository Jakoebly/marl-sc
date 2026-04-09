#!/bin/bash

# ============================================================================
# SBATCH directives
# ============================================================================

#SBATCH --job-name=marl-seed-eval              # Name of the job
#SBATCH --partition=mit_normal                 # Partition
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --ntasks-per-node=1                    # Number of tasks per node
#SBATCH --cpus-per-task=3                      # CPU cores per task
#SBATCH --mem=16G                              # Memory allocation
#SBATCH --time=12:00:00                        # Maximum walltime (hh:mm:ss)
#SBATCH --chdir=/home/jakobeh/projects/marl-sc # Working directory
#SBATCH --output=scripts/logs/%x_%A_%a.out     # Standard output
#SBATCH --error=scripts/logs/%x_%A_%a.err      # Standard error
#SBATCH --array=0-29%15                        # 10 configs x 3 runs = 30 tasks (indices 0-29), max 15 concurrent


# ============================================================================
# Parse arguments
# ============================================================================

# Usage:
#   sbatch run_seed_evaluation.sh --mode tune   --name <TuneName>   [--n-seeds 3] [--num-iterations N]
#   sbatch run_seed_evaluation.sh --mode single --name <RunName>    [--n-seeds 3] [--num-iterations N]
#   sbatch run_seed_evaluation.sh --mode aggregate --name <Name>

MODE=""
NAME=""
N_SEEDS=3
NUM_ITERATIONS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 && "$2" != -* ]] || 
        { echo "ERROR: --mode requires a value" >&2; exit 1; }
      MODE="$2"; shift 2 ;;
    --name)
      [[ $# -ge 2 && "$2" != -* ]] || 
        { echo "ERROR: --name requires a value" >&2; exit 1; }
      NAME="$2"; shift 2 ;;
    --n-seeds)
      [[ $# -ge 2 && "$2" != -* ]] || 
        { echo "ERROR: --n-seeds requires a value" >&2; exit 1; }
      [[ "$2" =~ ^[1-9][0-9]*$ ]]  || 
        { echo "ERROR: --n-seeds must be a positive integer, got: $2" >&2; exit 1; }
      N_SEEDS="$2"; shift 2 ;;
    --num-iterations)
      [[ $# -ge 2 && "$2" != -* ]] || 
        { echo "ERROR: --num-iterations requires a value" >&2; exit 1; }
      [[ "$2" =~ ^[1-9][0-9]*$ ]]  || 
        { echo "ERROR: --num-iterations must be a positive integer, got: $2" >&2; exit 1; }
      NUM_ITERATIONS="$2"; shift 2 ;;
    *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Validate that both mode and name are provided
if [ -z "$MODE" ] || [ -z "$NAME" ]; then
  echo "Usage: run_seed_evaluation.sh --mode <tune|single|aggregate> --name <name> [--n-seeds N] [--num-iterations N]"
  exit 1
fi

# Validate that mode is one of the allowed values
if [[ "$MODE" != "tune" && "$MODE" != "single" && "$MODE" != "aggregate" ]]; then
  echo "ERROR: --mode must be one of: tune, single, aggregate (got: ${MODE})"
  exit 1
fi

# Print the mode, name, number of seeds, and number of iterations
echo "MODE=${MODE}"
echo "NAME=${NAME}"
echo "N_SEEDS=${N_SEEDS}"
if [ -n "${NUM_ITERATIONS}" ]; then
  echo "NUM_ITERATIONS=${NUM_ITERATIONS}"
else
  echo "NUM_ITERATIONS=(use default)"
fi


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
# Process seed evaluation results (no Ray / SLURM array needed)
# ============================================================================

# ------------------------------------------------------------------
# Aggregate mode
# ------------------------------------------------------------------

# Aggregate seed evaluation results with inline python
if [ "$MODE" = "aggregate" ]; then

python - <<PY
import sys
from pathlib import Path
from src.experiments.utils.experiment_utils import find_experiment_dir, aggregate_seed_evaluation

name = "${NAME}"
matches = []

# Search for the experiment folder under experiment_outputs/Tuning/ and experiment_outputs/Runs/
for base_dir in ["experiment_outputs/Tuning", "experiment_outputs/Runs"]:
    if not Path(base_dir).exists():
        continue
    try:
        matches.append(find_experiment_dir(base_dir, name))
    except FileNotFoundError:
        pass
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


# If no matches are found, raise an error
if len(matches) == 0:
    print(
        f"ERROR: No experiment '{name}' found under "
        f"experiment_outputs/Tuning/ or experiment_outputs/Runs/",
        file=sys.stderr,
    )
    sys.exit(1)


# If multiple matches are found, raise an error
if len(matches) > 1:
    paths_str = "\n  ".join(str(m) for m in matches)
    print(
        f"ERROR: Multiple matches for '{name}':\n  {paths_str}",
        file=sys.stderr,
    )
    sys.exit(1)

# Get the first match and check if the seed evaluation directory exists
experiment_dir = matches[0]
seed_eval_dir = experiment_dir / "seed_evaluation"
if not seed_eval_dir.exists():
    print(
        f"ERROR: No seed_evaluation/ directory in {experiment_dir}",
        file=sys.stderr,
    )
    sys.exit(1)

# Run the aggregate_seed_evaluation function
aggregate_seed_evaluation(seed_eval_dir)
PY

  exit $?
fi


# ============================================================================
# Map SLURM_ARRAY_TASK_ID to config and seed
# ============================================================================

# Use the ID of the current task to compute the config index and seed number
ID=${SLURM_ARRAY_TASK_ID}
CONFIG_IDX=$(( ID / N_SEEDS ))
SEED_NUMBER=$(( ID % N_SEEDS + 1 ))

# ------------------------------------------------------------------
# Tune mode
# ------------------------------------------------------------------

# Map SLURM_ARRAY_TASK_ID to config and seed for tune mode
if [ "$MODE" = "tune" ]; then
  # Check if a tuning experiment with the given name exists
  EXPERIMENT_PATH="experiment_outputs/Tuning/${NAME}"
  if [ ! -d "$EXPERIMENT_PATH" ]; then
    echo "ERROR: Experiment directory not found: ${EXPERIMENT_PATH}"
    exit 1
  fi

  # Read trial_path and short_id from best_trial_results.yaml for the current config index
  PYTHON_OUTPUT=$(python - <<PY
import yaml, sys

yaml_path = "${EXPERIMENT_PATH}/best_trial_results.yaml"
with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

trials = data.get("top_k_trials", [])
idx = ${CONFIG_IDX}
if idx >= len(trials):
    print(
        f"ERROR: CONFIG_IDX={idx} exceeds top_k_trials count ({len(trials)})",
        file=sys.stderr,
    )
    sys.exit(1)

trial = trials[idx]
print(trial["trial_path"], f"{trial['short_id']}")
PY
  )
  PYTHON_EXIT=$? 

  # Check if the Python script failed or the output is empty (and exit if so)
  if [ $PYTHON_EXIT -ne 0 ] || [ -z "$PYTHON_OUTPUT" ]; then
    echo "ERROR: Failed to resolve trial path for CONFIG_IDX=${CONFIG_IDX}"
    exit 1
  fi

  # Store the output of the Python script as CONFIG_PATH and CONFIG_NAME 
  read -r CONFIG_PATH CONFIG_NAME <<< "$PYTHON_OUTPUT"

  STORAGE_DIR="${EXPERIMENT_PATH}/seed_evaluation"


# ------------------------------------------------------------------
# Single mode
# ------------------------------------------------------------------

# Map SLURM_ARRAY_TASK_ID to config and seed for single mode
elif [ "$MODE" = "single" ]; then
  # Check if the config index is 0 (and exit if not)
  if [ $CONFIG_IDX -ne 0 ]; then
    echo "ERROR: Single mode only supports 1 config (CONFIG_IDX=${CONFIG_IDX} != 0)."
    echo "  Array size should be N_SEEDS (${N_SEEDS}), got SLURM_ARRAY_TASK_ID=${ID}."
    exit 1
  fi

  # Search for the experiment folder under experiment_outputs/Runs/
  EXPERIMENT_PATH=$(python - <<PY
import sys
from src.experiments.utils.experiment_utils import find_experiment_dir

try:
    experiment_dir = find_experiment_dir("experiment_outputs/Runs", "${NAME}")
    print(str(experiment_dir))
except (FileNotFoundError, ValueError) as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PY
  )

  if [ $? -ne 0 ] || [ -z "$EXPERIMENT_PATH" ]; then
    echo "ERROR: Failed to find experiment '${NAME}' under experiment_outputs/Runs/"
    exit 1
  fi

  CONFIG_PATH="${EXPERIMENT_PATH}"
  CONFIG_NAME="${NAME}"
  STORAGE_DIR="${EXPERIMENT_PATH}/seed_evaluation"

fi

EXPERIMENT_NAME="${CONFIG_NAME}_seed${SEED_NUMBER}"

echo "Task $ID -> Config #${CONFIG_IDX} (${CONFIG_NAME}), Seed #${SEED_NUMBER}"
echo "  Config path: ${CONFIG_PATH}"
echo "  Storage dir:   ${STORAGE_DIR}"
echo "  Experiment:    ${EXPERIMENT_NAME}"


# ============================================================================
# Create temporary configs with overrides
# ============================================================================

# Create temporary config files 
TEMP_ENV_CONFIG=$(mktemp --suffix=.yaml)
TEMP_ALGO_CONFIG=$(mktemp --suffix=.yaml)

# Load the trial's environment and algorithm configs and override the num_iterations
python - <<PY
import yaml, sys
from pathlib import Path

config_path = Path("${CONFIG_PATH}")
env_cfg_path = config_path / "env_config.yaml"
algo_cfg_path = config_path / "algorithm_config.yaml"

if not env_cfg_path.exists():
    print(f"ERROR: {env_cfg_path} not found", file=sys.stderr)
    sys.exit(1)
if not algo_cfg_path.exists():
    print(f"ERROR: {algo_cfg_path} not found", file=sys.stderr)
    sys.exit(1)

with open(env_cfg_path, "r") as f:
    env_cfg = yaml.safe_load(f)
with open("${TEMP_ENV_CONFIG}", "w") as f:
    yaml.safe_dump(env_cfg, f, default_flow_style=False, sort_keys=False)

with open(algo_cfg_path, "r") as f:
    algo_cfg = yaml.safe_load(f)

num_iterations = "${NUM_ITERATIONS}".strip()
if num_iterations:
    algo_cfg["algorithm"]["shared"]["num_iterations"] = int(num_iterations)

with open("${TEMP_ALGO_CONFIG}", "w") as f:
    yaml.safe_dump(algo_cfg, f, default_flow_style=False, sort_keys=False)
PY

# Check if the Python script failed and exit if so
if [ $? -ne 0 ]; then
  echo "ERROR: Failed to create temporary config files"
  rm -f "$TEMP_ENV_CONFIG" "$TEMP_ALGO_CONFIG"
  exit 1
fi


# ============================================================================
# Start Ray explicitly (with port allocation)
# ============================================================================

# Make Ray not accidentally attach somewhere else
unset RAY_ADDRESS

# Get the number of CPUs from Slurm
CPUS=${SLURM_CPUS_PER_TASK:-1}

# Get the memory from Slurm
RAY_MEMORY_BYTES=$(( (${SLURM_MEM_PER_NODE:-32768} - 2048) * 1024 * 1024 ))

# Determine array size and number of tasks
MAX_TASK_ID=${SLURM_ARRAY_TASK_MAX:-${SLURM_ARRAY_TASK_ID}}
N_TASKS=$((MAX_TASK_ID + 1))

# Define the port range
BASE_PORT=20000
MAX_PORT=65535
AVAILABLE=$((MAX_PORT - BASE_PORT + 1))

# Define the preferred block size (i.e., number of ports) per task
PREFERRED_BLOCK_SIZE=200

# Reserve the first 20 ports in the block for fixed components and leave the rest for workers
RESERVED_WITHIN_BLOCK=20

# Define the minimum block size such that the worker range is non-empty
MIN_BLOCK_SIZE=$((RESERVED_WITHIN_BLOCK + 1))

# Cap the preferred block size to the maximum possible if necessary
BLOCK_SIZE=$PREFERRED_BLOCK_SIZE
MAX_BLOCK_SIZE=$((AVAILABLE / N_TASKS))
if [ $MAX_BLOCK_SIZE -lt $BLOCK_SIZE ]; then
  BLOCK_SIZE=$MAX_BLOCK_SIZE
fi

# Sanity check if the block size is too small
if [ $BLOCK_SIZE -lt $MIN_BLOCK_SIZE ]; then
  echo "ERROR: Array too large to allocate ports safely."
  echo "N_TASKS=${N_TASKS}, AVAILABLE_PORTS=${AVAILABLE}, computed BLOCK_SIZE=${BLOCK_SIZE} (< ${MIN_BLOCK_SIZE})"
  exit 1
fi

# Define the job width (i.e., number of ports per job)
ARRAY_WIDTH=$((N_TASKS * BLOCK_SIZE)) # total number of ports for all jobs in the array
ARRAY_SLOTS=$((AVAILABLE / ARRAY_WIDTH)) # number of arrays with the same size as the current array that can fit in the available port range

# Sanity check if the number of array slots is at least 1
if [ $ARRAY_SLOTS -lt 1 ]; then
  echo "ERROR: Not enough port space for even one array slot."
  exit 1
fi

# Get the array slot for the current array
ARRAY_SLOT=$((SLURM_JOB_ID % ARRAY_SLOTS))

TASK_ID=${SLURM_ARRAY_TASK_ID}
P=$((BASE_PORT + ARRAY_SLOT * ARRAY_WIDTH + TASK_ID * BLOCK_SIZE))

# Set the ports for the Ray components of the current task
RAY_GCS_PORT=$((P + 0))
RAY_NODE_MANAGER_PORT=$((P + 1))
RAY_OBJECT_MANAGER_PORT=$((P + 2))
RAY_MIN_WORKER_PORT=$((P + RESERVED_WITHIN_BLOCK))
RAY_MAX_WORKER_PORT=$((P + BLOCK_SIZE - 1))

# Sanity check if the maximum worker port is within the available port range
if [ $RAY_MAX_WORKER_PORT -gt $MAX_PORT ]; then
  echo "ERROR: Port calculation overflowed: ${RAY_MAX_WORKER_PORT} > ${MAX_PORT}"
  exit 1
fi

# Per-task Ray temp dir to avoid session/state collisions on shared nodes
RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RAY_TMPDIR"

# Cleanup function for temp config files
cleanup() {
  rm -f "$TEMP_ENV_CONFIG" "$TEMP_ALGO_CONFIG" >/dev/null 2>&1 || true
  rm -rf "$RAY_TMPDIR" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Force the current task's driver to connect to the current task's head
export RAY_ADDRESS="127.0.0.1:${RAY_GCS_PORT}"

# Start Ray explicitly with ports and number of CPUs
ray start --head \
  --port="${RAY_GCS_PORT}" \
  --node-manager-port="${RAY_NODE_MANAGER_PORT}" \
  --object-manager-port="${RAY_OBJECT_MANAGER_PORT}" \
  --min-worker-port="${RAY_MIN_WORKER_PORT}" \
  --max-worker-port="${RAY_MAX_WORKER_PORT}" \
  --num-cpus="${CPUS}" \
  --memory="${RAY_MEMORY_BYTES}" \
  --temp-dir="${RAY_TMPDIR}" \
  --include-dashboard=false \
  --disable-usage-stats \
  || { echo "ERROR: ray start failed"; exit 1; }
echo "Ray started successfully"


# ============================================================================
# Run training + evaluation
# ============================================================================

# Set the root seed as a deterministic but well-separated number (e.g., 100, 200, 300, ...)
ROOT_SEED=$(( SEED_NUMBER * 100 ))

# Run training
python src/experiments/run_experiment.py \
    --mode single \
    --env-config "$TEMP_ENV_CONFIG" \
    --algorithm-config "$TEMP_ALGO_CONFIG" \
    --storage-dir "${STORAGE_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --wandb-project marl-sc \
    --root-seed ${ROOT_SEED}

# Run evaluation on the trained checkpoint
python src/experiments/run_experiment.py \
    --mode evaluate \
    --storage-dir "${STORAGE_DIR}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --visualize \
    --eval-episodes 100 \
    --root-seed ${ROOT_SEED}


