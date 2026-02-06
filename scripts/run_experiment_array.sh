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
#SBATCH --array=0-48%1                          # Array for 49 jobs (indices 0-48) with 5 jobs at once per node


##############################
# Load modules + env
##############################

module load miniforge/25.11.0-0                 # Load the Python distribution
cd /home/jakobeh/projects/marl-sc               # Change to the project directory
source ~/projects/marl-sc/.venv/bin/activate    # Activate the virtual environment

export PYTHONPATH="/home/jakobeh/projects/marl-sc${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1


##############################
# Map SLURM_ARRAY_TASK_ID -> (n_warehouses, n_skus)
##############################

# Define possible values for warehouses and SKUs
WAREHOUSE_VALUES=(3 5 7 10 12 15 20)
SKU_VALUES=(1 3 5 10 20 50 100)

# Compute the number of possible tasks
N_WHVALS=${#WAREHOUSE_VALUES[@]}   # 7
N_SKUVALS=${#SKU_VALUES[@]}         # 7
N_TASKS=$((N_WHVALS * N_SKUVALS))     # 49

# Get the warehouse and SKU values for this task
ID=${SLURM_ARRAY_TASK_ID}
SKU_IDX=$(( ID / N_WHVALS ))
WH_IDX=$(( ID % N_WHVALS ))
N_SKUS=${SKU_VALUES[$SKU_IDX]}
N_WAREHOUSES=${WAREHOUSE_VALUES[$WH_IDX]}
N_REGIONS=$N_WAREHOUSES

echo "Task $ID -> n_warehouses=$N_WAREHOUSES, n_skus=$N_SKUS (n_regions=$N_REGIONS)"

##############################
# Create temporary config + export vars for Python
##############################

# Create temporary config file
TEMP_CONFIG=$(mktemp)

# Export variables for Python
export N_WAREHOUSES
export N_SKUS
export TEMP_CONFIG


##############################
# Generate costs + write temp YAML
##############################

python - <<'PY'
import os
import math
import yaml
import numpy as np

N_WAREHOUSES = int(os.environ["N_WAREHOUSES"])
N_SKUS       = int(os.environ["N_SKUS"])
TEMP_CONFIG  = os.environ["TEMP_CONFIG"]

# -----------------------------
# Base cost structure for 3 warehouses and 5 SKUs
# -----------------------------
base_penalty = np.array([8.6, 9.5, 11.5, 13.2, 15.2], dtype=float)
base_weights = np.array([0.45, 0.71, 1.41, 2.24, 3.46], dtype=float)
base_in_fixed = np.array([
    [10.0, 11.0, 10.5],
    [ 9.0, 10.0,  9.5],
    [13.0, 12.0, 13.5],
    [15.0, 14.0, 15.5],
    [17.0, 18.0, 16.5],
], dtype=float)
base_in_var = np.array([
    [0.12, 0.15, 0.13],
    [0.14, 0.16, 0.15],
    [0.18, 0.20, 0.19],
    [0.22, 0.24, 0.23],
    [0.28, 0.30, 0.26],
], dtype=float)
base_out_fixed = np.array([
    [25.0, 35.0, 32.0],
    [42.0, 22.0, 28.0],
    [38.0, 33.0, 18.0],
], dtype=float)

base_out_var = np.array([
    [0.15, 0.32, 0.28],
    [0.42, 0.18, 0.25],
    [0.38, 0.29, 0.12],
], dtype=float)
base_dist = np.array([
    [150, 320, 280],
    [420, 180, 250],
    [380, 290, 120],
], dtype=float)

# -----------------------------
# Helpers: deterministic "variation" (no RNG needed)
# -----------------------------
def smooth_factor(i: int, j: int, amp: float = 0.06) -> float:
    """ 
    Returns a some small bounded variation in the range 1.0+/-amp.
    """
    return 1.0 + amp * math.sin(0.9*i + 1.7*j + 0.3)

def extend_monotone_from_base(base: np.ndarray, n: int) -> np.ndarray:
    """
    Preserves the increasing pattern of costs. If n<=len(base), truncates the array to n elements.
    If n>len(base), extrapolate in log-space to keep ratios sensible.
    """
    # If n<=len(base), truncate the array to n elements
    if n <= len(base):
        return base[:n].copy() # Copy the first n elements

    # If n>len(base), extrapolate in log-space to keep ratios sensible
    x0 = np.arange(len(base))
    y0 = np.log(base)
    x1 = np.arange(n)
    coef = np.polyfit(x0, y0, deg=1)
    y1 = np.polyval(coef, x1)
    out = np.exp(y1)

    # Ensure strictly increasing (numerical safety)
    for k in range(1, n):
        if out[k] <= out[k-1]:
            out[k] = out[k-1] * 1.03
    return out

def ring_distance(i: int, r: int, n: int) -> int:
    """
    Returns the distance between two points i,r in a ring topology of n points to give "close/medium/far" tiers naturally.
    """
    d = abs(i - r)
    d = min(d, n - d)
    return d

# -----------------------------
# Build SKU-level arrays
# -----------------------------
penalty_cost = extend_monotone_from_base(base_penalty, N_SKUS)
sku_weights  = extend_monotone_from_base(base_weights, N_SKUS)

# Keep penalty and weights correlated (important for meaning):
# If extrapolation makes them too "far apart", gently re-align by scaling penalty to weight growth.
# (This stays deterministic and preserves monotonicity.)
penalty_cost *= (sku_weights / sku_weights[:len(base_weights)].mean()) / (penalty_cost / base_penalty.mean())

# -----------------------------
# Build inbound cost matrices
# -----------------------------
# Create per-SKU "base level" by extending from base mean across warehouses
in_fixed_level = extend_monotone_from_base(base_in_fixed.mean(axis=1), N_SKUS)
in_var_level   = extend_monotone_from_base(base_in_var.mean(axis=1),   N_SKUS)
inbound_fixed = np.zeros((N_SKUS, N_WAREHOUSES), dtype=float)
inbound_var   = np.zeros((N_SKUS, N_WAREHOUSES), dtype=float)

# Fill the inbound cost matrices with warehouse-specific variation
for s in range(N_SKUS):
    for w in range(N_WAREHOUSES):
        # Small warehouse-specific variation
        inbound_fixed[s, w] = in_fixed_level[s] * smooth_factor(s, w, amp=0.07)
        inbound_var[s, w]   = in_var_level[s]   * smooth_factor(s, w, amp=0.05)

# -----------------------------
# Build outbound + distances and preserve "home region cheapest" structure
# -----------------------------
# Determine "close/medium/far" tiers for distances
N_REGIONS = N_WAREHOUSES
close_dist  = float(np.median(np.diag(base_dist)))          # ~150-180
medium_dist = float(np.median(base_dist))                   # ~250-320
far_dist    = float(np.max(base_dist))                      # ~420

out_fixed_close  = float(np.median(np.diag(base_out_fixed)))         # ~22
out_fixed_medium = float(np.median(base_out_fixed))                  # ~32-33
out_fixed_far    = float(np.max(base_out_fixed))                     # ~42

out_var_close  = float(np.median(np.diag(base_out_var)))             # ~0.12-0.18
out_var_medium = float(np.median(base_out_var))                      # ~0.28-0.29
out_var_far    = float(np.max(base_out_var))                         # ~0.42

distances      = np.zeros((N_WAREHOUSES, N_REGIONS), dtype=float)
outbound_fixed = np.zeros((N_WAREHOUSES, N_REGIONS), dtype=float)
outbound_var   = np.zeros((N_WAREHOUSES, N_REGIONS), dtype=float)

# Fill the outbound cost matrices with warehouse-specific variation based on the "close/medium/far" tiers
for w in range(N_WAREHOUSES):
    for r in range(N_REGIONS):
        d = ring_distance(w, r, N_WAREHOUSES)
        if d == 0:
            dist = close_dist
            of   = out_fixed_close
            ov   = out_var_close
        elif d == 1:
            dist = medium_dist
            of   = out_fixed_medium
            ov   = out_var_medium
        else:
            dist = far_dist
            of   = out_fixed_far
            ov   = out_var_far
        distances[w, r]      = dist * smooth_factor(w, r, amp=0.06)
        outbound_fixed[w, r] = of   * smooth_factor(w, r, amp=0.08)
        outbound_var[w, r]   = ov   * smooth_factor(w, r, amp=0.06)

# -----------------------------
# Write updated YAML config
# -----------------------------
with open("config_files/environments/base_env.yaml", "r") as f:
    config = yaml.safe_load(f)

env = config["environment"]
env["n_warehouses"] = N_WAREHOUSES
env["n_skus"]       = N_SKUS
env["n_regions"]    = N_REGIONS

cs = env["cost_structure"]
cs["penalty_cost"] = [float(x) for x in penalty_cost]
cs["sku_weights"]  = [float(x) for x in sku_weights]
cs["distances"]    = distances.round(3).tolist()

ship = cs["shipment_cost"]
ship["outbound_fixed"]    = outbound_fixed.round(3).tolist()
ship["outbound_variable"] = outbound_var.round(5).tolist()
ship["inbound_fixed"]     = inbound_fixed.round(3).tolist()
ship["inbound_variable"]  = inbound_var.round(5).tolist()

with open(TEMP_CONFIG, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
PY


##############################
# Start Ray explicitly
##############################

# Get the number of CPUs from Slurm
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
    --algorithm-config config_files/algorithms/mappo.yaml \
    --output-dir ./experiment_outputs \
    --wandb-project marl-sc \
    --root-seed 42


##############################
# Cleanup
##############################

rm -f "$TEMP_CONFIG"