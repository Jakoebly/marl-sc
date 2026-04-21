#!/bin/bash
# ============================================================================
# Ray startup helper with deterministic per-task port allocation for SLURM.
#
# MUST be sourced, not executed:
#   source scripts/lib/start_ray.sh
#
# Executing it as a subprocess would lose the `trap` and `export RAY_ADDRESS`
# because both only affect the current shell.
#
# Optional knobs (set before sourcing; defaults in parentheses):
#   RAY_PREFERRED_BLOCK_SIZE   Ports per SLURM array task (200)
#   RAY_RESERVED_WITHIN_BLOCK  First N ports in block reserved for
#                              GCS/node-mgr/etc., rest are worker ports (20)
#   RAY_BASE_PORT              Lowest port to allocate from (20000)
#   RAY_MAX_PORT               Highest port to allocate up to (65535)
#   RAY_MEMORY_RESERVE_MB      MB held back for OS/overhead when computing
#                              --memory from SLURM_MEM_PER_NODE (2048)
#   RAY_FALLBACK_MEM_MB        Fallback for SLURM_MEM_PER_NODE if unset (16384)
#   RAY_EXTRA_CLEANUP_PATHS    Space-separated paths passed to `rm -rf` on
#                              EXIT in addition to the per-task Ray temp
#                              dir (empty)
#
# Sets/exports in the calling shell:
#   RAY_ADDRESS                127.0.0.1:<gcs_port>
#   RAY_TMPDIR                 Per-task Ray session dir
#   RAY_GCS_PORT, RAY_NODE_MANAGER_PORT, RAY_OBJECT_MANAGER_PORT,
#   RAY_MIN_WORKER_PORT, RAY_MAX_WORKER_PORT,
#   RAY_METRICS_EXPORT_PORT, RAY_DASHBOARD_AGENT_GRPC_PORT,
#   RAY_DASHBOARD_AGENT_HTTP_PORT, RAY_RUNTIME_ENV_AGENT_PORT
# ============================================================================

# Guard: if this file was executed instead of sourced, bail out with a hint.
# $0 is the invoked script path when executed; when sourced it's the parent.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "ERROR: start_ray.sh must be sourced, not executed." >&2
  echo "  Use: source scripts/lib/start_ray.sh" >&2
  exit 1
fi

# Apply defaults for any knob the caller did not set
: "${RAY_PREFERRED_BLOCK_SIZE:=200}"
: "${RAY_RESERVED_WITHIN_BLOCK:=20}"
: "${RAY_BASE_PORT:=20000}"
: "${RAY_MAX_PORT:=65535}"
: "${RAY_MEMORY_RESERVE_MB:=2048}"
: "${RAY_FALLBACK_MEM_MB:=16384}"
: "${RAY_EXTRA_CLEANUP_PATHS:=}"

# Make Ray not accidentally attach somewhere else
unset RAY_ADDRESS

# Get the number of CPUs from Slurm
CPUS=${SLURM_CPUS_PER_TASK:-1}

# Get the memory from Slurm
RAY_MEMORY_BYTES=$(( (${SLURM_MEM_PER_NODE:-$RAY_FALLBACK_MEM_MB} - RAY_MEMORY_RESERVE_MB) * 1024 * 1024 ))

# Define the port range
AVAILABLE=$((RAY_MAX_PORT - RAY_BASE_PORT + 1))

# Determine array size and number of tasks (single job: N_TASKS=1)
MAX_TASK_ID=${SLURM_ARRAY_TASK_MAX:-${SLURM_ARRAY_TASK_ID:-0}}
N_TASKS=$((MAX_TASK_ID + 1))

# Define the minimum block size such that the worker range is non-empty
MIN_BLOCK_SIZE=$((RAY_RESERVED_WITHIN_BLOCK + 1))

# Cap the preferred block size to the maximum possible if necessary
BLOCK_SIZE=$RAY_PREFERRED_BLOCK_SIZE
MAX_BLOCK_SIZE=$((AVAILABLE / N_TASKS))
if [ $MAX_BLOCK_SIZE -lt $BLOCK_SIZE ]; then
  BLOCK_SIZE=$MAX_BLOCK_SIZE
fi

# Sanity check if the block size is too small
if [ $BLOCK_SIZE -lt $MIN_BLOCK_SIZE ]; then
  echo "ERROR: Array too large to allocate ports safely."
  echo "N_TASKS=${N_TASKS}, AVAILABLE_PORTS=${AVAILABLE}, computed BLOCK_SIZE=${BLOCK_SIZE} (< ${MIN_BLOCK_SIZE})"
  echo "Fix: reduce array concurrency per node (e.g., use --exclusive) or lower RAY_BASE_PORT / change policy."
  return 1 2>/dev/null || exit 1
fi

# Define the job width (i.e., number of ports per job)
ARRAY_WIDTH=$((N_TASKS * BLOCK_SIZE)) # total number of ports for all jobs in the array
ARRAY_SLOTS=$((AVAILABLE / ARRAY_WIDTH)) # number of arrays with the same size as the current array that can fit in the available port range

# Sanity check if the number of array slots is at least 1
if [ $ARRAY_SLOTS -lt 1 ]; then
  echo "ERROR: Not enough port space for even one array slot (this should not happen if BLOCK_SIZE check passed)."
  return 1 2>/dev/null || exit 1
fi

# Get the array slot for the current job
ARRAY_SLOT=$((SLURM_JOB_ID % ARRAY_SLOTS))

# Compute the first port for the current task
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
P=$((RAY_BASE_PORT + ARRAY_SLOT * ARRAY_WIDTH + TASK_ID * BLOCK_SIZE))

# Set the ports for the Ray components of the current task
export RAY_GCS_PORT=$((P + 0))
export RAY_NODE_MANAGER_PORT=$((P + 1))
export RAY_OBJECT_MANAGER_PORT=$((P + 2))
export RAY_METRICS_EXPORT_PORT=$((P + 3))
export RAY_DASHBOARD_AGENT_GRPC_PORT=$((P + 4))
export RAY_DASHBOARD_AGENT_HTTP_PORT=$((P + 5))
export RAY_RUNTIME_ENV_AGENT_PORT=$((P + 6))
export RAY_MIN_WORKER_PORT=$((P + RAY_RESERVED_WITHIN_BLOCK))
export RAY_MAX_WORKER_PORT=$((P + BLOCK_SIZE - 1))

# Sanity check if the maximum worker port is within the available port range
if [ $RAY_MAX_WORKER_PORT -gt $RAY_MAX_PORT ]; then
  echo "ERROR: Port calculation overflowed: ${RAY_MAX_WORKER_PORT} > ${RAY_MAX_PORT}"
  return 1 2>/dev/null || exit 1
fi

# Per-task Ray temp dir to avoid session/state collisions on shared nodes
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}_${TASK_ID}"
mkdir -p "$RAY_TMPDIR"

# Cleanup function for Ray temp dir and any caller-supplied extra paths.
# If the caller already has a cleanup function registered on EXIT we
# preserve and chain it so user-owned temp files are still removed.
_start_ray_prev_exit_trap=$(trap -p EXIT | sed -n "s/^trap -- '\\(.*\\)' EXIT$/\\1/p")
_start_ray_cleanup() {
  if [ -n "$_start_ray_prev_exit_trap" ]; then
    eval "$_start_ray_prev_exit_trap" || true
  fi
  if [ -n "$RAY_EXTRA_CLEANUP_PATHS" ]; then
    rm -rf $RAY_EXTRA_CLEANUP_PATHS >/dev/null 2>&1 || true
  fi
  rm -rf "$RAY_TMPDIR" >/dev/null 2>&1 || true
}
trap _start_ray_cleanup EXIT

# Force the current task's driver to connect to the current task's head
export RAY_ADDRESS="127.0.0.1:${RAY_GCS_PORT}"

# Disable Ray's application-level OOM killer to prevent the kill-restart
# death spiral; rely on SLURM's cgroup memory enforcement instead
export RAY_memory_monitor_refresh_ms=0
export PYTHONWARNINGS="ignore::DeprecationWarning"

# Give Ray more time to start up to avoid premature termination due to heavy loads
export RAY_raylet_start_wait_time_s=300
sleep $(( RANDOM % 45 ))

# Start Ray explicitly with ports, CPUs, and memory
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
  --metrics-export-port="${RAY_METRICS_EXPORT_PORT}" \
  --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
  --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_HTTP_PORT}" \
  --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}" \
  || { echo "ERROR: ray start failed"; return 1 2>/dev/null || exit 1; }
echo "Ray started successfully (GCS port ${RAY_GCS_PORT}, worker ports ${RAY_MIN_WORKER_PORT}-${RAY_MAX_WORKER_PORT})"
