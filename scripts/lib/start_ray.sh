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
#   RAY_OBJECT_STORE_FRACTION  Fraction of --memory used for plasma object
#                              store. Ray's default is 0.3 of total node RAM,
#                              which routinely exceeds SLURM cgroup limits
#                              and triggers OOM. (0.25)
#   RAY_MAX_PORT_RETRIES       Number of retries with shifted ports if
#                              `ray start` fails due to collisions (5)
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


# ============================================================================
# Parse and defaults
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
: "${RAY_OBJECT_STORE_FRACTION:=0.25}"
: "${RAY_MAX_PORT_RETRIES:=5}"
: "${RAY_EXTRA_CLEANUP_PATHS:=}"


# ============================================================================
# Compute port allocation parameters
# ============================================================================

# Make Ray not accidentally attach somewhere else
unset RAY_ADDRESS

# Get the number of CPUs from Slurm
CPUS=${SLURM_CPUS_PER_TASK:-1}

# Compute --memory from SLURM_MEM_PER_NODE (minus reserve) so Ray's logical
# memory accounting matches what the cgroup actually allows.
RAY_MEM_MB_TOTAL=${SLURM_MEM_PER_NODE:-$RAY_FALLBACK_MEM_MB}
RAY_MEMORY_BYTES=$(( (RAY_MEM_MB_TOTAL - RAY_MEMORY_RESERVE_MB) * 1024 * 1024 ))

# Explicitly size the plasma object store as a fraction of --memory.
# Ray's default (DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.3 of total node RAM)
# is computed from /proc/meminfo (or the root cgroup), not the SLURM cgroup,
# and routinely exceeds the per-task cgroup limit.
RAY_OBJECT_STORE_MEMORY_BYTES=$(
  awk -v m="$RAY_MEMORY_BYTES" -v f="$RAY_OBJECT_STORE_FRACTION" \
    'BEGIN { printf "%d", m * f }'
)

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
ARRAY_SLOTS=$((AVAILABLE / ARRAY_WIDTH)) # number of arrays that can fit in the port range

# Sanity check if the number of array slots is at least 1
if [ $ARRAY_SLOTS -lt 1 ]; then
  echo "ERROR: Not enough port space for even one array slot (this should not happen if BLOCK_SIZE check passed)."
  return 1 2>/dev/null || exit 1
fi

# Derive the array slot from SLURM_ARRAY_JOB_ID 
SEED_JOB_ID=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}
ARRAY_SLOT=$((SEED_JOB_ID % ARRAY_SLOTS))


# ============================================================================
# Set Ray temp directory and cleanup function
# ============================================================================

# Create per-task Ray temp dir by prefering node-local $SLURM_TMPDIR when 
# available so plasma's shared-memory-backed files don't count against the 
# cgroup RAM limit via /dev/shm. Falls back to /tmp otherwise.
RAY_TMPDIR_BASE="${SLURM_TMPDIR:-/tmp}"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
export RAY_TMPDIR="${RAY_TMPDIR_BASE}/ray_${SEED_JOB_ID}_${TASK_ID}"
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


# ============================================================================
# Set Ray environment variables to avoid OOM issues
# ============================================================================

# Disable Ray's application-level OOM killer to prevent the kill-restart
# death spiral (source: ray's memory_monitor.cc — refresh_ms=0 disables it);
# rely on SLURM's cgroup memory enforcement instead.
export RAY_memory_monitor_refresh_ms=0
export PYTHONWARNINGS="ignore::DeprecationWarning"

# Give Ray more time to start up to avoid premature termination due to heavy loads
export RAY_raylet_start_wait_time_s=300

# Small random jitter to de-correlate simultaneous array-task starts on the
# same node (reduces probe/port-bind races).
sleep $(( RANDOM % 45 ))


# ============================================================================
# Port allocation helper functions
# ============================================================================

# Probe all Ray service ports and kill leftovers from prior
# failed runs. If any probed port cannot be freed, shift the whole task
# block by ARRAY_WIDTH onto the next array slot and retry up to
# RAY_MAX_PORT_RETRIES times.
_probe_and_kill_port() {
  local port=$1
  if lsof -ti ":${port}" >/dev/null 2>&1; then
    local pids
    pids=$(lsof -ti ":${port}" 2>/dev/null || true)
    if [ -n "$pids" ]; then
      local cmd
      cmd=$(ps -o comm= -p $pids 2>/dev/null | tr '\n' ' ')
      if echo "$cmd" | grep -qiE 'ray|raylet|gcs_server|plasma|python'; then
        echo "[WARN] Port ${port} in use by leftover ${cmd}; killing pids ${pids}"
        echo "$pids" | xargs -r kill -9 2>/dev/null || true
        sleep 1
      else
        echo "[WARN] Port ${port} in use by non-ray process (${cmd}); will try port shift"
        return 1
      fi
    fi
  fi
  return 0
}


# Assign ports for a given slot. If the port range overflows 
# (RAY_MAX_WORKER_PORT > RAY_MAX_PORT), returns 2 to indicate failure.
_assign_ports_for_slot() {
  local slot=$1
  local P=$((RAY_BASE_PORT + slot * ARRAY_WIDTH + TASK_ID * BLOCK_SIZE))
  export RAY_GCS_PORT=$((P + 0))
  export RAY_NODE_MANAGER_PORT=$((P + 1))
  export RAY_OBJECT_MANAGER_PORT=$((P + 2))
  export RAY_METRICS_EXPORT_PORT=$((P + 3))
  export RAY_DASHBOARD_AGENT_GRPC_PORT=$((P + 4))
  export RAY_DASHBOARD_AGENT_HTTP_PORT=$((P + 5))
  export RAY_RUNTIME_ENV_AGENT_PORT=$((P + 6))
  export RAY_MIN_WORKER_PORT=$((P + RAY_RESERVED_WITHIN_BLOCK))
  export RAY_MAX_WORKER_PORT=$((P + BLOCK_SIZE - 1))
  if [ $RAY_MAX_WORKER_PORT -gt $RAY_MAX_PORT ]; then
    return 2
  fi
  return 0
}



# ============================================================================
# Allocate ports
# ============================================================================

# Probe all fixed service ports. Worker ports are bound lazily by Ray, so
# probing the whole worker range is unnecessary (and noisy on shared nodes).
_probe_all_assigned_ports() {
  local port
  for port in \
    "$RAY_GCS_PORT" \
    "$RAY_NODE_MANAGER_PORT" \
    "$RAY_OBJECT_MANAGER_PORT" \
    "$RAY_METRICS_EXPORT_PORT" \
    "$RAY_DASHBOARD_AGENT_GRPC_PORT" \
    "$RAY_DASHBOARD_AGENT_HTTP_PORT" \
    "$RAY_RUNTIME_ENV_AGENT_PORT"
  do
    _probe_and_kill_port "$port" || return 1
  done
  return 0
}


# ============================================================================
# Start Ray with retries on failure
# ============================================================================

# Try to start Ray with the assigned ports
_try_start_ray() {
  ray start --head \
    --port="${RAY_GCS_PORT}" \
    --node-manager-port="${RAY_NODE_MANAGER_PORT}" \
    --object-manager-port="${RAY_OBJECT_MANAGER_PORT}" \
    --min-worker-port="${RAY_MIN_WORKER_PORT}" \
    --max-worker-port="${RAY_MAX_WORKER_PORT}" \
    --num-cpus="${CPUS}" \
    --memory="${RAY_MEMORY_BYTES}" \
    --object-store-memory="${RAY_OBJECT_STORE_MEMORY_BYTES}" \
    --plasma-directory="${RAY_TMPDIR}" \
    --temp-dir="${RAY_TMPDIR}" \
    --include-dashboard=false \
    --disable-usage-stats \
    --metrics-export-port="${RAY_METRICS_EXPORT_PORT}" \
    --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
    --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_HTTP_PORT}" \
    --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}"
}


# Loop over RAY_MAX_PORT_RETRIES attempts and try to start Ray
RAY_STARTED=0
for _attempt in $(seq 0 $RAY_MAX_PORT_RETRIES); do
  # Shift slot on each retry so we don't keep banging the same port set
  CUR_SLOT=$(( (ARRAY_SLOT + _attempt) % ARRAY_SLOTS ))
  
  # Assign ports for the current slot
  if ! _assign_ports_for_slot "$CUR_SLOT"; then
    echo "[WARN] Slot ${CUR_SLOT} overflows RAY_MAX_PORT, trying next slot"
    continue
  fi

  # Probe all assigned ports to ensure they are not in use by other processes
  if ! _probe_all_assigned_ports; then
    echo "[WARN] Attempt $((_attempt + 1)): port probe found unowned process on slot ${CUR_SLOT}, shifting slot"
    continue
  fi

  # Force the current task's driver to connect to the current task's head
  export RAY_ADDRESS="127.0.0.1:${RAY_GCS_PORT}"

  # Try to start Ray with the assigned (and freed) ports
  if _try_start_ray; then
    RAY_STARTED=1
    echo "Ray started successfully on attempt $((_attempt + 1)) (slot ${CUR_SLOT}, GCS ${RAY_GCS_PORT}, workers ${RAY_MIN_WORKER_PORT}-${RAY_MAX_WORKER_PORT})"
    echo "  --memory=${RAY_MEMORY_BYTES} (${RAY_MEM_MB_TOTAL} MiB - ${RAY_MEMORY_RESERVE_MB} MiB reserve)"
    echo "  --object-store-memory=${RAY_OBJECT_STORE_MEMORY_BYTES} (${RAY_OBJECT_STORE_FRACTION} of --memory)"
    echo "  --plasma-directory=${RAY_TMPDIR}"
    break
  fi

  echo "[WARN] Attempt $((_attempt + 1)): ray start failed on slot ${CUR_SLOT}, cleaning up and retrying"

  # Clean up only the half-started ray processes squatting on OUR assigned
  # ports. `ray stop --force` would kill every ray-like process the user
  # owns on this node, including any sibling array task co-packed onto the
  # same node. Killing by port is strictly narrower since anything bound to our
  # ports now must be our own failed ray attempt.
  for _cleanup_port in \
    "$RAY_GCS_PORT" \
    "$RAY_NODE_MANAGER_PORT" \
    "$RAY_OBJECT_MANAGER_PORT" \
    "$RAY_METRICS_EXPORT_PORT" \
    "$RAY_DASHBOARD_AGENT_GRPC_PORT" \
    "$RAY_DASHBOARD_AGENT_HTTP_PORT" \
    "$RAY_RUNTIME_ENV_AGENT_PORT"
  do
    _cleanup_pids=$(lsof -ti ":${_cleanup_port}" 2>/dev/null || true)
    if [ -n "$_cleanup_pids" ]; then
      echo "$_cleanup_pids" | xargs -r kill -9 2>/dev/null || true
    fi
  done

  sleep $(( 2 + RANDOM % 5 ))
done

# If Ray failed to start after all retries, exit with an error
if [ $RAY_STARTED -ne 1 ]; then
  echo "ERROR: ray start failed after $((RAY_MAX_PORT_RETRIES + 1)) attempts"
  return 1 2>/dev/null || exit 1
fi
