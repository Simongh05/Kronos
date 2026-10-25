#!/usr/bin/env bash
# run_job.sh - time-boxed, self-terminating wrapper for the btc_updown CLIs on a RunPod pod.
#
# Usage:
#   run_job.sh <python-module> <config.yaml> [--max-hours H] [--resume] [--no-terminate] [--foreground] [-- <extra CLI args>]
#
# Example (on the pod):
#   /workspace/Kronos/btc_updown/cloud/run_job.sh btc_updown.train.train_predictor \
#       btc_updown/configs/ft_small_1m.yaml --resume
#   /workspace/Kronos/btc_updown/cloud/run_job.sh btc_updown.train.train_predictor \
#       btc_updown/configs/ft_small_1m.yaml --max-hours 0.17 -- --max-steps 50     # dry run
#
# What it does
#   1. cd $KRONOS_DIR (default /workspace/Kronos); sources /workspace/env.sh if present.
#   2. Reads exp_name / run_dir / expected_hours from the YAML with a small python
#      helper (pyyaml if importable, a regex fallback otherwise). run_dir defaults to
#      /workspace/runs/<exp_name>; a "{exp_name}" placeholder is expanded and a
#      relative run_dir is taken relative to KRONOS_DIR.
#   3. Unless --foreground is given, re-launches itself detached with nohup, appending
#      to <run_dir>/job.log, prints the pid and returns at once (safe to close SSH).
#   4. The detached copy runs
#         python -m <module> --config <config> [--resume] [<extra CLI args after -->]
#      under `timeout` (GNU coreutils; a bash fallback is used where it is missing),
#      with a budget of 1.5 x expected_hours unless --max-hours overrides it.
#   5. Writes <run_dir>/job.status (key=value lines) when the job starts and when it
#      ends: status=running|succeeded|failed|timeout, exit_code, timestamps.
#   6. trap on EXIT: sync -f the volume, then terminate THIS pod through the RunPod
#      API (REST DELETE /pods/{id}, then GraphQL podTerminate, then runpodctl).
#      --no-terminate keeps the pod alive for debugging.
#
# Environment (all optional unless stated)
#   RUNPOD_POD_ID         set by RunPod inside every pod; without it nothing is terminated
#   RUNPOD_TERMINATE_KEY  API key used for self-termination (map a pod secret to it);
#                         falls back to RUNPOD_API_KEY, the pod-scoped key RunPod injects
#   RUNPOD_REST_URL       default https://rest.runpod.io/v1
#   RUNPOD_GRAPHQL_URL    default https://api.runpod.io/graphql
#   KRONOS_DIR            default /workspace/Kronos
#   KRONOS_WORKSPACE      default /workspace (what gets sync -f'ed)
#   RUNS_ROOT             default $KRONOS_WORKSPACE/runs
#   PYTHON                interpreter, default python (falls back to python3)
#   TIMEOUT_MULTIPLIER    default 1.5
#   KILL_GRACE_SECONDS    SIGTERM -> SIGKILL grace, default 60
#   TERMINATE_DELAY_SECONDS  pause before the API call, default 0
#
# Exit code: the wrapped command's exit code (124 on timeout), 2 on usage errors.
# shellcheck disable=SC2329  # trap handlers and their helpers are invoked indirectly
set -euo pipefail

ORIG_ARGS=("$@")
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")

usage() {
  # Print the comment header (everything up to the first blank line).
  awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$SELF"
}

log() {
  printf '[run_job %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 2
}

# ---------------------------------------------------------------- arguments
MODULE=""
CONFIG=""
MAX_HOURS=""
RESUME=0
NO_TERMINATE=0
FOREGROUND=${RUN_JOB_FOREGROUND:-0}
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --max-hours)
      [[ $# -ge 2 ]] || die "--max-hours needs a value"
      MAX_HOURS=$2; shift 2 ;;
    --max-hours=*) MAX_HOURS=${1#*=}; shift ;;
    --resume) RESUME=1; shift ;;
    --no-terminate) NO_TERMINATE=1; shift ;;
    --foreground) FOREGROUND=1; shift ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -*) die "unknown option: $1 (CLI options go after --)" ;;
    *)
      if [[ -z $MODULE ]]; then MODULE=$1
      elif [[ -z $CONFIG ]]; then CONFIG=$1
      else die "unexpected argument: $1"
      fi
      shift ;;
  esac
done

if [[ -z $MODULE || -z $CONFIG ]]; then
  usage >&2
  die "need <python-module> and <config.yaml>"
fi

KRONOS_DIR=${KRONOS_DIR:-/workspace/Kronos}
KRONOS_WORKSPACE=${KRONOS_WORKSPACE:-/workspace}
RUNS_ROOT=${RUNS_ROOT:-$KRONOS_WORKSPACE/runs}
RUNPOD_REST_URL=${RUNPOD_REST_URL:-https://rest.runpod.io/v1}
RUNPOD_GRAPHQL_URL=${RUNPOD_GRAPHQL_URL:-https://api.runpod.io/graphql}
TIMEOUT_MULTIPLIER=${TIMEOUT_MULTIPLIER:-1.5}
KILL_GRACE_SECONDS=${KILL_GRACE_SECONDS:-60}
TERMINATE_DELAY_SECONDS=${TERMINATE_DELAY_SECONDS:-0}

if [[ -z ${PYTHON:-} ]]; then
  if command -v python >/dev/null 2>&1; then PYTHON=python
  elif command -v python3 >/dev/null 2>&1; then PYTHON=python3
  else die "no python interpreter found"
  fi
elif [[ $PYTHON == */* && $PYTHON != /* ]]; then
  PYTHON=$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")   # keep it valid after cd
fi

# Resolve the config to an absolute path before changing directory.
if [[ -f $CONFIG && $CONFIG != /* ]]; then
  CONFIG=$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")
fi

[[ -d $KRONOS_DIR ]] || die "KRONOS_DIR not found: $KRONOS_DIR (run bootstrap.sh first)"
cd "$KRONOS_DIR"
[[ -f $CONFIG ]] || die "config not found: $CONFIG"

if [[ -f $KRONOS_WORKSPACE/env.sh ]]; then
  # shellcheck disable=SC1091
  source "$KRONOS_WORKSPACE/env.sh"
fi

# ---------------------------------------------------------------- read the YAML
read_config() {
  "$PYTHON" - "$1" <<'PY'
import re
import sys

path = sys.argv[1]
cfg = None
try:
    import yaml  # type: ignore
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
except Exception:
    cfg = None


def find(node, key):
    """First value for `key`, top level first, then depth-first."""
    if isinstance(node, dict):
        if key in node and node[key] not in (None, ""):
            return node[key]
        for value in node.values():
            found = find(value, key)
            if found is not None:
                return found
    return None


if cfg is None:
    text = open(path).read()

    def find(_, key, _text=text):  # noqa: F811 - regex fallback without pyyaml
        match = re.search(r"^\s*%s\s*:\s*[\"']?([^\"'#\n]+)" % re.escape(key), _text, re.M)
        return match.group(1).strip() if match else None

exp_name = find(cfg, "exp_name")
run_dir = find(cfg, "run_dir")
hours = find(cfg, "expected_hours")
if run_dir and exp_name:
    run_dir = str(run_dir).replace("{exp_name}", str(exp_name))
print(exp_name or "")
print(run_dir or "")
print("" if hours is None else hours)
PY
}

CFG_OUT=$(read_config "$CONFIG") || die "could not parse $CONFIG"
EXP_NAME=$(printf '%s\n' "$CFG_OUT" | sed -n 1p)
RUN_DIR=$(printf '%s\n' "$CFG_OUT" | sed -n 2p)
EXPECTED_HOURS=$(printf '%s\n' "$CFG_OUT" | sed -n 3p)

[[ -n $EXP_NAME ]] || die "no exp_name in $CONFIG"
[[ $EXP_NAME =~ ^[A-Za-z0-9._-]+$ ]] || die "exp_name must be [A-Za-z0-9._-]+: '$EXP_NAME'"
[[ -n $RUN_DIR ]] || RUN_DIR=$RUNS_ROOT/$EXP_NAME
[[ $RUN_DIR == /* ]] || RUN_DIR=$KRONOS_DIR/$RUN_DIR   # relative run_dir is relative to the repo

if [[ -n $MAX_HOURS ]]; then
  BUDGET_HOURS=$MAX_HOURS
  BUDGET_SOURCE="--max-hours"
elif [[ -n $EXPECTED_HOURS ]]; then
  BUDGET_HOURS=$("$PYTHON" -c 'import sys; print(float(sys.argv[1]) * float(sys.argv[2]))' \
    "$EXPECTED_HOURS" "$TIMEOUT_MULTIPLIER")
  BUDGET_SOURCE="${TIMEOUT_MULTIPLIER} x expected_hours=${EXPECTED_HOURS}"
else
  BUDGET_HOURS=1.5
  BUDGET_SOURCE="default (no expected_hours in config)"
fi
MAX_SECONDS=$("$PYTHON" -c 'import sys; v = float(sys.argv[1]) * 3600.0; print("%.3f" % max(v, 0.5))' \
  "$BUDGET_HOURS") || die "bad --max-hours / expected_hours value"

mkdir -p "$RUN_DIR"
LOG_FILE=$RUN_DIR/job.log
STATUS_FILE=$RUN_DIR/job.status

# ---------------------------------------------------------------- detach
if [[ $FOREGROUND -ne 1 ]]; then
  if [[ -f $STATUS_FILE ]] && grep -q '^status=running$' "$STATUS_FILE"; then
    old_pid=$(sed -n 's/^pid=//p' "$STATUS_FILE")
    if [[ -n $old_pid ]] && kill -0 "$old_pid" 2>/dev/null; then
      die "a job for $EXP_NAME is already running (pid $old_pid); see $STATUS_FILE"
    fi
  fi
  log "detaching: python -m $MODULE --config $CONFIG (budget ${MAX_SECONDS}s from $BUDGET_SOURCE)"
  RUN_JOB_FOREGROUND=1 nohup bash "$SELF" "${ORIG_ARGS[@]}" >>"$LOG_FILE" 2>&1 </dev/null &
  child=$!
  printf '%s\n' "$child" >"$RUN_DIR/job.pid"
  log "started pid $child"
  log "log:    $LOG_FILE"
  log "status: $STATUS_FILE"
  log "follow: tail -f $LOG_FILE"
  exit 0
fi

# ---------------------------------------------------------------- foreground worker
START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
START_EPOCH=$(date +%s)
FINAL_STATUS=""
CHILD_PID=""
WATCHDOG_PID=""
TIMEOUT_BIN=""
if command -v timeout >/dev/null 2>&1; then TIMEOUT_BIN=$(command -v timeout); fi

write_status() { # write_status <status> [exit_code]
  local status=$1 code=${2:-} tmp=$STATUS_FILE.tmp
  {
    echo "status=$status"
    echo "exp_name=$EXP_NAME"
    echo "module=$MODULE"
    echo "config=$CONFIG"
    echo "run_dir=$RUN_DIR"
    echo "resume=$RESUME"
    echo "extra_args=${EXTRA_ARGS[*]:-}"
    echo "max_seconds=$MAX_SECONDS"
    echo "budget_source=$BUDGET_SOURCE"
    echo "started_at=$START_TS"
    echo "pid=$$"
    echo "pod_id=${RUNPOD_POD_ID:-}"
    echo "terminate_pod=$((1 - NO_TERMINATE))"
    echo "hostname=$(hostname 2>/dev/null || echo unknown)"
    if [[ -n $code ]]; then
      echo "exit_code=$code"
      echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "elapsed_seconds=$(( $(date +%s) - START_EPOCH ))"
    fi
  } >"$tmp" && mv -f "$tmp" "$STATUS_FILE"
}

flush_volume() {
  sync -f "$KRONOS_WORKSPACE" 2>/dev/null || sync 2>/dev/null || true
}

# curl with its config (URL, headers, data) fed through stdin so the API key never
# appears in argv / `ps`. Prints the HTTP status code (000 on transport failure).
curl_cfg() { # curl_cfg <output-file>   (config on stdin)
  curl -sS --max-time 60 -K - -o "$1" -w '%{http_code}' 2>>"$RUN_DIR/terminate.err" || echo 000
}

terminate_pod() {
  local pod=${RUNPOD_POD_ID:-} key=${RUNPOD_TERMINATE_KEY:-${RUNPOD_API_KEY:-}}
  local code body
  if [[ -z $pod ]]; then
    log "RUNPOD_POD_ID is not set: not running on a RunPod pod, nothing to terminate"
    return 0
  fi
  if [[ $TERMINATE_DELAY_SECONDS -gt 0 ]]; then
    log "waiting ${TERMINATE_DELAY_SECONDS}s before terminating pod $pod"
    sleep "$TERMINATE_DELAY_SECONDS"
  fi
  if [[ -n $key ]]; then
    # 1. REST: DELETE https://rest.runpod.io/v1/pods/{podId}  (204 on success)
    code=$(printf 'url = "%s/pods/%s"\nrequest = "DELETE"\nheader = "Authorization: Bearer %s"\n' \
      "$RUNPOD_REST_URL" "$pod" "$key" | curl_cfg /dev/null)
    case $code in
      2*) log "pod $pod terminated via REST DELETE (HTTP $code)"; return 0 ;;
      *)  log "REST DELETE /pods/$pod returned HTTP $code; trying GraphQL" ;;
    esac
    # 2. GraphQL: mutation { podTerminate(input: {podId: "..."}) }
    body=$(mktemp)
    printf '{"query":"mutation { podTerminate(input: {podId: \\"%s\\"}) }"}' "$pod" >"$body"
    code=$(printf 'url = "%s?api_key=%s"\nrequest = "POST"\nheader = "Content-Type: application/json"\nheader = "Authorization: Bearer %s"\ndata = "@%s"\n' \
      "$RUNPOD_GRAPHQL_URL" "$key" "$key" "$body" | curl_cfg "$body.out")
    if [[ $code == 2* ]] && ! grep -q '"errors"' "$body.out" 2>/dev/null; then
      log "pod $pod terminated via GraphQL podTerminate (HTTP $code)"
      rm -f "$body" "$body.out"
      return 0
    fi
    log "GraphQL podTerminate returned HTTP $code: $(tr -d '\n' <"$body.out" 2>/dev/null | cut -c1-300)"
    rm -f "$body" "$body.out"
  else
    log "no RUNPOD_TERMINATE_KEY / RUNPOD_API_KEY in the environment; skipping the HTTP API"
  fi
  # 3. runpodctl (preinstalled on pods with a pod-scoped key)
  if command -v runpodctl >/dev/null 2>&1; then
    if runpodctl pod delete "$pod" >>"$LOG_FILE" 2>&1; then
      log "pod $pod terminated via 'runpodctl pod delete'"
      return 0
    fi
    if runpodctl remove pod "$pod" >>"$LOG_FILE" 2>&1; then
      log "pod $pod terminated via legacy 'runpodctl remove pod'"
      return 0
    fi
    log "runpodctl could not delete pod $pod"
  else
    log "runpodctl not found"
  fi
  log "!!! POD $pod IS STILL RUNNING AND BILLING: terminate it in the RunPod console or with watchdog.sh --kill-all --apply"
  return 1
}

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ -z $FINAL_STATUS ]]; then
    # Reached without a normal finish: set -e failure, signal, or an internal error.
    [[ $rc -eq 0 ]] && rc=1
    write_status failed "$rc"
    log "wrapper exited early (exit $rc)"
  fi
  flush_volume
  if [[ $NO_TERMINATE -eq 1 ]]; then
    log "--no-terminate given: pod left running"
  else
    terminate_pod || true
    flush_volume
  fi
  exit "$rc"
}

on_signal() { # forward to the child so the EXIT trap still runs
  local sig=$1
  log "received SIG$sig; stopping the job"
  if [[ -n $CHILD_PID ]]; then kill -s "$sig" "$CHILD_PID" 2>/dev/null || true; fi
  if [[ -n $WATCHDOG_PID ]]; then kill "$WATCHDOG_PID" 2>/dev/null || true; fi
  case $sig in
    INT) exit 130 ;;
    *)   exit 143 ;;
  esac
}

trap on_exit EXIT
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP

run_timeboxed() { # run_timeboxed <seconds> <command...>; returns 124 on timeout
  local secs=$1; shift
  local rc=0
  if [[ -n $TIMEOUT_BIN ]]; then
    "$TIMEOUT_BIN" -k "$KILL_GRACE_SECONDS" "$secs" "$@" &
    CHILD_PID=$!
    wait "$CHILD_PID" || rc=$?
    CHILD_PID=""
    return "$rc"
  fi
  # Fallback for systems without GNU timeout (e.g. macOS when testing).
  local marker=$RUN_DIR/.timeout_fired
  rm -f "$marker"
  "$@" &
  CHILD_PID=$!
  (
    sleep "$secs"
    if kill -0 "$CHILD_PID" 2>/dev/null; then
      touch "$marker"
      kill -TERM "$CHILD_PID" 2>/dev/null || true
      sleep "$KILL_GRACE_SECONDS"
      kill -KILL "$CHILD_PID" 2>/dev/null || true
    fi
  ) 2>/dev/null &
  WATCHDOG_PID=$!
  wait "$CHILD_PID" || rc=$?
  CHILD_PID=""
  # Stop the watchdog quietly (bash would otherwise print "Terminated: sleep ...").
  exec 3>&2 2>/dev/null
  pkill -P "$WATCHDOG_PID" || true
  kill "$WATCHDOG_PID" || true
  wait "$WATCHDOG_PID" || true
  exec 2>&3 3>&-
  WATCHDOG_PID=""
  if [[ -e $marker ]]; then
    rm -f "$marker"
    return 124
  fi
  return "$rc"
}

CMD=("$PYTHON" -m "$MODULE" --config "$CONFIG")
if [[ $RESUME -eq 1 ]]; then CMD+=(--resume); fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then CMD+=("${EXTRA_ARGS[@]}"); fi

write_status running
log "exp_name=$EXP_NAME run_dir=$RUN_DIR"
log "budget ${MAX_SECONDS}s ($BUDGET_SOURCE); terminate_pod=$((1 - NO_TERMINATE)); pod=${RUNPOD_POD_ID:-none}"
log "timeout implementation: ${TIMEOUT_BIN:-bash fallback}"
log "running: ${CMD[*]}"

RC=0
run_timeboxed "$MAX_SECONDS" "${CMD[@]}" || RC=$?

ELAPSED=$(( $(date +%s) - START_EPOCH ))
MAX_INT=${MAX_SECONDS%.*}
if [[ $RC -eq 0 ]]; then
  FINAL_STATUS=succeeded
elif [[ $RC -eq 124 ]]; then
  FINAL_STATUS=timeout
elif [[ ($RC -eq 137 || $RC -eq 143) && $ELAPSED -ge ${MAX_INT:-0} ]]; then
  FINAL_STATUS=timeout
  RC=124
else
  FINAL_STATUS=failed
fi
write_status "$FINAL_STATUS" "$RC"
log "job $FINAL_STATUS (exit $RC) after ${ELAPSED}s; status in $STATUS_FILE"
exit "$RC"
