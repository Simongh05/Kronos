#!/usr/bin/env bash
# sync.sh - rsync between the Mac and a RunPod pod over the pod's public-IP SSH.
#
# Usage:
#   sync.sh <pod-ip-or-host> <ssh-port> [--key ~/.ssh/id_ed25519] [--user root]
#           [--pull-only | --push-only] [--dry-run] [--verbose] [--no-host-check]
#
# Pulls  /workspace/runs/  and /workspace/cubes/  -> <repo>/data/cloud/{runs,cubes}/
# Pushes <repo>/btc_updown/configs/               -> /workspace/Kronos/btc_updown/configs/
# with  --partial --inplace  (resumable, checkpoints updated in place) and
# excluding *.tmp (temp-then-rename files that are still being written). Idempotent:
# a second run transfers only what changed. Nothing is deleted on either side.
#
# The host/port come from the pod's "Connect" panel (TCP port 22 mapping) or from
#   watchdog.sh --json   /  RUNPOD_PUBLIC_IP + RUNPOD_TCP_PORT_22 inside the pod.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
STATE_DIR=${KRONOS_CLOUD_DIR:-$REPO_ROOT/data/cloud}
REMOTE_WORKSPACE=${KRONOS_WORKSPACE:-/workspace}
REMOTE_REPO=${KRONOS_DIR:-$REMOTE_WORKSPACE/Kronos}
RSYNC=${RSYNC:-rsync}

log() { printf '[sync %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "ERROR: $*" >&2; exit 2; }
usage() { awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"; }

HOST=""
PORT=""
KEY=${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}
USER_NAME=root
PULL=1
PUSH=1
DRY_RUN=0
VERBOSE=0
HOST_CHECK=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --key) [[ $# -ge 2 ]] || die "--key needs a path"; KEY=$2; shift 2 ;;
    --user) [[ $# -ge 2 ]] || die "--user needs a name"; USER_NAME=$2; shift 2 ;;
    --pull-only) PUSH=0; shift ;;
    --push-only) PULL=0; shift ;;
    --dry-run|-n) DRY_RUN=1; shift ;;
    --verbose|-v) VERBOSE=1; shift ;;
    --no-host-check) HOST_CHECK=0; shift ;;
    -*) die "unknown option: $1" ;;
    *)
      if [[ -z $HOST ]]; then HOST=$1
      elif [[ -z $PORT ]]; then PORT=$1
      else die "unexpected argument: $1"
      fi
      shift ;;
  esac
done

[[ -n $HOST && -n $PORT ]] || { usage >&2; die "need <host> <port>"; }
[[ $PORT =~ ^[0-9]+$ ]] || die "port must be numeric: $PORT"
[[ -f $KEY ]] || die "SSH key not found: $KEY (pass --key or set RUNPOD_SSH_KEY)"
command -v "$RSYNC" >/dev/null 2>&1 || die "rsync not found"

SSH_OPTS=(-p "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 -o BatchMode=yes)
if [[ $HOST_CHECK -eq 1 ]]; then
  # Pods are recreated often and IP:port pairs get reused, so keep their keys apart.
  mkdir -p "$STATE_DIR"
  SSH_OPTS+=(-o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$STATE_DIR/known_hosts")
else
  SSH_OPTS+=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
fi
REMOTE=$USER_NAME@$HOST

# `ssh` command string for rsync -e; the options contain no spaces except the
# known_hosts path, which is quoted.
SSH_CMD="ssh"
for opt in "${SSH_OPTS[@]}"; do
  case $opt in
    *" "*) SSH_CMD="$SSH_CMD '$opt'" ;;
    *) SSH_CMD="$SSH_CMD $opt" ;;
  esac
done

RSYNC_OPTS=(-az --partial --inplace --exclude '*.tmp' --exclude '__pycache__/' -e "$SSH_CMD")
[[ $DRY_RUN -eq 1 ]] && RSYNC_OPTS+=(--dry-run)
[[ $VERBOSE -eq 1 ]] && RSYNC_OPTS+=(-v --progress)
[[ $DRY_RUN -eq 1 && $VERBOSE -eq 0 ]] && RSYNC_OPTS+=(-v)

log "checking ssh $REMOTE -p $PORT"
# shellcheck disable=SC2029  # the paths are meant to expand here, on the client
if ! ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p '$REMOTE_WORKSPACE/runs' '$REMOTE_WORKSPACE/cubes' '$REMOTE_REPO/btc_updown/configs' && test -d '$REMOTE_WORKSPACE'"; then
  if [[ $HOST_CHECK -eq 1 ]]; then
    die "ssh failed. If the host key changed (reused IP:port), delete the line from $STATE_DIR/known_hosts or pass --no-host-check"
  fi
  die "ssh failed"
fi

rc=0
if [[ $PULL -eq 1 ]]; then
  for d in runs cubes; do
    mkdir -p "$STATE_DIR/$d"
    log "pull $REMOTE:$REMOTE_WORKSPACE/$d/ -> $STATE_DIR/$d/"
    "$RSYNC" "${RSYNC_OPTS[@]}" "$REMOTE:$REMOTE_WORKSPACE/$d/" "$STATE_DIR/$d/" || { rc=$?; log "rsync pull $d failed (exit $rc)"; }
  done
fi
if [[ $PUSH -eq 1 ]]; then
  if [[ -d $REPO_ROOT/btc_updown/configs ]]; then
    log "push $REPO_ROOT/btc_updown/configs/ -> $REMOTE:$REMOTE_REPO/btc_updown/configs/"
    "$RSYNC" "${RSYNC_OPTS[@]}" "$REPO_ROOT/btc_updown/configs/" "$REMOTE:$REMOTE_REPO/btc_updown/configs/" || { rc=$?; log "rsync push configs failed (exit $rc)"; }
  else
    log "no local btc_updown/configs/ directory; nothing to push"
  fi
fi

if [[ $rc -eq 0 ]]; then
  log "done$( [[ $DRY_RUN -eq 1 ]] && echo ' (dry run)' )"
else
  log "finished with errors (exit $rc); re-run, rsync resumes partial files"
fi
exit "$rc"
