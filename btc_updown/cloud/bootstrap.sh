#!/usr/bin/env bash
# bootstrap.sh - prepare a fresh RunPod pod (image runpod/pytorch:<2.4+, CUDA 12.x>,
# Ubuntu, network volume mounted at /workspace) for the btc_updown jobs. Idempotent:
# re-running only fetches/installs what is missing.
#
# Usage (on the pod, as root):
#   REPO_URL=https://github.com/<you>/Kronos.git [GIT_TOKEN=...] bash bootstrap.sh [--refresh-klines] [--skip-klines] [--skip-tests]
#
# Steps
#   1. apt: git rsync jq (only if missing)
#   2. clone or fast-forward branch $REPO_BRANCH (btc-updown) into $KRONOS_DIR
#      (/workspace/Kronos). GIT_TOKEN, if set, is handed to git through GIT_ASKPASS
#      only: it is never put in the URL, in .git/config, or echoed.
#   3. pip install -r requirements.txt + pyyaml pyarrow zstandard scikit-learn pytest
#   4. download the Binance 1m klines to $KLINES_DIR (/workspace/klines) unless
#      $KLINES_DIR/spot_BTCUSDC/meta.json exists (or --refresh-klines is given)
#   5. smoke test: python -m pytest btc_updown/tests -q -x
#   6. print nvidia-smi, torch/CUDA version, bf16 support, free disk on /workspace
#   7. write /workspace/env.sh (HF cache on the volume) if it does not exist
#
# Environment
#   REPO_URL      required: HTTPS clone URL of the Kronos fork holding btc-updown
#   REPO_BRANCH   default btc-updown
#   GIT_TOKEN     optional: token for a private repo (GitHub PAT / fine-grained token)
#   GIT_USER      user name sent with GIT_TOKEN, default x-access-token
#   KRONOS_DIR    default /workspace/Kronos
#   KLINES_DIR    default /workspace/klines
#   KLINES_START  default 2019-01      KLINES_END  default 2026-10
#   PYTHON        default python (falls back to python3)
set -euo pipefail

log() { printf '[bootstrap %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

REFRESH_KLINES=0
SKIP_KLINES=0
SKIP_TESTS=0
for arg in "$@"; do
  case "$arg" in
    --refresh-klines) REFRESH_KLINES=1 ;;
    --skip-klines) SKIP_KLINES=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    -h|--help)
      awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"
      exit 0 ;;
    *) die "unknown option: $arg" ;;
  esac
done

REPO_BRANCH=${REPO_BRANCH:-btc-updown}
KRONOS_DIR=${KRONOS_DIR:-/workspace/Kronos}
WORKSPACE=$(dirname "$KRONOS_DIR")
KLINES_DIR=${KLINES_DIR:-/workspace/klines}
KLINES_START=${KLINES_START:-2019-01}
KLINES_END=${KLINES_END:-2026-10}
export PIP_DISABLE_PIP_VERSION_CHECK=1
export DEBIAN_FRONTEND=noninteractive

[[ -n ${REPO_URL:-} ]] || die "REPO_URL is not set (HTTPS clone URL of the fork that has branch $REPO_BRANCH)"
[[ $REPO_URL == https://* ]] || die "REPO_URL must be an https:// URL (got something else)"
case $REPO_URL in *@*) die "REPO_URL must not contain credentials; pass GIT_TOKEN instead" ;; esac

if [[ -z ${PYTHON:-} ]]; then
  if command -v python >/dev/null 2>&1; then PYTHON=python
  elif command -v python3 >/dev/null 2>&1; then PYTHON=python3
  else die "no python interpreter found"
  fi
elif [[ $PYTHON == */* && $PYTHON != /* ]]; then
  PYTHON=$(cd "$(dirname "$PYTHON")" && pwd)/$(basename "$PYTHON")   # keep it valid after cd
fi

SUDO=""
if [[ $(id -u) -ne 0 ]] && command -v sudo >/dev/null 2>&1; then SUDO=sudo; fi

[[ -d $WORKSPACE ]] || die "$WORKSPACE does not exist: is the network volume mounted?"
mkdir -p "$WORKSPACE/runs" "$WORKSPACE/cubes" "$KLINES_DIR"

# ---------------------------------------------------------------- 1. apt packages
missing=()
for tool in git rsync jq; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  log "installing: ${missing[*]}"
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq --no-install-recommends "${missing[@]}"
else
  log "git rsync jq already installed"
fi

# ---------------------------------------------------------------- 2. repo
ASKPASS=""
cleanup() { [[ -n $ASKPASS ]] && rm -f "$ASKPASS"; }
trap cleanup EXIT
if [[ -n ${GIT_TOKEN:-} ]]; then
  ASKPASS=$(mktemp)
  chmod 700 "$ASKPASS"
  cat >"$ASKPASS" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  *sername*) printf '%s\n' "${GIT_USER:-x-access-token}" ;;
  *) printf '%s\n' "${GIT_TOKEN:-}" ;;
esac
EOF
  export GIT_TOKEN GIT_USER=${GIT_USER:-x-access-token}
  export GIT_ASKPASS=$ASKPASS GIT_TERMINAL_PROMPT=0
  log "GIT_TOKEN present: using GIT_ASKPASS (token never written to disk or URLs)"
fi
GIT=(git -c credential.helper=)

if [[ -d $KRONOS_DIR/.git ]]; then
  log "updating $KRONOS_DIR to origin/$REPO_BRANCH"
  "${GIT[@]}" -C "$KRONOS_DIR" remote set-url origin "$REPO_URL"
  "${GIT[@]}" -C "$KRONOS_DIR" fetch --quiet origin "$REPO_BRANCH"
  "${GIT[@]}" -C "$KRONOS_DIR" checkout --quiet -B "$REPO_BRANCH" --no-track FETCH_HEAD 2>/dev/null \
    || "${GIT[@]}" -C "$KRONOS_DIR" reset --hard --quiet FETCH_HEAD
else
  log "cloning $REPO_URL ($REPO_BRANCH) into $KRONOS_DIR"
  "${GIT[@]}" clone --quiet --branch "$REPO_BRANCH" --single-branch "$REPO_URL" "$KRONOS_DIR"
fi
cd "$KRONOS_DIR"
log "repo at $(git rev-parse --short HEAD) ($(git log -1 --format=%cd --date=short)): $(git log -1 --format=%s)"

# ---------------------------------------------------------------- 3. python deps
log "pip install (requirements.txt + pyyaml pyarrow zstandard scikit-learn pytest)"
"$PYTHON" -m pip install --quiet -r requirements.txt pyyaml pyarrow zstandard scikit-learn pytest
"$PYTHON" -c 'import yaml, pyarrow, zstandard, sklearn, pytest' || die "python dependencies missing after pip install"

# ---------------------------------------------------------------- 7. env.sh (early: later steps use it)
ENV_SH=$WORKSPACE/env.sh
if [[ ! -f $ENV_SH ]]; then
  cat >"$ENV_SH" <<EOF
# Sourced by run_job.sh and meant for interactive shells: 'source $ENV_SH'
export HF_HOME=$WORKSPACE/hf            # pretrained weights persist on the volume
export PYTHONUNBUFFERED=1
export KRONOS_DIR=$KRONOS_DIR
export KLINES_DIR=$KLINES_DIR
EOF
  log "wrote $ENV_SH"
fi
# shellcheck disable=SC1090
source "$ENV_SH"
mkdir -p "${HF_HOME:-$WORKSPACE/hf}"

# ---------------------------------------------------------------- 4. klines
KLINES_MARKER=$KLINES_DIR/spot_BTCUSDC/meta.json
if [[ $SKIP_KLINES -eq 1 ]]; then
  log "--skip-klines: not touching $KLINES_DIR"
elif [[ $REFRESH_KLINES -eq 1 || ! -f $KLINES_MARKER ]]; then
  log "downloading klines to $KLINES_DIR ($KLINES_START .. $KLINES_END)"
  "$PYTHON" -m btc_updown.data.binance_klines \
    --series spot_BTCUSDC spot_BTCUSDT um_BTCUSDT \
    --start "$KLINES_START" --end "$KLINES_END" \
    --out "$KLINES_DIR"
  [[ -f $KLINES_MARKER ]] || die "kline download finished but $KLINES_MARKER is missing"
else
  log "klines present ($KLINES_MARKER); pass --refresh-klines to re-download"
fi

# ---------------------------------------------------------------- 5. smoke test
if [[ $SKIP_TESTS -eq 1 ]]; then
  log "--skip-tests: skipping pytest"
else
  log "smoke test: python -m pytest btc_updown/tests -q -x"
  "$PYTHON" -m pytest btc_updown/tests -q -x
fi

# ---------------------------------------------------------------- 6. hardware report
log "---- GPU ----"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || nvidia-smi || true
else
  log "nvidia-smi not found (no GPU?)"
fi
"$PYTHON" - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device", p.name, "| %.1f GB" % (p.total_memory / 2**30), "| capability %d.%d" % (p.major, p.minor))
    print("bf16 supported", torch.cuda.is_bf16_supported())
PY
log "---- disk ----"
df -h "$WORKSPACE" | tail -n +1
log "---- summary ----"
log "repo:    $KRONOS_DIR @ $(git rev-parse --short HEAD) [$REPO_BRANCH]"
log "klines:  $KLINES_DIR $( [[ -f $KLINES_MARKER ]] && echo ready || echo MISSING )"
log "env:     source $ENV_SH"
log "pod:     id=${RUNPOD_POD_ID:-unknown} gpu_count=${RUNPOD_GPU_COUNT:-?} dc=${RUNPOD_DC_ID:-?}"
log "next:    $KRONOS_DIR/btc_updown/cloud/run_job.sh <module> <config.yaml> [--resume]"
log "bootstrap done"
