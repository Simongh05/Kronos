#!/usr/bin/env bash
# watchdog.sh - Mac-side guard against forgotten RunPod pods. Needs only curl and jq.
#
# Usage:
#   watchdog.sh                       list pods, show which are past their deadline (dry run)
#   watchdog.sh --apply [--yes]       terminate pods past their deadline (asks first unless --yes)
#   watchdog.sh --kill-all --apply    terminate every pod (asks first unless --yes)
#   watchdog.sh --set-deadline <pod_id> <hours>   record now+hours as the pod's deadline
#   watchdog.sh --json                print the normalized pod list as JSON and exit
#
# Options
#   --env-file FILE     where RUNPOD_API_KEY= is read from (default <repo>/.env);
#                       an exported RUNPOD_API_KEY wins over the file
#   --deadlines FILE    default <repo>/data/cloud/deadlines.json, {"<pod_id>": "<ISO-8601 UTC>"}
#   --pod ID            restrict --apply / --kill-all to one pod id
#   --graphql           list and terminate through api.runpod.io/graphql instead of REST
#
# Exit codes: 0 ok, 1 a termination failed, 2 usage / auth / API error.
# The API key is never printed. Suitable for launchd (see com.kronos.runpod-watchdog.plist).
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
STATE_DIR=${KRONOS_CLOUD_DIR:-$REPO_ROOT/data/cloud}
ENV_FILE=${KRONOS_ENV_FILE:-$REPO_ROOT/.env}
DEADLINES=$STATE_DIR/deadlines.json
RUNPOD_REST_URL=${RUNPOD_REST_URL:-https://rest.runpod.io/v1}
RUNPOD_GRAPHQL_URL=${RUNPOD_GRAPHQL_URL:-https://api.runpod.io/graphql}

APPLY=0
YES=0
KILL_ALL=0
ONLY_POD=""
USE_GRAPHQL=0
JSON_OUT=0
SET_DEADLINE_POD=""
SET_DEADLINE_HOURS=""

log() { printf '[watchdog %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "ERROR: $*" >&2; exit 2; }
usage() { awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --apply) APPLY=1; shift ;;
    --yes|-y) YES=1; shift ;;
    --kill-all) KILL_ALL=1; shift ;;
    --json) JSON_OUT=1; shift ;;
    --graphql) USE_GRAPHQL=1; shift ;;
    --pod) [[ $# -ge 2 ]] || die "--pod needs an id"; ONLY_POD=$2; shift 2 ;;
    --env-file) [[ $# -ge 2 ]] || die "--env-file needs a path"; ENV_FILE=$2; shift 2 ;;
    --deadlines) [[ $# -ge 2 ]] || die "--deadlines needs a path"; DEADLINES=$2; shift 2 ;;
    --set-deadline)
      [[ $# -ge 3 ]] || die "--set-deadline needs <pod_id> <hours>"
      SET_DEADLINE_POD=$2; SET_DEADLINE_HOURS=$3; shift 3 ;;
    *) die "unknown option: $1" ;;
  esac
done

command -v curl >/dev/null 2>&1 || die "curl not found"
command -v jq >/dev/null 2>&1 || die "jq not found (brew install jq)"
mkdir -p "$(dirname "$DEADLINES")"
LOG_FILE=$(dirname "$DEADLINES")/watchdog.log

# ---------------------------------------------------------------- --set-deadline
if [[ -n $SET_DEADLINE_POD ]]; then
  [[ $SET_DEADLINE_HOURS =~ ^[0-9]+([.][0-9]+)?$ ]] || die "hours must be a number"
  [[ -f $DEADLINES ]] || echo '{}' >"$DEADLINES"
  tmp=$DEADLINES.tmp
  jq --arg id "$SET_DEADLINE_POD" --argjson h "$SET_DEADLINE_HOURS" \
    '. + {($id): ((now + $h * 3600) | todate)}' "$DEADLINES" >"$tmp" && mv -f "$tmp" "$DEADLINES"
  log "deadline for $SET_DEADLINE_POD: $(jq -r --arg id "$SET_DEADLINE_POD" '.[$id]' "$DEADLINES") (in $DEADLINES)"
  exit 0
fi

# ---------------------------------------------------------------- API key
API_KEY=${RUNPOD_API_KEY:-}
if [[ -z $API_KEY ]]; then
  [[ -f $ENV_FILE ]] || die "no RUNPOD_API_KEY in the environment and no env file at $ENV_FILE"
  line=$(grep -E '^[[:space:]]*(export[[:space:]]+)?RUNPOD_API_KEY[[:space:]]*=' "$ENV_FILE" | tail -n 1 || true)
  [[ -n $line ]] || die "RUNPOD_API_KEY= not found in $ENV_FILE"
  API_KEY=${line#*=}
  API_KEY=${API_KEY%%#*}
  API_KEY=$(printf '%s' "$API_KEY" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/")
fi
[[ -n $API_KEY ]] || die "RUNPOD_API_KEY is empty"

# curl with URL/headers on stdin so the key never shows up in argv. Prints HTTP code.
curl_cfg() { # curl_cfg <output-file>
  curl -sS --max-time 60 -K - -o "$1" -w '%{http_code}' 2>/dev/null || echo 000
}

# ---------------------------------------------------------------- list pods
RAW=$(mktemp)
NORMALIZED=$(mktemp)
trap 'rm -f "$RAW" "$NORMALIZED"' EXIT

if [[ $USE_GRAPHQL -eq 0 ]]; then
  # REST: GET https://rest.runpod.io/v1/pods  -> array of pods
  code=$(printf 'url = "%s/pods"\nheader = "Authorization: Bearer %s"\n' "$RUNPOD_REST_URL" "$API_KEY" | curl_cfg "$RAW")
  [[ $code == 2* ]] || die "GET $RUNPOD_REST_URL/pods returned HTTP $code: $(cut -c1-300 "$RAW" | tr -d '\n')"
  jq '[ (if type == "array" then .[] else (.pods // [])[] end)
        | { id: .id,
            name: (.name // ""),
            gpu: (.gpu.displayName // .machine.gpuType.displayName // .machine.gpuTypeId // "?"),
            cost_per_hr: ((.costPerHr // 0) | tonumber),
            status: (.desiredStatus // "?"),
            started_at: (.lastStartedAt // null),
            uptime_s: null } ]' "$RAW" >"$NORMALIZED" || die "unexpected REST response: $(cut -c1-300 "$RAW")"
else
  # GraphQL: query { myself { pods { ... } } }
  body=$(mktemp)
  printf '%s' '{"query":"query { myself { pods { id name costPerHr desiredStatus lastStartedAt machine { gpuDisplayName } runtime { uptimeInSeconds } } } }"}' >"$body"
  code=$(printf 'url = "%s?api_key=%s"\nrequest = "POST"\nheader = "Content-Type: application/json"\nheader = "Authorization: Bearer %s"\ndata = "@%s"\n' \
    "$RUNPOD_GRAPHQL_URL" "$API_KEY" "$API_KEY" "$body" | curl_cfg "$RAW")
  rm -f "$body"
  [[ $code == 2* ]] || die "GraphQL returned HTTP $code: $(cut -c1-300 "$RAW" | tr -d '\n')"
  jq -e '.errors == null' "$RAW" >/dev/null || die "GraphQL errors: $(jq -c '.errors' "$RAW" | cut -c1-300)"
  jq '[ .data.myself.pods[]
        | { id: .id,
            name: (.name // ""),
            gpu: (.machine.gpuDisplayName // "?"),
            cost_per_hr: ((.costPerHr // 0) | tonumber),
            status: (.desiredStatus // "?"),
            started_at: (.lastStartedAt // null),
            uptime_s: (.runtime.uptimeInSeconds // null) } ]' "$RAW" >"$NORMALIZED" || die "unexpected GraphQL response"
fi

[[ -f $DEADLINES ]] || echo '{}' >"$DEADLINES"
jq -e 'type == "object"' "$DEADLINES" >/dev/null || die "$DEADLINES must be a JSON object {pod_id: iso_deadline}"

# Enrich: uptime from lastStartedAt when the API gives none, cost so far, deadline, overdue.
# ISO timestamps are normalized to the second and to Z so jq's fromdateiso8601 accepts them.
ENRICHED=$(jq --slurpfile dl "$DEADLINES" '
  def iso2epoch: if . == null then null else
    (tostring | sub("\\.[0-9]+"; "") | sub("\\+00:00$"; "Z") | (if test("Z$") then . else . + "Z" end) | fromdateiso8601) end;
  [ .[] | select(.status != "TERMINATED")
    | .started_epoch = (.started_at | iso2epoch)
    | .uptime_s = (if .uptime_s != null then .uptime_s
                   elif .started_epoch != null then ((now - .started_epoch) | floor) else null end)
    | .cost_so_far = (if .uptime_s != null then (.cost_per_hr * .uptime_s / 3600 * 100 | round / 100) else null end)
    | .deadline = ($dl[0][.id] // null)
    | .deadline_epoch = (.deadline | iso2epoch)
    | .overdue = (.deadline_epoch != null and now > .deadline_epoch) ]' "$NORMALIZED")

if [[ $JSON_OUT -eq 1 ]]; then
  printf '%s\n' "$ENRICHED"
  exit 0
fi

count=$(printf '%s' "$ENRICHED" | jq 'length')
log "$count pod(s) (API: $([[ $USE_GRAPHQL -eq 1 ]] && echo graphql || echo rest))"
if [[ $count -gt 0 ]]; then
  {
    printf 'POD_ID\tNAME\tGPU\tSTATUS\tUPTIME_H\t$/H\tCOST_SO_FAR\tDEADLINE\tOVERDUE\n'
    printf '%s' "$ENRICHED" | jq -r '.[] | [
        .id, (.name | if . == "" then "-" else . end), .gpu, .status,
        (if .uptime_s == null then "?" else (.uptime_s / 36 | round / 100 | tostring) end),
        (.cost_per_hr | tostring),
        (if .cost_so_far == null then "?" else (.cost_so_far | tostring) end),
        (.deadline // "-"), (if .overdue then "YES" else "no" end)
      ] | @tsv'
  } | column -t -s "$(printf '\t')"
fi

# ---------------------------------------------------------------- choose victims
if [[ $KILL_ALL -eq 1 ]]; then
  VICTIMS=$(printf '%s' "$ENRICHED" | jq -r '.[].id')
  REASON="--kill-all"
else
  VICTIMS=$(printf '%s' "$ENRICHED" | jq -r '.[] | select(.overdue) | .id')
  REASON="past deadline"
fi
if [[ -n $ONLY_POD ]]; then
  VICTIMS=$(printf '%s\n' "$VICTIMS" | grep -Fx "$ONLY_POD" || true)
fi
VICTIMS=$(printf '%s\n' "$VICTIMS" | sed '/^$/d')

if [[ -z $VICTIMS ]]; then
  log "nothing to terminate ($REASON)"
  exit 0
fi

n_victims=$(printf '%s\n' "$VICTIMS" | wc -l | tr -d ' ')
if [[ $APPLY -eq 0 ]]; then
  while IFS= read -r pod; do
    log "DRY RUN: would terminate pod $pod ($REASON); re-run with --apply"
  done <<<"$VICTIMS"
  exit 0
fi

if [[ $YES -eq 0 ]]; then
  printf 'Terminate %s pod(s) [%s]: %s ? [y/N] ' "$n_victims" "$REASON" "$(printf '%s' "$VICTIMS" | tr '\n' ' ')"
  if [[ -r /dev/tty ]]; then read -r answer </dev/tty; else read -r answer; fi
  [[ $answer == y || $answer == Y || $answer == yes ]] || { log "aborted"; exit 0; }
fi

terminate_pod() { # terminate_pod <id>; 0 on success
  local pod=$1 code out
  out=$(mktemp)
  if [[ $USE_GRAPHQL -eq 0 ]]; then
    # DELETE https://rest.runpod.io/v1/pods/{podId} -> 204
    code=$(printf 'url = "%s/pods/%s"\nrequest = "DELETE"\nheader = "Authorization: Bearer %s"\n' \
      "$RUNPOD_REST_URL" "$pod" "$API_KEY" | curl_cfg "$out")
    if [[ $code == 2* ]]; then rm -f "$out"; return 0; fi
    log "DELETE /pods/$pod -> HTTP $code $(cut -c1-200 "$out" | tr -d '\n')"
  else
    local body
    body=$(mktemp)
    printf '{"query":"mutation { podTerminate(input: {podId: \\"%s\\"}) }"}' "$pod" >"$body"
    code=$(printf 'url = "%s?api_key=%s"\nrequest = "POST"\nheader = "Content-Type: application/json"\nheader = "Authorization: Bearer %s"\ndata = "@%s"\n' \
      "$RUNPOD_GRAPHQL_URL" "$API_KEY" "$API_KEY" "$body" | curl_cfg "$out")
    rm -f "$body"
    if [[ $code == 2* ]] && ! grep -q '"errors"' "$out"; then rm -f "$out"; return 0; fi
    log "podTerminate($pod) -> HTTP $code $(cut -c1-200 "$out" | tr -d '\n')"
  fi
  rm -f "$out"
  return 1
}

failed=0
while IFS= read -r pod; do
  if terminate_pod "$pod"; then
    log "terminated pod $pod ($REASON)"
    printf '%s terminated %s (%s)\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$pod" "$REASON" >>"$LOG_FILE"
  else
    log "FAILED to terminate pod $pod; do it in the console: https://console.runpod.io/pods"
    printf '%s FAILED %s (%s)\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$pod" "$REASON" >>"$LOG_FILE"
    failed=1
  fi
done <<<"$VICTIMS"
exit "$failed"
