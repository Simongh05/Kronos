# RunPod cloud scripts for `btc_updown`

Everything GPU-related runs on one RunPod **Secure Cloud** RTX 4090 pod at a time, with a
100 GB network volume mounted at `/workspace`. Total budget for the exploration is about
$100, so every job is unattended, resumable, time-boxed and **terminates its own pod**.
The Mac keeps a second line of defence (`watchdog.sh`) and prepaid credits are the hard stop.

| File | Runs on | Purpose |
|---|---|---|
| `bootstrap.sh` | pod | install tools, clone/update the repo, pip deps, download klines, smoke test, print GPU/disk |
| `run_job.sh` | pod | wrap one CLI under `timeout`, log to the volume, write `job.status`, terminate the pod on exit |
| `watchdog.sh` | Mac | list pods with cost so far; terminate pods past their deadline (`--apply`), or all (`--kill-all`) |
| `sync.sh` | Mac | rsync `/workspace/{runs,cubes}` down to `data/cloud/`, push `btc_updown/configs/` up |
| `com.kronos.runpod-watchdog.plist` | Mac | launchd agent that runs the watchdog every 15 minutes |
| `../tests/test_cloud_scripts.py` | Mac | `bash -n`, shellcheck, and end-to-end runs of `run_job.sh`/`watchdog.sh` against a fake RunPod API |

Nothing in this folder has been run against a real RunPod account yet (none existed when
it was written). See [What must be confirmed once the account exists](#what-must-be-confirmed-once-the-account-exists).

---

## 1. Account setup (user, once)

1. **Create the account** at <https://www.runpod.io> (email + password or GitHub). Enable 2FA.
2. **Prepaid credits are the hard stop.** Billing → add credits once, e.g. $100 (or $50 to start:
   the first week needs < $20). Do **not** enable auto-pay / auto-recharge. When the balance hits
   zero RunPod stops the pods; the network volume survives for a grace period but is also at risk,
   so keep a few dollars of headroom and pull results with `sync.sh` after every job.
3. **Spend alert.** Settings → Notifications: turn on low-balance e-mails and set the threshold
   to about half the budget. RunPod also shows "time remaining" on the balance; glance at it when
   a pod is running.
4. **SSH key.** On the Mac, if `~/.ssh/id_ed25519.pub` does not exist: `ssh-keygen -t ed25519`.
   Paste the public key into Settings → SSH Public Keys. RunPod injects it into every pod as
   `PUBLIC_KEY`, and the `runpod/pytorch` images start `sshd` with it.
5. **API keys** (Settings → API Keys, <https://console.runpod.io/user/credentials>). Keys are
   shown once; store them immediately.
   * **Mac key** — permission *All*; name it `mac-watchdog`. Put it in
     `/Users/streetteam/Kronos/.env` as
     ```
     RUNPOD_API_KEY=rpa_xxxxxxxx
     ```
     `.env` is git-ignored. `watchdog.sh` reads it; nothing else does.
   * **Pod termination key** — a second, separate key named `pod-self-terminate`, so it can be
     revoked without touching the Mac key. RunPod's "Restricted" permission level is scoped to
     *Serverless endpoints only* (checked on docs.runpod.io 2026-10-25), so a key that may delete
     pods must currently be *All* as well; the separation is organisational, not a permission
     boundary. Store it as a **pod secret**: Settings → Secrets → Add Secret, name
     `pod_terminate_key` (it becomes `RUNPOD_SECRET_pod_terminate_key`). In the pod template map it
     to the environment variable `RUNPOD_TERMINATE_KEY={{ RUNPOD_SECRET_pod_terminate_key }}`.
   * RunPod also injects a **pod-scoped `RUNPOD_API_KEY`** into every pod and ships `runpodctl`
     configured with it. `run_job.sh` falls back to that key and to `runpodctl pod delete` when
     `RUNPOD_TERMINATE_KEY` is absent. If the dry run (section 4) shows the pod-scoped key is enough
     to delete its own pod, the second key is unnecessary and can be deleted.

## 2. Network volume (once)

Storage → Network Volumes → New. Network volumes exist only in **Secure Cloud**, and a pod can
mount only a volume from its own datacenter, so pick the datacenter first: filter the
datacenter list for one that lists **RTX 4090** availability (several EU and US sites do; check
*Secure Cloud* availability, not Community). Size **100 GB** (≈ $0.07/GB/month ≈ $7/month; the
klines are ~15 GB, checkpoints a few GB each). Name `kronos`. It mounts at `/workspace` in
every pod created with it; `/workspace` is the only path that survives pod termination.

## 3. Pod template (once)

Templates → New Template (Pod):

| Setting | Value |
|---|---|
| Name | `kronos-4090` |
| Container image | a current `runpod/pytorch` tag with torch ≥ 2.4 and CUDA 12.x, e.g. `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` (pick from the list the console offers; any 2.4+/cu12x works) |
| Container disk | 20 GB (pip cache, apt, scratch; the volume holds data) |
| Volume mount path | `/workspace` |
| Expose TCP ports | `22` (needed for `sync.sh`; HTTP 8888 can stay) |
| Start command | leave the image default (starts sshd and jupyter); we SSH in |
| Environment variables | `REPO_URL=https://github.com/<your-fork>/Kronos.git` <br> `RUNPOD_TERMINATE_KEY={{ RUNPOD_SECRET_pod_terminate_key }}` <br> `GIT_TOKEN={{ RUNPOD_SECRET_git_token }}` only if the fork is private (fine-grained PAT, contents: read) |

Branch `btc-updown` must be pushed to that fork; `bootstrap.sh` clones it.

**Deploying a pod** (every job): Pods → Deploy → *Secure Cloud* → datacenter of the volume →
RTX 4090 → template `kronos-4090` → attach network volume `kronos` → select the SSH key → Deploy.
Expected price ≈ $0.69/h (Secure Cloud; Community 4090s are cheaper but cannot mount the volume).
Then immediately on the Mac:

```bash
# records a deadline so the watchdog will kill the pod if the job's own trap fails
btc_updown/cloud/watchdog.sh --set-deadline <pod_id> <expected_hours * 1.5 + 0.5>
```

## 4. First boot and the 10-minute dry run

Connect: the pod card → Connect → "SSH over exposed TCP" shows `ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519`.
(The `ssh.runpod.io` proxy form works for a shell but not for rsync.)

```bash
# on the pod
cd /workspace
curl -fsSL https://raw.githubusercontent.com/<your-fork>/Kronos/btc-updown/btc_updown/cloud/bootstrap.sh -o bootstrap.sh
REPO_URL=https://github.com/<your-fork>/Kronos.git bash bootstrap.sh
```

`bootstrap.sh` is idempotent. On the first pod it downloads 2019–2026 klines for three series
(~15 GB from data.binance.vision, 10–20 minutes) into `/workspace/klines`; later pods skip that
because `/workspace/klines/spot_BTCUSDC/meta.json` exists. It ends with the GPU name, torch and
CUDA versions, `bf16 supported True`, and free space on `/workspace`. It also writes
`/workspace/env.sh` (`HF_HOME=/workspace/hf`, so pretrained weights are downloaded once).

**Dry run** (≈ 10 min of pod time, < $0.15): 50 training steps of Kronos-small, then
self-termination.

```bash
# on the pod
source /workspace/env.sh
cd /workspace/Kronos
btc_updown/cloud/run_job.sh btc_updown.train.train_predictor btc_updown/configs/ft_small_1m.yaml \
    --max-hours 0.17 -- --max-steps 50 --set data.root=/workspace/klines --set exp_name=dryrun_small
tail -f /workspace/runs/dryrun_small/job.log      # until "job succeeded" ... "pod ... terminated"
```

Notes: the config's `run_dir: /workspace/runs/{exp_name}` is expanded by `run_job.sh`; everything
after `--` goes to the CLI unchanged; `--max-hours 0.17` time-boxes the run to ~10 minutes (the
default is 1.5 × `expected_hours` from the YAML). While it runs, on the Mac:

```bash
btc_updown/cloud/watchdog.sh            # should list the pod, its $/h and cost so far
```

**Verify the stop** (do this once, carefully, before trusting the trap with multi-hour jobs):

1. The SSH session drops within a few seconds of the "terminated via ..." log line.
2. Console → Pods: the pod is gone (a terminated pod disappears from the list within a minute;
   `watchdog.sh` shows `0 pod(s)`).
3. Console → Billing → usage/history: the pod's charge stops at roughly 10 minutes × $0.69/h ≈ $0.12,
   and the balance is no longer decreasing an hour later.
4. Pull the artefacts and read the status file:
   ```bash
   # the pod is gone, so the files come from the next pod, or create a cheap CPU pod on the volume
   btc_updown/cloud/sync.sh <ip> <port>
   cat data/cloud/runs/dryrun_small/job.status       # status=succeeded exit_code=0 ...
   ```
   (`job.status` is key=value lines: `status`, `exit_code`, `started_at`, `finished_at`,
   `elapsed_seconds`, `max_seconds`, `pod_id`, ...)

If the pod is **still running** after step 1: `btc_updown/cloud/watchdog.sh --kill-all --apply`
on the Mac, then read `/workspace/runs/dryrun_small/terminate.err` and `job.log` on the next pod to
see which of the three termination paths failed (section 7).

## 5. Running real jobs

```bash
# on the pod (after bootstrap.sh; each pod starts fresh, the volume does not)
source /workspace/env.sh && cd /workspace/Kronos
btc_updown/cloud/run_job.sh btc_updown.train.train_predictor btc_updown/configs/ft_small_1m.yaml --resume
btc_updown/cloud/run_job.sh btc_updown.infer.cube           btc_updown/configs/cube_e0_small_val.yaml
btc_updown/cloud/run_job.sh btc_updown.train.train_tokenizer btc_updown/configs/<tok>.yaml --resume
```

`run_job.sh` detaches itself with `nohup` (closing SSH is safe), prints the pid and the log path,
and the detached copy:

* runs `python -m <module> --config <yaml> [--resume] [<extra args>]` under GNU `timeout`
  with a budget of 1.5 × `expected_hours` (`--max-hours` overrides; SIGTERM, then SIGKILL after 60 s),
* writes `<run_dir>/job.status` with `status=running` at start and `succeeded` / `failed` /
  `timeout` plus `exit_code` and timestamps at the end (atomic temp-then-rename),
* on EXIT (success, failure, timeout or signal): `sync -f /workspace`, then terminates the pod
  via REST → GraphQL → `runpodctl`, in that order, stopping at the first success.
  `--no-terminate` keeps the pod for debugging.
* refuses to start a second job for the same `exp_name` while one is running.

A job that timed out or was killed is resumed by deploying a new pod and re-running the same
command with `--resume` (the trainers checkpoint every `ckpt_every` steps; the cube is resumable
per day). Always add a watchdog deadline right after deploying.

Set `data.root: /workspace/klines` in pod configs (the defaults point at the Mac path).

## 6. Mac side: watchdog and sync

```bash
btc_updown/cloud/watchdog.sh                         # dry run: table + "would terminate" lines
btc_updown/cloud/watchdog.sh --apply                 # terminate overdue pods, asks y/N
btc_updown/cloud/watchdog.sh --apply --yes           # no prompt (what launchd runs)
btc_updown/cloud/watchdog.sh --kill-all --apply      # panic button
btc_updown/cloud/watchdog.sh --set-deadline <pod> 5  # deadline = now + 5 h
btc_updown/cloud/watchdog.sh --json                  # machine-readable list
```

Deadlines live in `data/cloud/deadlines.json` as `{"<pod_id>": "2026-10-26T18:00:00Z"}` (UTC).
Actions are appended to `data/cloud/watchdog.log`. The key is read from `.env` and never printed.

launchd (every 15 minutes, terminates overdue pods without asking):

```bash
cp btc_updown/cloud/com.kronos.runpod-watchdog.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.kronos.runpod-watchdog.plist
launchctl kickstart -k gui/$(id -u)/com.kronos.runpod-watchdog    # run once now
tail -f data/cloud/watchdog.launchd.log
```

Sync (pull `runs/` + `cubes/`, push `configs/`; resumable, nothing deleted):

```bash
btc_updown/cloud/sync.sh <ip> <port>                 # key: ~/.ssh/id_ed25519 (or --key / RUNPOD_SSH_KEY)
btc_updown/cloud/sync.sh <ip> <port> --pull-only --dry-run
```

macOS ships `openrsync`, which supports the flags used (`-az --partial --inplace --exclude`);
`brew install rsync` gives progress output with `--verbose`. Pods reuse IP:port pairs, so host keys
are kept in `data/cloud/known_hosts`; on a "host key changed" error delete that line or pass
`--no-host-check`.

## 7. RunPod API endpoints used (verify at setup time)

I could not call any of these live. They were read from the public docs on 2026-10-25; re-check
them against <https://docs.runpod.io> before the first real job, and watch the dry run closely.

| Used by | Call | Source |
|---|---|---|
| `watchdog.sh` list | `GET https://rest.runpod.io/v1/pods`, header `Authorization: Bearer <key>`; array of pods with `id`, `name`, `gpu.displayName`, `costPerHr`, `desiredStatus`, `lastStartedAt`, `publicIp`, `portMappings` | <https://docs.runpod.io/api-reference/pods/GET/pods> |
| `run_job.sh`, `watchdog.sh` terminate | `DELETE https://rest.runpod.io/v1/pods/{podId}` → `204` | <https://docs.runpod.io/api-reference/pods/DELETE/pods/podId> |
| fallback | `POST https://api.runpod.io/graphql?api_key=<key>` with `mutation { podTerminate(input: {podId: "<id>"}) }` (`PodTerminateInput { podId: String! }`, returns `Void`); `--graphql` listing uses `query { myself { pods { id name costPerHr desiredStatus lastStartedAt machine { gpuDisplayName } runtime { uptimeInSeconds } } } }` | <https://docs.runpod.io/sdks/graphql/manage-pods>, <https://graphql-spec.runpod.io/> |
| last resort | `runpodctl pod delete <id>` (preinstalled on pods with a pod-scoped key; legacy `runpodctl remove pod <id>` tried second) | <https://github.com/runpod/runpodctl/blob/main/README.md> |
| env inside pods | `RUNPOD_POD_ID`, `RUNPOD_API_KEY` (pod-scoped), `RUNPOD_PUBLIC_IP`, `RUNPOD_TCP_PORT_22`, `RUNPOD_GPU_COUNT`, `RUNPOD_DC_ID`, `PUBLIC_KEY` | <https://docs.runpod.io/pods/references/environment-variables> |
| secrets | `RUNPOD_SECRET_<name>`, referenced as `{{ RUNPOD_SECRET_<name> }}` in template env vars | <https://docs.runpod.io/pods/templates/secrets>, <https://docs.runpod.io/get-started/api-keys> |
| SSH | public-IP form `ssh root@<ip> -p <port> -i <key>` supports scp/rsync; the `ssh.runpod.io` proxy does not | <https://docs.runpod.io/pods/configuration/use-ssh> |
| volumes | Secure Cloud only, same datacenter as the pod, mounted at `/workspace`, $0.07/GB/month | <https://docs.runpod.io/pods/storage/network-volumes> |

Implementation details that matter for the review: the API key is passed to `curl` through a
config read on stdin (`-K -`), so it never appears in `ps` or shell traces; `GIT_TOKEN` is handed
to git via `GIT_ASKPASS`, never placed in a URL or `.git/config`; `RUNPOD_REST_URL` /
`RUNPOD_GRAPHQL_URL` can be pointed at a local fake server, which is how the tests exercise the
termination paths.

### What must be confirmed once the account exists

1. `DELETE /v1/pods/{id}` returns 2xx with the *pod-scoped* `RUNPOD_API_KEY` from inside the pod.
   If it returns 401/403, the `RUNPOD_TERMINATE_KEY` secret (an *All* key) is required.
2. The GraphQL endpoint accepts the key as `?api_key=` (documented) and/or `Authorization: Bearer`
   (both are sent).
3. `GET /v1/pods` field names (`gpu.displayName`, `costPerHr`, `lastStartedAt`) match; the watchdog
   derives uptime from `lastStartedAt` because REST has no uptime field. If names differ, use
   `watchdog.sh --graphql`, which reads `runtime.uptimeInSeconds` directly.
4. `runpodctl pod delete` exists on the chosen image (older images ship a `runpodctl` that only
   knows `remove pod`; both are tried).
5. The image tag, that `python`, `pip`, `nvidia-smi` and `timeout` exist in it (all four do in the
   official `runpod/pytorch` images), and that `apt-get` works as root.
6. The dry run's billing line stops when the pod disappears (section 4, step 3).
7. The secret-mapping syntax `{{ RUNPOD_SECRET_name }}` in template environment variables.

## 8. Cost table (from the plan; secure 4090 at ≈ $0.69/h, FLOP-based ±2×)

| Order | Exp | What | GPU | Cost |
|---|---|---|---|---|
| 0 | Mac controls | proxy flip rate, null pricer Brier on val, tokenizer round-trip | 0 | $0 |
| 1 | E0 | Zero-shot Kronos-small cube on val: 1 point/min, N=128, lookback 256, T grid {0.8, 1.0, 1.3}; calibration fit; c-coefficient test | ~5 h | ~$4 |
| 2 | E1b-probe | Frozen pretrained small backbone → last-position features for all train/val minutes once; DirectHead on Mac/CPU or 20 min GPU | ~0.5 h | ~$0.5 |
| 3 | E1b | Joint fine-tune: CE + DirectHead BCE, predictor only, frozen tokenizer, 3 series, ~30k steps batch 64 bf16, val every 5k steps | ~3 h | ~$2 |
| 4 | E1 | Predictor-only CE fine-tune (sampling path) if E1b shows signal or the round-trip test is clean; cube on val | ~3 h + ~5 h | ~$6 |
| 5 | E-tok | Tokenizer fine-tune, only if the round-trip test fails; then rerun E1b | ~1 h + ~3 h | ~$3 |
| 6 | E2 | Kronos-base version of the best of E1/E1b, only if c's CI excludes 0 on val | ~12 h | ~$9 |
| 7 | DEV | Freeze (model, T, calibration, execution rule); one cube over dev (N=128, lookback 256) and one backtest run | ~7 h | ~$5 |
| | | Storage 2 months (100 GB ≈ $14), dry run, reruns, idle minutes while bootstrapping | | ~$35 |
| | | **Total** | | **≈ $65 of the $100** |

Billing is per minute while the pod exists (also while bootstrapping, downloading, or sitting idle
after a job if termination failed): the trap plus the watchdog plus prepaid credits are the three
layers that keep this table honest.

## 9. Tests

```bash
.venv/bin/python -m pytest btc_updown/tests/test_cloud_scripts.py -q
```

Covers: `bash -n` and shellcheck on every script (shellcheck skipped if not installed);
`run_job.sh` success / failure / `--resume` / extra-arg pass-through / timeout (`--max-hours 0.0003`
→ `status=timeout`, exit 124) / nohup detach / duplicate-job refusal / `{exp_name}` expansion;
pod termination through REST and the GraphQL fallback against a local fake API, including that
the key never appears in any output; `watchdog.sh` listing, overdue detection, `--apply`,
`--kill-all`, `--pod`, `--set-deadline`, failure exit code, and missing-key refusal.
`sync.sh` is only syntax/shellcheck-tested (it needs a live pod).
