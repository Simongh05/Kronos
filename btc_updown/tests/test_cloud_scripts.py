"""Tests for btc_updown/cloud/*.sh.

Everything runs locally on the Mac (or any box with bash, curl, jq): no RunPod
account is touched. The RunPod API is replaced by a tiny HTTP server on
127.0.0.1 through the RUNPOD_REST_URL / RUNPOD_GRAPHQL_URL overrides.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

CLOUD = Path(__file__).resolve().parents[1] / "cloud"
SCRIPTS = sorted(CLOUD.glob("*.sh"))
RUN_JOB = CLOUD / "run_job.sh"
WATCHDOG = CLOUD / "watchdog.sh"
BASH = shutil.which("bash")
FAKE_KEY = "fake-runpod-key-MUST-NOT-APPEAR-ANYWHERE"
FAKE_POD = "fakepod123"

pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")


# ----------------------------------------------------------------- static checks
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_bash_syntax(script: Path) -> None:
    assert SCRIPTS, "no scripts found"
    subprocess.run([BASH, "-n", str(script)], check=True)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_shellcheck(script: Path) -> None:
    shellcheck = shutil.which("shellcheck")
    if shellcheck is None:
        pytest.skip("shellcheck not installed")
    result = subprocess.run([shellcheck, "-x", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_expected_files_exist() -> None:
    for name in ("bootstrap.sh", "run_job.sh", "watchdog.sh", "sync.sh", "README.md"):
        assert (CLOUD / name).is_file(), name
    for script in SCRIPTS:
        head = script.read_text().splitlines()[:80]
        assert head[0] == "#!/usr/bin/env bash", script.name
        assert any(line.startswith("set -euo pipefail") for line in head), script.name


# ----------------------------------------------------------------- fake RunPod API
class _Handler(BaseHTTPRequestHandler):
    server: "FakeRunPod"

    def log_message(self, *_args) -> None:  # silence
        return

    def _record(self, body: str) -> None:
        self.server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "auth": self.headers.get("Authorization", ""),
                "body": body,
            }
        )

    def _send(self, code: int, payload) -> None:
        data = json.dumps(payload).encode() if payload is not None else b""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

    def do_GET(self) -> None:
        self._record("")
        if self.path.startswith("/v1/pods"):
            self._send(200, self.server.pods)
        else:
            self._send(404, {"error": "not found"})

    def do_DELETE(self) -> None:
        self._record("")
        if self.path.startswith("/v1/pods/"):
            code = self.server.delete_status
            self._send(code, None if code == 204 else {"error": "boom"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode() if length else ""
        self._record(body)
        if self.path.startswith("/graphql"):
            self._send(200, self.server.graphql_response)
        else:
            self._send(404, {"error": "not found"})


class FakeRunPod(HTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests: list[dict] = []
        self.pods: list[dict] = []
        self.delete_status = 204
        self.graphql_response = {"data": {"podTerminate": None}}
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    @property
    def rest_url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/v1"

    @property
    def graphql_url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/graphql"

    def close(self) -> None:
        self.shutdown()
        self.server_close()


@pytest.fixture
def api() -> FakeRunPod:
    server = FakeRunPod()
    yield server
    server.close()


# ----------------------------------------------------------------- run_job fixtures
FAKE_MODULE = textwrap.dedent(
    """
    import json, pathlib, sys, time
    args = sys.argv[1:]
    cfg = pathlib.Path(args[args.index("--config") + 1])
    (cfg.parent / "argv_{name}.json").write_text(json.dumps(args))
    time.sleep({sleep})
    sys.exit({code})
    """
)


class JobEnv:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        pkg = tmp_path / "fakejob"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        for name, sleep, code in (("ok", 2, 0), ("fail", 0.1, 3), ("slow", 30, 0)):
            (pkg / f"{name}.py").write_text(FAKE_MODULE.format(name=name, sleep=sleep, code=code))
        self.exp_name = "dryrun_test"
        self.run_dir = tmp_path / "runs" / self.exp_name
        self.config = tmp_path / "cfg.yaml"
        self.config.write_text(
            f"exp_name: {self.exp_name}\nrun_dir: {self.run_dir}\nexpected_hours: 0.001\nmax_steps: 50\n"
        )
        self.env = os.environ.copy()
        self.env.update(
            {
                "PYTHON": sys.executable,
                "PYTHONPATH": str(tmp_path),
                "KRONOS_DIR": str(tmp_path),
                "KRONOS_WORKSPACE": str(tmp_path),
                "RUNPOD_API_KEY": FAKE_KEY,
                "RUNPOD_POD_ID": FAKE_POD,
                # Point the API at something that refuses connections unless a test overrides it.
                "RUNPOD_REST_URL": "http://127.0.0.1:9/v1",
                "RUNPOD_GRAPHQL_URL": "http://127.0.0.1:9/graphql",
                "KILL_GRACE_SECONDS": "2",
                "PATH": os.environ.get("PATH", ""),
            }
        )
        self.env.pop("RUNPOD_TERMINATE_KEY", None)

    def run(self, module: str, *extra: str, timeout: float = 60) -> subprocess.CompletedProcess:
        cmd = [BASH, str(RUN_JOB), f"fakejob.{module}", str(self.config), *extra]
        return subprocess.run(cmd, env=self.env, capture_output=True, text=True, timeout=timeout)

    def status(self) -> dict:
        text = (self.run_dir / "job.status").read_text()
        return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)

    def wait_terminal(self, limit: float = 25) -> dict:
        deadline = time.time() + limit
        while time.time() < deadline:
            if (self.run_dir / "job.status").exists():
                st = self.status()
                if st.get("status") != "running":
                    return st
            time.sleep(0.2)
        raise AssertionError("job did not reach a terminal status in time")

    def argv(self, name: str) -> list:
        return json.loads((self.config.parent / f"argv_{name}.json").read_text())

    def all_text(self) -> str:
        parts = []
        for p in self.run_dir.glob("*"):
            if p.is_file():
                parts.append(p.read_text(errors="replace"))
        return "\n".join(parts)


@pytest.fixture
def job(tmp_path: Path) -> JobEnv:
    return JobEnv(tmp_path)


# ----------------------------------------------------------------- run_job tests
def test_run_job_usage_error(job: JobEnv) -> None:
    result = subprocess.run([BASH, str(RUN_JOB)], env=job.env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Usage" in result.stderr or "need <python-module>" in result.stderr


def test_run_job_success_foreground(job: JobEnv) -> None:
    t0 = time.time()
    result = job.run("ok", "--no-terminate", "--foreground")
    assert result.returncode == 0, result.stdout + result.stderr
    assert time.time() - t0 >= 2
    st = job.status()
    assert st["status"] == "succeeded"
    assert st["exit_code"] == "0"
    assert st["exp_name"] == job.exp_name
    assert st["pod_id"] == FAKE_POD
    assert st["terminate_pod"] == "0"
    assert st["started_at"].endswith("Z") and st["finished_at"].endswith("Z")
    assert int(st["elapsed_seconds"]) >= 2
    # 1.5 x expected_hours(0.001 h) = 5.4 s
    assert abs(float(st["max_seconds"]) - 5.4) < 0.01
    assert "--config" in job.argv("ok") and "--resume" not in job.argv("ok")
    assert "--no-terminate given" in result.stdout
    assert FAKE_KEY not in result.stdout + result.stderr + job.all_text()


def test_run_job_failure_and_resume_passthrough(job: JobEnv) -> None:
    result = job.run("fail", "--no-terminate", "--foreground", "--resume")
    assert result.returncode == 3
    st = job.status()
    assert st["status"] == "failed"
    assert st["exit_code"] == "3"
    assert st["resume"] == "1"
    assert job.argv("fail")[-1] == "--resume"


def test_run_job_passes_extra_cli_args_after_double_dash(job: JobEnv) -> None:
    result = job.run("fail", "--no-terminate", "--foreground", "--resume", "--", "--max-steps", "50", "--set", "train.steps=20")
    assert result.returncode == 3
    assert job.argv("fail")[-5:] == ["--resume", "--max-steps", "50", "--set", "train.steps=20"]
    assert job.status()["extra_args"] == "--max-steps 50 --set train.steps=20"


def test_run_job_expands_exp_name_placeholder_and_relative_run_dir(job: JobEnv) -> None:
    job.config.write_text("exp_name: smoke_x\nrun_dir: data/runs/{exp_name}\nexpected_hours: 0.01\n")
    result = job.run("fail", "--no-terminate", "--foreground")
    assert result.returncode == 3
    status = job.root / "data" / "runs" / "smoke_x" / "job.status"
    assert status.exists(), result.stdout + result.stderr
    assert "run_dir=" + str(job.root / "data" / "runs" / "smoke_x") in status.read_text()


def test_run_job_timeout(job: JobEnv) -> None:
    t0 = time.time()
    result = job.run("slow", "--no-terminate", "--foreground", "--max-hours", "0.0003")
    elapsed = time.time() - t0
    assert result.returncode == 124, result.stdout + result.stderr
    assert elapsed < 20, f"timeout path took {elapsed:.1f}s"
    st = job.status()
    assert st["status"] == "timeout"
    assert st["exit_code"] == "124"
    assert st["budget_source"] == "--max-hours"
    assert abs(float(st["max_seconds"]) - 1.08) < 0.01
    assert not (job.run_dir / ".timeout_fired").exists()


def test_run_job_detaches_with_nohup(job: JobEnv) -> None:
    t0 = time.time()
    result = job.run("ok", "--no-terminate")
    assert result.returncode == 0, result.stdout + result.stderr
    assert time.time() - t0 < 2, "launcher should return before the job finishes"
    assert "started pid" in result.stdout
    pid = int((job.run_dir / "job.pid").read_text().strip())
    assert pid > 0
    st = job.wait_terminal()
    assert st["status"] == "succeeded"
    assert st["exit_code"] == "0"
    log = (job.run_dir / "job.log").read_text()
    assert "job succeeded" in log
    # A second launch while running is refused: start a slow one, then try again.
    slow = job.run("slow", "--no-terminate")
    assert slow.returncode == 0
    time.sleep(0.5)
    again = job.run("slow", "--no-terminate")
    assert again.returncode == 2 and "already running" in again.stderr
    subprocess.run(["pkill", "-TERM", "-f", "fakejob.slow"], check=False)
    job.wait_terminal()


def test_run_job_terminates_pod_via_rest(job: JobEnv, api: FakeRunPod) -> None:
    job.env["RUNPOD_REST_URL"] = api.rest_url
    job.env["RUNPOD_GRAPHQL_URL"] = api.graphql_url
    result = job.run("fail", "--foreground")
    assert result.returncode == 3
    deletes = [r for r in api.requests if r["method"] == "DELETE"]
    assert deletes == [
        {"method": "DELETE", "path": f"/v1/pods/{FAKE_POD}", "auth": f"Bearer {FAKE_KEY}", "body": ""}
    ]
    assert not [r for r in api.requests if r["method"] == "POST"], "GraphQL must not be used when REST works"
    assert "terminated via REST DELETE" in result.stdout
    assert job.status()["terminate_pod"] == "1"
    assert FAKE_KEY not in result.stdout + result.stderr + job.all_text()


def test_run_job_prefers_terminate_key_and_falls_back_to_graphql(job: JobEnv, api: FakeRunPod) -> None:
    api.delete_status = 500
    job.env["RUNPOD_REST_URL"] = api.rest_url
    job.env["RUNPOD_GRAPHQL_URL"] = api.graphql_url
    job.env["RUNPOD_TERMINATE_KEY"] = "restricted-" + FAKE_KEY
    result = job.run("ok", "--foreground")
    assert result.returncode == 0
    methods = [r["method"] for r in api.requests]
    assert methods == ["DELETE", "POST"]
    post = api.requests[1]
    assert post["auth"] == "Bearer restricted-" + FAKE_KEY
    assert post["path"] == "/graphql?api_key=restricted-" + FAKE_KEY
    assert "podTerminate" in post["body"] and FAKE_POD in post["body"]
    assert "terminated via GraphQL" in result.stdout
    assert FAKE_KEY not in result.stdout + result.stderr + job.all_text()


def test_run_job_without_pod_id_does_not_call_api(job: JobEnv, api: FakeRunPod) -> None:
    job.env["RUNPOD_REST_URL"] = api.rest_url
    job.env.pop("RUNPOD_POD_ID")
    result = job.run("fail", "--foreground")
    assert result.returncode == 3
    assert api.requests == []
    assert "nothing to terminate" in result.stdout


# ----------------------------------------------------------------- watchdog tests
def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


@pytest.fixture
def watchdog_env(tmp_path: Path, api: FakeRunPod) -> dict:
    now = datetime.now(timezone.utc)
    api.pods = [
        {
            "id": FAKE_POD,
            "name": "kronos-e1b",
            "gpu": {"id": "NVIDIA GeForce RTX 4090", "count": 1, "displayName": "RTX 4090"},
            "costPerHr": 0.69,
            "desiredStatus": "RUNNING",
            "lastStartedAt": _iso(now - timedelta(hours=2)),
        },
        {
            "id": "freshpod456",
            "name": "kronos-e0",
            "gpu": {"displayName": "RTX 4090"},
            "costPerHr": "0.69",
            "desiredStatus": "RUNNING",
            "lastStartedAt": _iso(now - timedelta(minutes=5)),
        },
        {"id": "deadpod", "name": "old", "desiredStatus": "TERMINATED", "costPerHr": 0},
    ]
    env_file = tmp_path / ".env"
    env_file.write_text(f"# comment\nOTHER=1\nRUNPOD_API_KEY=\"{FAKE_KEY}\"\n")
    deadlines = tmp_path / "cloud" / "deadlines.json"
    deadlines.parent.mkdir()
    deadlines.write_text(json.dumps({FAKE_POD: _iso(now - timedelta(minutes=1)), "freshpod456": _iso(now + timedelta(hours=3))}))
    env = os.environ.copy()
    env.pop("RUNPOD_API_KEY", None)
    env["RUNPOD_REST_URL"] = api.rest_url
    env["RUNPOD_GRAPHQL_URL"] = api.graphql_url
    return {"env": env, "env_file": env_file, "deadlines": deadlines}


def _watchdog(watchdog_env: dict, *args: str) -> subprocess.CompletedProcess:
    cmd = [BASH, str(WATCHDOG), "--env-file", str(watchdog_env["env_file"]), "--deadlines", str(watchdog_env["deadlines"]), *args]
    return subprocess.run(cmd, env=watchdog_env["env"], capture_output=True, text=True, timeout=60)


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")
def test_watchdog_dry_run_lists_and_flags_overdue(watchdog_env: dict, api: FakeRunPod) -> None:
    result = _watchdog(watchdog_env)
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert FAKE_POD in out and "freshpod456" in out and "deadpod" not in out
    assert "RTX 4090" in out
    assert f"DRY RUN: would terminate pod {FAKE_POD}" in out
    assert "would terminate pod freshpod456" not in out
    assert api.requests[0]["auth"] == f"Bearer {FAKE_KEY}"
    assert not [r for r in api.requests if r["method"] == "DELETE"]
    assert FAKE_KEY not in out + result.stderr
    # --json: uptime ~2h and cost so far ~1.38 for the overdue pod
    js = json.loads(_watchdog(watchdog_env, "--json").stdout)
    pod = next(p for p in js if p["id"] == FAKE_POD)
    assert pod["overdue"] is True and 7100 <= pod["uptime_s"] <= 7300
    assert abs(pod["cost_so_far"] - 1.38) < 0.05
    assert next(p for p in js if p["id"] == "freshpod456")["overdue"] is False


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")
def test_watchdog_apply_terminates_overdue_only(watchdog_env: dict, api: FakeRunPod) -> None:
    result = _watchdog(watchdog_env, "--apply", "--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    deletes = [r for r in api.requests if r["method"] == "DELETE"]
    assert [d["path"] for d in deletes] == [f"/v1/pods/{FAKE_POD}"]
    assert deletes[0]["auth"] == f"Bearer {FAKE_KEY}"
    assert f"terminated pod {FAKE_POD}" in result.stdout
    log = (watchdog_env["deadlines"].parent / "watchdog.log").read_text()
    assert FAKE_POD in log and FAKE_KEY not in log


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")
def test_watchdog_kill_all_and_failure_exit_code(watchdog_env: dict, api: FakeRunPod) -> None:
    result = _watchdog(watchdog_env, "--kill-all", "--apply", "--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    deletes = sorted(r["path"] for r in api.requests if r["method"] == "DELETE")
    assert deletes == [f"/v1/pods/{FAKE_POD}", "/v1/pods/freshpod456"]
    api.requests.clear()
    api.delete_status = 500
    result = _watchdog(watchdog_env, "--kill-all", "--apply", "--yes", "--pod", "freshpod456")
    assert result.returncode == 1
    assert [r["path"] for r in api.requests if r["method"] == "DELETE"] == ["/v1/pods/freshpod456"]
    assert "FAILED to terminate pod freshpod456" in result.stdout


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")
def test_watchdog_set_deadline(watchdog_env: dict) -> None:
    result = _watchdog(watchdog_env, "--set-deadline", "newpod789", "2.5")
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(watchdog_env["deadlines"].read_text())
    assert set(data) == {FAKE_POD, "freshpod456", "newpod789"}
    deadline = datetime.strptime(data["newpod789"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    delta = deadline - datetime.now(timezone.utc)
    assert timedelta(hours=2.4) < delta < timedelta(hours=2.6)


def test_watchdog_refuses_without_key(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("RUNPOD_API_KEY", None)
    missing = tmp_path / "none.env"
    result = subprocess.run([BASH, str(WATCHDOG), "--env-file", str(missing)], env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "no env file" in result.stderr
