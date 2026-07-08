#!/usr/bin/env python3
"""
Queue runner for tasks.csv — runs one training per node across 8 nodes.

- Sorts pending tasks largest-model-first (400M → 50M).
- Launches one task per node in a named tmux session.
- Polls nodes; when a node's session dies, rsyncs results + marks done/failed.
- Assigns the next task to the freed node.
- Continues until all tasks are complete.

Usage:
    python3 run_task_queue.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[6]  # AGI-1699-New-Dataset-Runs/
TASKS_CSV = EXP_DIR / "results" / "optimize" / "base" / "realistic" / "tasks.csv"
DEFAULT_CLUSTER_JSONS = [EXP_DIR / "multinode" / "cluster_one.json"]
REMOTE_DIR = "/workspace/gradient-routing/experiments/AGI-1699-New-Dataset-Runs"
LOCAL_RESULTS = EXP_DIR / "results" / "optimize" / "base" / "realistic"

SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
]
POLL_INTERVAL = 60


def ssh(ip: str, cmd: str, check: bool = False, input_str: str | None = None, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *SSH_OPTS, f"user@{ip}", cmd],
        capture_output=True, text=True, check=check,
        input=input_str, timeout=timeout,
    )


def ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def load_tasks() -> list[dict]:
    with open(TASKS_CSV) as f:
        rows = list(csv.DictReader(f))
    # Normalize any stray whitespace/CR
    for r in rows:
        for k, v in list(r.items()):
            if v is not None:
                r[k] = v.strip()
    return rows


def save_tasks(rows: list[dict]) -> None:
    fieldnames = ["model_size", "seed", "trial_num", "lr", "batch_size", "status", "IP"]
    tmp = TASKS_CSV.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    tmp.replace(TASKS_CSV)


def update_task(rows: list[dict], task: dict, **updates) -> None:
    for r in rows:
        if (r["model_size"] == task["model_size"]
                and r["seed"] == task["seed"]
                and r["trial_num"] == task["trial_num"]):
            r.update({k: str(v) for k, v in updates.items()})
            break
    save_tasks(rows)


def load_nodes(cluster_jsons: list[Path]) -> list[str]:
    nodes = []
    for path in cluster_jsons:
        data = json.loads(Path(path).read_text())
        for n in data:
            ip = n["public_ip"]
            if ip not in nodes:
                nodes.append(ip)
    return nodes


def sort_tasks(rows: list[dict]) -> list[dict]:
    pending = [r for r in rows if r["status"] == "pending"]
    # Largest model first, then seed asc, then trial_num asc
    pending.sort(key=lambda r: (-int(r["model_size"]), int(r["seed"]), int(r["trial_num"])))
    return pending


def session_name(task: dict) -> str:
    return f"opt-{task['model_size']}M-seed{task['seed']}-trial{task['trial_num']}"


def is_session_alive(ip: str, session: str) -> bool | None:
    """Return True if session alive, False if confirmed dead, None if uncertain
    (ssh failed / timed out). Caller should treat None as 'keep polling'."""
    # Use a marker so we can distinguish "ssh worked, session not found" from
    # "ssh failed to respond at all".
    r = ssh(
        ip,
        f"tmux has-session -t {session} 2>/dev/null; echo __RC__=$?",
    )
    if r.returncode != 0:
        return None
    out = r.stdout or ""
    if "__RC__=0" in out:
        return True
    if "__RC__=1" in out:
        return False
    return None


def task_succeeded(ip: str, task: dict) -> bool:
    model_size = task["model_size"]
    seed = task["seed"]
    cmd = (f"grep -l 'Finished\\. See' "
           f"{REMOTE_DIR}/results/optimize/base/realistic/{model_size}M/seed_{seed}/*/training.log 2>/dev/null")
    r = ssh(ip, cmd)
    return bool(r.stdout.strip())


def rsync_results(ip: str, task: dict) -> None:
    model_size = task["model_size"]
    seed = task["seed"]
    local = LOCAL_RESULTS / f"{model_size}M" / f"seed_{seed}"
    local.mkdir(parents=True, exist_ok=True)
    src = f"user@{ip}:{REMOTE_DIR}/results/optimize/base/realistic/{model_size}M/seed_{seed}/"
    subprocess.run([
        "rsync", "-az", "--exclude=*.pth",
        "-e", "ssh " + " ".join(SSH_OPTS),
        src, f"{local}/",
    ], check=False, timeout=600)


def launch_task(ip: str, task: dict) -> str:
    session = session_name(task)
    model_size = task["model_size"]
    param_str = f"{model_size}M"
    seed = task["seed"]
    lr = task["lr"]
    bs = task["batch_size"]
    trial = task["trial_num"]
    logfile = f"/tmp/opt-{param_str}-seed{seed}-trial{trial}.log"

    inner = f"""#!/bin/bash
trap '' HUP
cd {REMOTE_DIR}
export OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH"
exec uv run torchrun --nnodes=1 --nproc_per_node=8 --master_port=29500 \\
  -m src.run.experiment.optimize.base.realistic.run \\
  --model_size {param_str} --seed {seed} --lr {lr} --eff_bs {bs}
"""
    remote_cmd = f"""tmux kill-session -t {session} 2>/dev/null || true
# Clean up any lingering torchrun or training processes holding GPU/port.
# Prior sessions used `trap '' HUP`, so their children survive tmux kill —
# need explicit SIGKILL to free up port 29500 and GPU memory.
pkill -9 -f 'torchrun.*optimize.base.realistic' 2>/dev/null || true
pkill -9 -f 'optimize.base.realistic.run' 2>/dev/null || true
# Wait up to 10s for port 29500 to clear
for i in $(seq 1 10); do
  if ! ss -tln 2>/dev/null | grep -q ':29500 '; then break; fi
  sleep 1
done
cat > /tmp/launch_{session}.sh <<'SCRIPT'
{inner}
SCRIPT
chmod +x /tmp/launch_{session}.sh
tmux new-session -d -s {session} "script -f {logfile} -c 'bash /tmp/launch_{session}.sh'"
"""
    r = subprocess.run(
        ["ssh", *SSH_OPTS, f"user@{ip}", "bash", "-s"],
        input=remote_cmd, capture_output=True, text=True,
        timeout=30,
    )
    if r.returncode != 0:
        log(f"[{ip}] LAUNCH FAILED: {r.stderr.strip()}")
        raise RuntimeError(f"launch failed: {r.stderr}")
    return session


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster", type=Path, nargs="+", default=DEFAULT_CLUSTER_JSONS,
        help="One or more cluster JSON files. Nodes from all are pooled.",
    )
    args = parser.parse_args()

    rows = load_tasks()
    nodes = load_nodes(args.cluster)

    queue = sort_tasks(rows)
    if not queue and not any(r["status"] == "running" for r in rows):
        log("Nothing to do.")
        return 0

    queue_idx = 0
    # node_ip -> (task_dict, session_name) or None
    node_state: dict[str, tuple[dict, str] | None] = {ip: None for ip in nodes}

    # --- Adopt still-running sessions from a previous runner instance ---
    adopted = 0
    orphaned = 0
    for r in rows:
        if r["status"] != "running":
            continue
        ip = r.get("IP", "").strip()
        if not ip:
            # No IP recorded — can't adopt, put back in queue as pending
            update_task(rows, r, status="pending")
            continue
        if ip not in nodes:
            log(f"[{ip}] was running {session_name(r)} but IP not in current cluster — re-queuing")
            update_task(rows, r, status="pending")
            continue
        session = session_name(r)
        try:
            alive = is_session_alive(ip, session)
        except subprocess.TimeoutExpired:
            alive = None
        if alive is None:
            # SSH failed — can't confirm dead, assume alive and let normal polling handle it
            node_state[ip] = (r, session)
            adopted += 1
            log(f"[{ip}] adopted {session} (ssh failed during adoption check, will poll)")
        elif alive:
            node_state[ip] = (r, session)
            adopted += 1
            log(f"[{ip}] adopted live session {session}")
        else:
            # Session died without the runner harvesting — try to rsync + mark based on success
            try:
                succeeded = task_succeeded(ip, r)
            except Exception:
                succeeded = False
            try:
                rsync_results(ip, r)
            except Exception as e:
                log(f"[{ip}] rsync error during adoption: {e}")
            if succeeded:
                update_task(rows, r, status="done", IP=ip)
                log(f"[{ip}] {session} finished while runner was down — marked done")
            else:
                update_task(rows, r, status="pending")
                log(f"[{ip}] {session} died incomplete — re-queuing as pending")
            orphaned += 1

    # Rebuild queue after adoption (tasks re-queued from orphaned sessions)
    queue = sort_tasks(rows)

    log(f"{len(nodes)} nodes, {adopted} adopted, {orphaned} orphaned/harvested, {len(queue)} pending tasks")

    def assign_free_nodes():
        nonlocal queue_idx
        for ip in nodes:
            if node_state[ip] is None and queue_idx < len(queue):
                task = queue[queue_idx]
                queue_idx += 1
                try:
                    session = launch_task(ip, task)
                except Exception as e:
                    log(f"[{ip}] failed to launch {session_name(task)}: {e}")
                    update_task(rows, task, status="failed", IP=ip)
                    continue
                node_state[ip] = (task, session)
                update_task(rows, task, status="running", IP=ip)
                log(f"[{ip}] launched {session_name(task)} (lr={task['lr']} bs={task['batch_size']})")

    assign_free_nodes()

    while True:
        # Poll every node with a live task
        for ip in nodes:
            state = node_state[ip]
            if state is None:
                continue
            task, session = state
            try:
                alive = is_session_alive(ip, session)
            except subprocess.TimeoutExpired:
                log(f"[{ip}] ssh timeout; will retry")
                continue

            if alive is None:
                # ssh failed — don't assume dead; retry next poll
                log(f"[{ip}] ssh check inconclusive for {session}; will retry")
                continue

            if alive:
                continue

            # Session ended — harvest
            label = session_name(task)
            try:
                succeeded = task_succeeded(ip, task)
            except subprocess.TimeoutExpired:
                log(f"[{ip}] timeout checking success; treating as failed")
                succeeded = False

            log(f"[{ip}] {label} session ended — {'SUCCESS' if succeeded else 'FAILED'}, rsyncing")
            try:
                rsync_results(ip, task)
            except subprocess.TimeoutExpired:
                log(f"[{ip}] rsync timed out")
            except Exception as e:
                log(f"[{ip}] rsync error: {e}")

            update_task(rows, task, status=("done" if succeeded else "failed"), IP=ip)
            node_state[ip] = None

        assign_free_nodes()

        # Done?
        any_running = any(v is not None for v in node_state.values())
        if queue_idx >= len(queue) and not any_running:
            log("All tasks complete.")
            break

        time.sleep(POLL_INTERVAL)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("Interrupted — tasks may still be running on nodes.")
        sys.exit(130)
