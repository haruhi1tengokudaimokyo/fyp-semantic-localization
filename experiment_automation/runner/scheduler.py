"""Resource-aware persistent DAG controller. Workers are separate systemd services."""

import argparse, fcntl, json, subprocess, time, hashlib, os
from pathlib import Path
from core import (
    ROOT,
    REPO,
    db,
    init,
    register,
    imported,
    event,
    resources,
    fingerprint,
    atomic,
)

PYTHON = "/data/fyp_3d/envs/macvo/bin/python"


def unit_state(unit):
    p = subprocess.run(
        ["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"],
        capture_output=True,
        text=True,
    )
    return p.stdout.strip()


def tick(root=ROOT):
    now = time.time()
    r = resources()
    with db(root) as c:
        c.execute("INSERT INTO resources VALUES (?,?)", (now, json.dumps(r)))
        for row in c.execute("SELECT * FROM tasks WHERE status='RUNNING'").fetchall():
            if (
                row["unit"]
                and unit_state(row["unit"]) not in ["active", "activating"]
                and now - (row["start"] or now) > 10
            ):
                s = json.loads(row["spec"])
                status = (
                    "RETRYING"
                    if row["attempt"] < s.get("max_attempts", 2)
                    else "FAILED"
                )
                reason = "Independent worker unit exited before a validated completion receipt"
                c.execute(
                    "UPDATE tasks SET status=?,failure=?,retry_at=?,end=? WHERE task_id=?",
                    (status, reason, now + 30, now, row["task_id"]),
                )
                event(c, row["task_id"], "ORPHAN_RECOVERED", reason)
        rows = c.execute("SELECT * FROM tasks").fetchall()
        states = {x["task_id"]: x["status"] for x in rows}
        running = [json.loads(x["spec"]) for x in rows if x["status"] == "RUNNING"]
        gpu_busy = any(s.get("resources", {}).get("gpu", False) for s in running)
        cpu_running = sum(s.get("resources", {}).get("cpu_slots", 1) for s in running)
        downloads = sum(s.get("kind") == "download" for s in running)
        candidates = sorted(
            [x for x in rows if x["status"] in ["PENDING", "RETRYING", "BLOCKED"]],
            key=lambda x: -json.loads(x["spec"]).get("priority", 0),
        )
        for row in candidates:
            s = json.loads(row["spec"])
            tid = row["task_id"]
            if s.get("blocked_reason") or row["retry_at"] > now:
                continue
            deps = s.get("depends_on", [])
            missing = [d for d in deps if states.get(d) != "COMPLETED"]
            if missing:
                if any(states.get(d) in ["FAILED", "BLOCKED"] for d in missing):
                    c.execute(
                        "UPDATE tasks SET status='BLOCKED',failure=? WHERE task_id=?",
                        ("Waiting for prerequisite: " + ",".join(missing), tid),
                    )
                continue
            absent = [p for p in s.get("inputs", []) if not Path(p).exists()]
            if absent:
                c.execute(
                    "UPDATE tasks SET status='BLOCKED',failure=? WHERE task_id=?",
                    ("Missing verified input: " + ",".join(absent), tid),
                )
                continue
            if not s.get("command"):
                c.execute(
                    "UPDATE tasks SET status='BLOCKED',failure='No executable implementation; never counted as completed' WHERE task_id=?",
                    (tid,),
                )
                continue
            need = s.get("resources", {})
            isgpu = need.get("gpu", False)
            if isgpu and (
                gpu_busy
                or r.get("gpu_free_mib", 0) < need.get("min_gpu_free_mib", 4500)
            ):
                continue
            if cpu_running + need.get("cpu_slots", 1) > 4 or (
                s.get("kind") == "download" and downloads >= 2
            ):
                continue
            if (
                r["memory_available_bytes"]
                < (need.get("ram_mib", 1024) + 2048) * 1024**2
            ):
                continue
            if r["disk_free_bytes"] < (need.get("disk_gib", 1) + 30) * 1024**3:
                continue
            attempt = row["attempt"] + 1
            unit = (
                "fyp-exp-"
                + hashlib.sha256(tid.encode()).hexdigest()[:12]
                + "-a"
                + str(attempt)
            )
            c.execute(
                "UPDATE tasks SET status='RUNNING',attempt=?,unit=?,start=?,end=NULL,failure=NULL WHERE task_id=?",
                (attempt, unit, now, tid),
            )
            event(c, tid, "DISPATCHED", {"unit": unit, "resources": need})
            c.commit()
            cmd = [
                "systemd-run",
                "--user",
                "--unit=" + unit,
                "--property=Type=exec",
                "--property=KillMode=control-group",
                "--property=MemoryAccounting=yes",
                "--property=CPUAccounting=yes",
                "--property=WorkingDirectory=" + str(REPO),
                PYTHON,
                str(ROOT / "runner/worker.py"),
                tid,
                "--root",
                str(root),
            ]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode:
                c.execute(
                    "UPDATE tasks SET status='RETRYING',failure=?,retry_at=? WHERE task_id=?",
                    (p.stderr, now + 30, tid),
                )
                event(c, tid, "DISPATCH_FAILED", p.stderr)
            else:
                states[tid] = "RUNNING"
                gpu_busy |= isgpu
                cpu_running += need.get("cpu_slots", 1)
                downloads += int(s.get("kind") == "download")
                r["memory_available_bytes"] -= need.get("ram_mib", 1024) * 1024**2
        c.commit()
    atomic(
        Path(root) / "state/controller_heartbeat.json",
        {"pid": os.getpid(), "time": time.time(), "resources": r},
    )


def main(a):
    root = Path(a.root)
    init(root)
    lock = open(root / "state/controller.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    specs = json.loads((ROOT / "manifests/tasks.json").read_text())
    register(specs, root)
    imported(root)
    while True:
        try:
            tick(root)
        except Exception as exc:
            with db(root) as c:
                event(c, None, "CONTROLLER_ERROR", repr(exc))
        if a.once:
            break
        time.sleep(5)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT))
    p.add_argument("--once", action="store_true")
    main(p.parse_args())
