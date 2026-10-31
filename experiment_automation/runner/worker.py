"""One experiment per independent systemd unit; survives controller exit."""

import argparse, json, os, resource, subprocess, time, traceback
from pathlib import Path
from core import (
    ROOT,
    REPO,
    db,
    event,
    fingerprint,
    validate,
    atomic,
    utc,
    resources,
    source_snapshot,
)


def run(tid, root=ROOT):
    with db(root) as c:
        row = c.execute("SELECT * FROM tasks WHERE task_id=?", (tid,)).fetchone()
        s = json.loads(row["spec"])
        attempt = row["attempt"]
        started = time.time()
        source = source_snapshot(root)
        h = fingerprint(s)
        atomic(
            Path(root) / "artifacts" / f"{tid}.attempt{attempt}.provenance.json",
            {"source": source, "task_spec": s, "input_hash": h, "started": started},
        )
        c.execute(
            "UPDATE tasks SET pid=?,start=?,heartbeat=?,input_hash=?,code_commit=? WHERE task_id=?",
            (
                os.getpid(),
                started,
                started,
                h,
                subprocess.check_output(
                    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
                ).strip(),
                tid,
            ),
        )
        event(
            c,
            tid,
            "WORKER_STARTED",
            {"pid": os.getpid(), "attempt": attempt, "input_hash": h},
        )
    log = Path(root) / "logs" / f"{tid}.attempt{attempt}.log"
    command = s["command"]
    failure = None
    samples = []
    try:
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": str(s.get("threads", 4)),
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONPATH": str(REPO / "research3d"),
        }
        network_file = Path(root) / "state/network_environment.json"
        if network_file.exists():
            env.update(json.loads(network_file.read_text()))
        with log.open("a") as f:
            f.write(f"\nSTART {utc()} task={tid} attempt={attempt}\n")
            f.flush()
            p = subprocess.Popen(
                command,
                cwd=s.get("cwd", str(REPO)),
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
            )
            atomic(
                Path(root) / "state" / f"{tid}.process.json",
                {
                    "worker_pid": os.getpid(),
                    "payload_pid": p.pid,
                    "started": started,
                    "command": command,
                    "log": str(log),
                },
            )
            while p.poll() is None:
                with db(root) as c:
                    c.execute(
                        "UPDATE tasks SET heartbeat=? WHERE task_id=?",
                        (time.time(), tid),
                    )
                if len(samples) % 6 == 0:
                    samples.append(resources())
                else:
                    samples.append({"time_utc": utc()})
                time.sleep(5)
            if p.returncode:
                raise RuntimeError(f"Payload exit code {p.returncode}; evidence: {log}")
        accepted = validate(s, started)
        receipt = {
            **accepted,
            "task_id": tid,
            "source": source,
            "input_hash": h,
            "attempt": attempt,
            "wall_seconds": time.time() - started,
            "child_maxrss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            "resource_samples": samples,
        }
        atomic(Path(root) / "artifacts" / f"{tid}.receipt.json", receipt)
        with db(root) as c:
            c.execute(
                "UPDATE tasks SET status='COMPLETED',end=?,validation=?,output_paths=?,failure=NULL WHERE task_id=?",
                (
                    time.time(),
                    json.dumps(accepted),
                    json.dumps([v["path"] for v in s["validation"]]),
                    tid,
                ),
            )
            event(c, tid, "COMPLETED", receipt)
    except BaseException as exc:
        failure = {
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "log": str(log),
            "attempt": attempt,
        }
        atomic(
            Path(root) / "artifacts" / f"{tid}.attempt{attempt}.failure.json", failure
        )
        with db(root) as c:
            status = "RETRYING" if attempt < s.get("max_attempts", 2) else "FAILED"
            c.execute(
                "UPDATE tasks SET status=?,end=?,failure=?,retry_at=? WHERE task_id=?",
                (
                    status,
                    time.time(),
                    json.dumps(failure),
                    time.time() + s.get("retry_delay_s", 30),
                    tid,
                ),
            )
            event(c, tid, status, failure)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("task_id")
    p.add_argument("--root", default=str(ROOT))
    a = p.parse_args()
    run(a.task_id, Path(a.root))
