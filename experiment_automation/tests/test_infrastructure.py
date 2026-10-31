"""Infrastructure-only tests, deliberately isolated from scientific experiment DB."""

import json, os, signal, sys, tempfile, time, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
from core import ROOT, init, register, db, atomic, due, utc
from scheduler import tick, PYTHON
from reporting import report


def main():
    root = Path("/data/fyp_3d/automation/selftests") / str(time.time_ns())
    init(root)
    specs = []
    for suffix, flag in [("resume", False), ("retry", True)]:
        out = root / "outputs" / suffix
        tid = "infra." + suffix + "." + root.name
        s = dict(
            task_id=tid,
            experiment_id="INFRASTRUCTURE_TEST_ONLY",
            dataset_id="no_scientific_data",
            command=[
                PYTHON,
                str(ROOT / "runner/jobs.py"),
                "selftest",
                "--path",
                str(out),
            ]
            + (["--fail-once"] if flag else []),
            validation=[
                dict(
                    path=str(out / "complete.json"),
                    kind="json",
                    equals={
                        "status": "passed",
                        "steps": 12,
                        "infrastructure_test_only": True,
                    },
                )
            ],
            max_attempts=3,
            retry_delay_s=1,
            resources={"gpu": False, "cpu_slots": 1, "ram_mib": 20, "disk_gib": 0},
            formal=False,
            priority=1,
        )
        specs.append(s)
    register(specs, root)
    killed = False
    deadline = time.time() + 120
    while time.time() < deadline:
        tick(root)
        with db(root) as c:
            rows = [dict(r) for r in c.execute("SELECT * FROM tasks")]
        resume = next(r for r in rows if ".resume." in r["task_id"])
        cp = root / "outputs/resume/checkpoint.json"
        if not killed and cp.exists() and 2 <= json.loads(cp.read_text())["step"] < 12:
            subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "kill",
                    "--kill-whom=all",
                    "--signal=KILL",
                    resume["unit"],
                ],
                check=True,
            )
            killed = True
        if all(r["status"] == "COMPLETED" for r in rows):
            break
        time.sleep(1)
    assert killed and all(r["status"] == "COMPLETED" for r in rows), [
        (r["task_id"], r["status"], r["failure"]) for r in rows
    ]
    assert all(r["attempt"] >= 2 for r in rows)
    assert not due(1000, 1000, 15399) and due(1000, 1000, 15400)
    assert not due(1000, 15400, 29799) and due(1000, 15400, 29800)
    path = report(root, True)
    assert path.exists() and (root / "reports/LATEST.json").exists()
    result = {
        "status": "passed",
        "actual_utc": utc(),
        "test_root": str(root),
        "scientific_data_used": False,
        "checkpoint_worker_was_killed": True,
        "resumed_steps": json.loads(cp.read_text())["step"],
        "tasks": [
            {"task_id": r["task_id"], "status": r["status"], "attempt": r["attempt"]}
            for r in rows
        ],
        "four_hour_boundary_tests": "14399s not due, 14400s due, next interval anchored correctly",
        "actual_report_file": str(path),
    }
    atomic(ROOT / "artifacts/infrastructure_selftest.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
