import argparse, json, subprocess
from core import ROOT, db, init
from reporting import snapshot, report

p = argparse.ArgumentParser()
p.add_argument("action", choices=["status", "report", "start", "restart", "logs"])
p.add_argument("task", nargs="?")
a = p.parse_args()
init()
if a.action == "status":
    s = snapshot()
    print(
        "Started",
        s["service_started_utc"],
        "Elapsed hours",
        round(s["elapsed_seconds"] / 3600, 2),
    )
    print(json.dumps(s["counts"], ensure_ascii=False))
    print("Dashboard http://127.0.0.1:8766")
    print("Next report", s["next_four_hour_due_utc"])
    for t in s["tasks"]:
        if t["status"] in ["RUNNING", "FAILED", "RETRYING"]:
            print(
                t["status"],
                t["task_id"],
                "PID",
                t["pid"],
                "attempt",
                t["attempt"],
                t["failure"] or t.get("log_path", ""),
            )
elif a.action == "report":
    report(force=True)
elif a.action in ["start", "restart"]:
    subprocess.run(
        ["systemctl", "--user", a.action, "fyp-experiments.service"], check=True
    )
elif a.action == "logs":
    if not a.task:
        raise SystemExit("Provide a task_id")
    paths = sorted((ROOT / "logs").glob(a.task + ".attempt*.log"))
    print(
        "\n".join(paths[-1].read_text(errors="replace").splitlines()[-30:])
        if paths
        else "No log yet"
    )
