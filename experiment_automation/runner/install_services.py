"""Install own user services; never invokes sudo or changes system-wide services."""

import datetime, os, subprocess
from pathlib import Path
from core import ROOT, REPO, init, db, atomic

PYTHON = "/data/fyp_3d/envs/macvo/bin/python"


def main():
    init()
    units = Path(os.environ["HOME"]) / ".config/systemd/user"
    units.mkdir(parents=True, exist_ok=True)

    def service(name, script, args="", restart="on-failure"):
        text = f"""[Unit]
Description=FYP {name}
After=default.target
[Service]
Type=exec
WorkingDirectory={REPO}
ExecStart={PYTHON} {ROOT}/runner/{script} {args}
Restart={restart}
RestartSec=10
KillMode=process
Environment=PYTHONUNBUFFERED=1
[Install]
WantedBy=default.target
"""
        (units / (name + ".service")).write_text(text)

    service("fyp-experiments", "scheduler.py")
    service("fyp-experiment-dashboard", "dashboard.py")
    (units / "fyp-experiment-report.service").write_text(
        f"""[Unit]
Description=FYP actual four-hour experiment snapshot
[Service]
Type=oneshot
WorkingDirectory={REPO}
ExecStart={PYTHON} {ROOT}/runner/reporting.py --force
"""
    )
    with db() as c:
        start = float(
            c.execute("SELECT value FROM meta WHERE key='service_started'").fetchone()[
                0
            ]
        )
    t = datetime.datetime.fromtimestamp(start, datetime.timezone.utc)
    hours = ",".join(str((t.hour + i * 4) % 24).zfill(2) for i in range(6))
    calendar = f"*-*-* {hours}:{t.minute:02d}:{t.second:02d} UTC"
    (units / "fyp-experiment-report.timer").write_text(
        f"""[Unit]
Description=FYP exact four-hour report timer, anchored to service creation UTC
[Timer]
OnCalendar={calendar}
Persistent=true
AccuracySec=1s
RandomizedDelaySec=0
Unit=fyp-experiment-report.service
[Install]
WantedBy=timers.target
"""
    )
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(
        [
            "systemctl",
            "--user",
            "enable",
            "--now",
            "fyp-experiments.service",
            "fyp-experiment-dashboard.service",
            "fyp-experiment-report.timer",
        ],
        check=True,
    )
    subprocess.run(
        ["systemctl", "--user", "start", "fyp-experiment-report.service"], check=True
    )
    atomic(
        ROOT / "artifacts/service_installation.json",
        {
            "calendar": calendar,
            "user_linger": subprocess.check_output(
                ["loginctl", "show-user", os.environ["USER"], "-p", "Linger"], text=True
            ).strip(),
            "services": [
                "fyp-experiments.service",
                "fyp-experiment-dashboard.service",
                "fyp-experiment-report.timer",
            ],
            "report_created_at_utc": t.isoformat(),
        },
    )
    print(calendar)


if __name__ == "__main__":
    main()
