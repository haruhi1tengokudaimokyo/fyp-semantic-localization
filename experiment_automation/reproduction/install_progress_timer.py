"""Install/test only the new report timer; never restart experiment workers."""

import datetime, json, os, subprocess, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "experiment_automation"
units = Path(os.environ["HOME"]) / ".config/systemd/user"
units.mkdir(parents=True, exist_ok=True)
name = "fyp-reproduction-progress"
(units / (name + ".service")).write_text(
    f"""[Unit]
Description=FYP measured four-paper ETA and time checkpoints
[Service]
Type=oneshot
WorkingDirectory={REPO}
ExecStart=/data/fyp_3d/envs/macvo/bin/python {ROOT}/reproduction/progress.py
TimeoutStartSec=120
"""
)
(units / (name + ".timer")).write_text(
    f"""[Unit]
Description=FYP independent measured reproduction progress timer
[Timer]
OnActiveSec=2s
OnCalendar=*-*-* *:0/5:00
Persistent=true
AccuracySec=1s
RandomizedDelaySec=0
Unit={name}.service
[Install]
WantedBy=timers.target
"""
)
start = time.time()
subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
subprocess.run(["systemctl", "--user", "enable", "--now", name + ".timer"], check=True)
# Observe the timer's own first trigger, rather than manually starting the service.
deadline = time.monotonic() + 35
while time.monotonic() < deadline:
    p = ROOT / "state/reproduction_progress.json"
    if p.exists():
        data = json.loads(p.read_text())
        actual = datetime.datetime.fromisoformat(
            data["last_actual_run_cst"]
        ).timestamp()
        if actual >= start - 1:
            break
    time.sleep(1)
else:
    raise RuntimeError("Timer did not autonomously generate a fresh report")
properties = subprocess.check_output(
    [
        "systemctl",
        "--user",
        "show",
        name + ".service",
        "-p",
        "Result",
        "-p",
        "ExecMainStatus",
        "-p",
        "InvocationID",
        "-p",
        "ExecMainStartTimestamp",
        "-p",
        "ExecMainExitTimestamp",
    ],
    text=True,
)
assert "Result=success" in properties and "ExecMainStatus=0" in properties, properties
receipt = dict(
    status="passed",
    observed_automatic_timer_trigger=True,
    properties=properties,
    state=data,
    scientific_completion_claimed=False,
    four_paper_time_baseline=data["baseline_status"],
    original_controller_restarted=False,
    experiment_workers_restarted=False,
    test_start_epoch=start,
    observed_epoch=time.time(),
)
t = ROOT / "artifacts/reproduction_timer_test.json"
tmp = t.with_suffix(".tmp")
tmp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
tmp.replace(t)
print(json.dumps(receipt, ensure_ascii=False, indent=2))
