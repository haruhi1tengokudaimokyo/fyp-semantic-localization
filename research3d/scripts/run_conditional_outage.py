"""Full real office sequences with a fixed 8--12s camera packet outage.
Vendor-fused base odometry is labelled explicitly; this is not a people benchmark.
"""

import argparse, json, subprocess, sys
from pathlib import Path


def main(a):
    root = Path(a.root)
    scripts = Path(__file__).parent
    out = root.with_name(root.name + "_outage")
    subprocess.run(
        [
            sys.executable,
            str(scripts / "derive_visual_outage.py"),
            str(root),
            "--start",
            "8",
            "--end",
            "12",
        ],
        check=True,
    )
    audit = json.loads((out / "audit.json").read_text())
    assert not audit["pure_wheel"]
    assert audit["controlled_visual_outage"]["removed_camera_messages"]
    modes = ["imu_wheel", "stereo_imu_wheel"]
    for mode in modes:
        execution = out / mode / "execution.json"
        if execution.exists() and json.loads(execution.read_text())["returncode"] == 0:
            continue
        subprocess.run(
            [
                sys.executable,
                str(scripts / "run_mins_suite.py"),
                str(out),
                "--binary",
                a.binary,
                "--modes",
                mode,
            ],
            check=True,
        )
    subprocess.run(
        [sys.executable, str(scripts / "evaluate_mins.py"), str(out), "--modes"]
        + modes,
        check=True,
    )
    result = dict(
        status="completed",
        GT_online=False,
        pure_wheel=False,
        real_people_benchmark=False,
        controlled_camera_packet_outage=True,
        conditional_planar_assumption=True,
        outage=audit["controlled_visual_outage"],
        comparison=json.loads((out / "comparison.json").read_text()),
        source_run=str(root),
        claim="Fixed camera packet removal only; original IMU and vendor-fused encoder+chassis-IMU base odometry retained. No pure-Wheel or pedestrian-dynamics claim.",
    )
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--binary", required=True)
    main(p.parse_args())
