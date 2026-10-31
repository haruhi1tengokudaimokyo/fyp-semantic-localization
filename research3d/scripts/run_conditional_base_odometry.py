"""Explicit vendor-base-odometry conditional experiments, NEVER raw Wheel results.
2D/3D constraints are preregistered variants with identical noise/inputs/GT protocol.
"""

import argparse, json, subprocess, sys
from pathlib import Path


def main(a):
    src = Path(a.source)
    dst = Path(a.output)
    dst.mkdir(parents=True, exist_ok=True)
    provenance = json.loads(Path(a.provenance).read_text())
    assert provenance["status"] == "completed" and not provenance["pure_wheel_eligible"]
    assert not provenance["GT_or_LiDAR_fusion_reported_by_authors"]
    audit = json.loads((src / "audit.json").read_text())
    audit.update(
        odometry_source=provenance["conditional_input_name"],
        pure_wheel=False,
        conditional_constraint=(
            "zero lateral wheel-frame velocity"
            if a.dimension == 2
            else "zero lateral AND vertical wheel-frame velocities on assumed level ground"
        ),
        provenance_file=a.provenance,
    )
    (dst / "audit.json").write_text(json.dumps(audit, indent=2))
    for name in ["ground_truth_eval_only.npz", "imu_calibrated.npz"]:
        path = dst / name
        if not path.exists():
            path.symlink_to((src / name).resolve())
    modes = ["imu_wheel", "stereo_imu_wheel"]
    scripts = Path(__file__).parent
    for mode in modes:
        d = dst / mode
        d.mkdir(exist_ok=True)
        for p in (src / mode).glob("config*.yaml"):
            text = p.read_text().replace(str(src / mode), str(d))
            if a.dimension == 3:
                text = text.replace("type: Wheel2DCen", "type: Wheel3DCen")
            target = d / p.name
            if not target.exists():
                target.write_text(text)
        execution = d / "execution.json"
        if (
            execution.exists()
            and json.loads(execution.read_text())["returncode"] == 0
            and (d / "run.log").exists()
        ):
            continue
        subprocess.run(
            [
                sys.executable,
                str(scripts / "run_mins_suite.py"),
                str(dst),
                "--binary",
                a.binary,
                "--modes",
                mode,
            ],
            check=True,
        )
    subprocess.run(
        [sys.executable, str(scripts / "evaluate_mins.py"), str(dst), "--modes"]
        + modes,
        check=True,
    )
    result = dict(
        status="completed",
        constraint_dimension=a.dimension,
        pure_wheel=False,
        GT_online=False,
        online_sensor_combination=provenance["conditional_input_name"],
        conditional_planar_assumption=True,
        comparison=json.loads((dst / "comparison.json").read_text()),
        prior_map_available=False,
        claim="Vendor base-odometry twist + T265 IMU, with/without stereo, conditional kinematic constraints. Not independent raw wheel measurements, not AI improvement.",
    )
    (dst / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("output")
    p.add_argument("--provenance", required=True)
    p.add_argument("--binary", required=True)
    p.add_argument("--dimension", type=int, choices=[2, 3], required=True)
    main(p.parse_args())
