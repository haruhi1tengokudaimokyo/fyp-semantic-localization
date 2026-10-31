"""Execute actual MINS CLI modes sequentially with runtime/return-code audits."""

import argparse, json, os, subprocess, time
from pathlib import Path


def main(a):
    root = Path(a.root)
    audit = json.loads((root / "audit.json").read_text())
    assert not any(t in audit["online_topics"] for t in ["/gt", "/scan"])
    results = {}
    for mode in a.modes:
        d = root / mode
        p = d / "config_init.yaml"
        text = p.read_text()
        text = text.replace("imu_only_init: true", "imu_only_init: false")
        p.write_text(text)
        # Wheel-enabled modes use upstream IMU-wheel dynamic initialization. With wheel
        # disabled the same flag falls back to upstream IMU-only, never GT.
        for n in ["trajectory.txt", "timing.txt"]:
            q = d / n
            if q.exists():
                raise RuntimeError("Refusing to overwrite completed run: " + str(q))
        start = time.perf_counter()
        with (d / "run.log").open("w") as log:
            proc = subprocess.run(
                [a.binary, str(d / "config.yaml")],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"},
            )
        results[mode] = dict(
            returncode=proc.returncode,
            wall_seconds=time.perf_counter() - start,
            trajectory_created=(d / "trajectory.txt").exists(),
            use_gt=False,
            wheel_initialization=mode != "stereo_imu",
            threads=4,
        )
        (d / "execution.json").write_text(json.dumps(results[mode], indent=2))
        print(mode, results[mode], flush=True)
        if proc.returncode:
            raise RuntimeError("MINS CLI failed; see run.log")
    (root / "execution.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["imu_wheel", "stereo_imu", "stereo_imu_wheel"],
        default=["imu_wheel", "stereo_imu", "stereo_imu_wheel"],
    )
    p.add_argument("--binary", required=True)
    main(p.parse_args())
