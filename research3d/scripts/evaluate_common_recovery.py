"""Controlled fault replay using completed same-scene ACE and actual MACVO motion.
These faults are explicitly not physical kidnapping, real people or sensor recordings.
"""

import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np


def main(a):
    root = Path(a.dataset)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((root / "acceptance.json").read_text())
    assert meta["simulated"] and meta["continuous_stereo_frames"] == 600
    absolute = np.load(Path(a.ace_run) / "predictions.npz")
    raw = np.load(Path(a.macvo_run) / "trajectory.npz")
    T = raw["T_world_camera_raw"]
    lost = raw["lost_before_interpolation"]
    n = len(T)
    assert len(absolute["frame"]) == n
    delta = np.repeat(np.eye(4)[None], n, 0)
    delta[1:] = np.linalg.inv(T[:-1]) @ T[1:]
    valid = np.r_[False, ~(lost[:-1] | lost[1:])]
    np.savez_compressed(
        out / "motion.npz",
        T_previous_current=delta,
        valid=valid,
        frame=absolute["frame"],
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("evaluate_recovery_gate.py")),
            str(root / "ace_scene"),
            a.ace_run,
            str(out),
            str(out / "replay"),
            "--scaled-cases",
            "--motion-source",
            "actual same-scene MACVO raw relative transformations; simulated true stereo measurements; no GT-derived motion",
        ],
        check=True,
    )
    result = json.loads((out / "replay/summary.json").read_text())
    result["scope"] = dict(
        frames=n,
        GT_online=False,
        simulated=True,
        physical_kidnapping=False,
        real_people=False,
        controlled_faults_only=True,
        readonly_map_sha256=meta["readonly_map_sha256"],
    )
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["scope"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("ace_run")
    p.add_argument("macvo_run")
    p.add_argument("output")
    main(p.parse_args())
