"""Compare estimates on the exact same real-time grid, retaining startup failures."""

import argparse, json
from pathlib import Path
import numpy as np
from evaluate_mins import interpolate
from motion.metrics import evaluate


def main(a):
    root = Path(a.root)
    tr = {}
    for d in root.iterdir():
        if d.is_dir() and (d / "evaluated_trajectory.npz").exists():
            tr[d.name] = np.load(d / "evaluated_trajectory.npz")
    begin = max(z["timestamp_s"][0] for z in tr.values())
    end = min(z["timestamp_s"][-1] for z in tr.values())
    t = begin + np.arange(int((end - begin) * 20) + 1) / 20
    result = dict(
        common_first_last_s=t[[0, -1]].tolist(),
        sample_period_s=0.05,
        grid_count=len(t),
        modes={},
    )
    gz = np.load(root / "ground_truth_eval_only.npz")
    audit = json.loads((root / "audit.json").read_text())
    common_gt = interpolate(
        gz["timestamp_ns"] / 1e9, gz["T_world_base"] @ np.array(audit["T_base_imu"]), t
    )
    for name, z in tr.items():
        T = interpolate(z["timestamp_s"], z["T_world_imu"], t)
        gt = common_gt
        m = evaluate(T, gt)
        m.pop("timestamp_note")
        m.pop("failure_duration_seconds")
        m.pop("tracking_availability")
        m.pop("lost_frames")
        m.pop("max_failure_run_frames")
        de = np.linalg.inv(T[:-20]) @ T[20:]
        dg = np.linalg.inv(gt[:-20]) @ gt[20:]
        e = np.linalg.inv(dg) @ de
        from scipy.spatial.transform import Rotation

        m["rpe_1s"] = dict(
            translation_rmse_m=float(np.sqrt(np.mean(np.sum(e[:, :3, 3] ** 2, 1)))),
            rotation_rmse_deg=float(
                np.sqrt(
                    np.mean(
                        np.degrees(Rotation.from_matrix(e[:, :3, :3]).magnitude()) ** 2
                    )
                )
            ),
        )
        m.pop("rpe_frame_delta")
        result["modes"][name] = m
    (root / "common_window_comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    main(p.parse_args())
