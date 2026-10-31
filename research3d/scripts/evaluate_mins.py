"""Offline evaluation of actual MINS ROS-free trajectories, preserving init failures.
MINS logs JPL q_GtoI; the same xyzw interpreted as Hamilton is R_ItoG.
No GT is supplied to MINS. Times are actual recorded sensor timestamps.
"""

import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from motion.metrics import evaluate


def interpolate(t, T, q):
    out = np.repeat(np.eye(4)[None], len(q), 0)
    tt = t - t[0]
    qq = q - t[0]
    out[:, :3, 3] = np.stack([np.interp(qq, tt, T[:, j, 3]) for j in range(3)], 1)
    out[:, :3, :3] = Slerp(tt, Rotation.from_matrix(T[:, :3, :3]))(qq).as_matrix()
    return out


def main(a):
    root = Path(a.root)
    audit = json.loads((root / "audit.json").read_text())
    z = np.load(root / "ground_truth_eval_only.npz")
    tg = z["timestamp_ns"] / 1e9
    gt = z["T_world_base"] @ np.array(audit["T_base_imu"])
    imu = np.load(root / "imu_calibrated.npz")["timestamp_ns"] / 1e9
    results = {}
    for mode in ["imu_wheel", "stereo_imu", "stereo_imu_wheel"]:
        d = root / mode
        p = d / "trajectory.txt"
        log = (d / "run.log").read_text(errors="replace")
        protocol = dict(
            online_topics=audit["online_topics"],
            GT_online=False,
            no_LiDAR_GPS=True,
            max_slam=0,
            raw_duration_s=float(imu[-1] - imu[0]),
            ground_truth_frame="gt_map -> base_link -> t265_imu",
            pose_convention="MINS JPL q_GtoI; Hamilton xyzw gives R_ItoG; p_IinG",
            timing="real header timestamps, seconds",
            sequence=root.name,
        )
        if not p.exists() or p.stat().st_size < 200:
            result = dict(
                status="initialization_failed",
                initialized=False,
                tracking_availability=0.0,
                failure_duration_s=float(imu[-1] - imu[0]),
                reason="No estimated trajectory; IMU-only initialization requires excitation",
                protocol=protocol,
            )
        else:
            data = np.loadtxt(p)
            data = np.atleast_2d(data)
            t = data[:, 0]
            # Logs may repeat current state for multiple incoming IMU samples.
            _, unique = np.unique(t, return_index=True)
            data = data[unique]
            t = t[unique]
            T = np.repeat(np.eye(4)[None], len(t), 0)
            T[:, :3, 3] = data[:, 1:4]
            T[:, :3, :3] = Rotation.from_quat(data[:, 4:8]).as_matrix()
            valid = (t >= tg[0]) & (t <= tg[-1])
            t = t[valid]
            T = T[valid]
            truth = interpolate(tg, gt, t)
            anchor = truth[0] @ np.linalg.inv(T[0])
            errR = np.linalg.inv(truth[:, :3, :3]) @ (anchor[:3, :3] @ T[:, :3, :3])
            err_deg = np.degrees(Rotation.from_matrix(errR).magnitude())
            metrics = evaluate(T, truth)
            metrics.pop("rpe_frame_delta")
            metrics.pop("failure_duration_seconds")
            metrics.pop("timestamp_note")
            metrics.pop("max_failure_run_frames")
            metrics.pop("lost_frames")
            metrics.pop("tracking_availability")
            later = t + 1.0
            mask = later <= t[-1]
            i = np.flatnonzero(mask)
            de = np.linalg.inv(T[i]) @ interpolate(t, T, later[mask])
            dg = np.linalg.inv(truth[i]) @ interpolate(tg, gt, later[mask])
            er = np.linalg.inv(dg) @ de
            gaps = np.diff(t)
            startup = float(t[0] - imu[0])
            duration = float(imu[-1] - imu[0])
            coverage = float(
                (t[-1] - t[0] - np.maximum(0, gaps - 0.05).sum()) / duration
            )
            metrics.update(
                rotation_first_pose_anchored_rmse_deg=float(
                    np.sqrt(np.mean(err_deg**2))
                ),
                rpe_1s_translation_rmse_m=float(
                    np.sqrt(np.mean(np.sum(er[:, :3, 3] ** 2, 1)))
                ),
                rpe_1s_rotation_rmse_deg=float(
                    np.sqrt(
                        np.mean(
                            np.degrees(Rotation.from_matrix(er[:, :3, :3]).magnitude())
                            ** 2
                        )
                    )
                ),
                estimated_first_last_s=t[[0, -1]].tolist(),
                initialization_delay_s=startup,
                max_output_gap_s=float(gaps.max()),
                trajectory_time_coverage=coverage,
                coverage_definition="(last-first - sum(max(0, gap-0.05s)))/raw IMU duration; includes initialization loss; not visual feature availability",
                raw_duration_s=duration,
            )
            np.savez_compressed(
                d / "evaluated_trajectory.npz",
                timestamp_s=t,
                T_world_imu=T,
                T_world_imu_gt=truth,
                T_world_imu_first_anchored=anchor @ T,
            )
            result = dict(
                status="completed", initialized=True, metrics=metrics, protocol=protocol
            )
        (d / "summary.json").write_text(json.dumps(result, indent=2))
        results[mode] = result
    (root / "comparison.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    main(p.parse_args())
