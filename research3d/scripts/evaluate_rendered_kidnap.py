"""Offline-only recovery metrics on an actually re-rendered camera teleport."""

import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def main(a):
    data = Path(a.dataset)
    run = Path(a.run)
    meta = json.loads((data / "acceptance.json").read_text())
    assert meta["actual_camera_teleport"] and not meta["image_postprocessing_fault"]
    assert meta["measured_rendered_camera_jump_m"] >= 2
    z = np.load(run / "global_predictions.npz")
    truth = np.load(data / "ground_truth_eval_only.npz")["T_world_camera"]
    n = len(truth)
    origin = meta["kidnap_frame"]
    results = {}
    dt = meta["simulated_timestamp_interval_s"]
    for name in z.files:
        P = z[name]
        valid = np.isfinite(P).all((1, 2))
        te = np.full(n, np.inf)
        re = te.copy()
        te[valid] = np.linalg.norm(P[valid, :3, 3] - truth[valid, :3, 3], axis=1)
        re[valid] = np.degrees(
            Rotation.from_matrix(
                P[valid, :3, :3].transpose(0, 2, 1) @ truth[valid, :3, :3]
            ).magnitude()
        )
        metrics = {}
        for t, r in [(0.05, 5), (0.25, 10), (1.0, 30)]:
            good = (te <= t) & (re <= r)
            hits = [i for i in range(origin, n - 2) if good[i : i + 3].all()]
            metrics[f"{t}m_{r}deg"] = dict(
                recovered=bool(hits),
                first_three_correct_start_frame=hits[0] if hits else None,
                latency_frames=hits[0] - origin if hits else None,
                latency_simulated_seconds=(hits[0] - origin) * dt if hits else None,
                confirmation_frame=hits[0] + 2 if hits else None,
                confirmation_latency_frames=hits[0] + 2 - origin if hits else None,
                confirmation_latency_simulated_seconds=(
                    (hits[0] + 2 - origin) * dt if hits else None
                ),
                post_jump_recall=float(good[origin:].mean()),
            )
        results[name] = dict(
            recovery=metrics,
            valid_pose_fraction=float(valid.mean()),
            post_jump_unavailable_frames=int((~valid[origin:]).sum()),
            post_jump_wrong_over25cm_frames=int(
                ((te[origin:] > 0.25) & valid[origin:]).sum()
            ),
            maximum_observed_pose_jump_m=(
                float(np.nanmax(np.linalg.norm(np.diff(P[:, :3, 3], axis=0), axis=1)))
                if (valid[:-1] & valid[1:]).any()
                else None
            ),
        )
    summary = dict(
        status="completed",
        actual_simulated_camera_teleport=True,
        real_world_kidnapping=False,
        GT_online=False,
        frames=n,
        jump_frame=origin,
        jump_m=meta["measured_rendered_camera_jump_m"],
        simulation_dt_s=dt,
        latency_scope="Frame-indexed replay and simulation-clock latency; not combined-model processing wall-clock latency. Neural/motion run costs are recorded separately.",
        results=results,
        scope="Actual Habitat camera teleport produces new stereo images; distinct from earlier injected pose/cache corruption; recovery measured at all preregistered thresholds, failures retained",
    )
    target = Path(a.output)
    target.mkdir(parents=True, exist_ok=True)
    temporary = target / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2))
    temporary.replace(target / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("run")
    p.add_argument("output")
    main(p.parse_args())
