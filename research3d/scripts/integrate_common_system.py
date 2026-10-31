"""Actual continuous state propagation and absolute correction on ONE common map.
No dataset mixing. GT enters only after all ACE-only/fused/gated estimates finish.
No network confidence is called pose covariance.
"""

import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from motion.recovery import RecoveryGate
from motion.metrics import evaluate


def main(a):
    data = Path(a.dataset)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data / "acceptance.json").read_text())
    assert meta["simulated"] and not meta["GT_online"]
    mapfile = data / "readonly_prior_map.npz"
    digest = hashlib.sha256(mapfile.read_bytes()).hexdigest()
    assert digest == meta["readonly_map_sha256"]
    ace = np.load(Path(a.ace_run) / "predictions.npz")
    absolute = ace["T_world_camera"]
    m = np.load(Path(a.macvo_run) / "trajectory.npz")
    local = m["T_world_camera_raw"]
    lost = m["lost_before_interpolation"]
    quality = np.array(
        [
            int(r["inliers"])
            for r in csv.DictReader((Path(a.ace_run) / "per_frame.csv").open())
        ]
    )
    n = meta["continuous_stereo_frames"]
    assert len(local) == len(absolute) == len(quality) == n
    gate = RecoveryGate()
    naive = None
    fused = []
    gated = []
    states = []
    for i in range(n):
        delta = np.eye(4) if i == 0 else np.linalg.inv(local[i - 1]) @ local[i]
        valid = i > 0 and not bool(lost[i] or lost[i - 1])
        candidate = (
            absolute[i] if np.isfinite(absolute[i]).all() and quality[i] >= 50 else None
        )
        if naive is not None and valid:
            naive = naive @ delta
        if candidate is not None:
            naive = candidate.copy()
        fused.append(np.full((4, 4), np.nan) if naive is None else naive.copy())
        step = gate.step(delta, valid, candidate, int(quality[i]))
        gated.append(
            np.full((4, 4), np.nan)
            if step["T_world_camera"] is None
            else step.pop("T_world_camera")
        )
        step.pop("T_world_camera", None)
        states.append(dict(frame=i, motion_valid=valid, **step))
    predictions = {
        "ACE_only": absolute,
        "MACVO_ACE": np.stack(fused),
        "MACVO_ACE_gate": np.stack(gated),
    }
    np.savez_compressed(out / "global_predictions.npz", **predictions)
    (out / "online_diagnostics.json").write_text(json.dumps(states, indent=2))
    assert hashlib.sha256(mapfile.read_bytes()).hexdigest() == digest
    truth = np.load(data / "ground_truth_eval_only.npz")["T_world_camera"]
    result = {}
    for name, T in predictions.items():
        valid = np.isfinite(T).all((1, 2))
        te = np.full(n, np.inf)
        re = np.full(n, np.inf)
        te[valid] = np.linalg.norm(T[valid, :3, 3] - truth[valid, :3, 3], axis=1)
        re[valid] = np.degrees(
            Rotation.from_matrix(
                T[valid, :3, :3].transpose(0, 2, 1) @ truth[valid, :3, :3]
            ).magnitude()
        )
        result[name] = dict(
            n=n,
            absolute_map_translation_rmse_m=(
                float(np.sqrt(np.mean(te[valid] ** 2))) if valid.any() else None
            ),
            recall_5cm_5deg=float(np.mean((te <= 0.05) & (re <= 5))),
            recall_25cm_10deg=float(np.mean((te <= 0.25) & (re <= 10))),
            recall_1m_30deg=float(np.mean((te <= 1) & (re <= 30))),
            valid_pose_fraction=float(valid.mean()),
            median_translation_m=float(np.median(te[valid])) if valid.any() else None,
            median_rotation_deg=float(np.median(re[valid])) if valid.any() else None,
        )
    result["MACVO_only_local_odometry"] = evaluate(local, truth, lost)
    result["audit"] = dict(
        simulated=True,
        same_dataset_and_map=True,
        readonly_map_sha256=digest,
        map_unchanged=True,
        GT_online=False,
        frames=n,
        continuous_state_propagation=True,
        absolute_correction=True,
        gate_enabled_comparison=True,
        pose_covariance_available=False,
        warning="MACVO-only is local odometry: aligned/anchored errors are not map-coordinate absolute errors. Fusion uses map-coordinate ACE hypotheses without GT initialization.",
    )
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("ace_run")
    p.add_argument("macvo_run")
    p.add_argument("output")
    main(p.parse_args())
