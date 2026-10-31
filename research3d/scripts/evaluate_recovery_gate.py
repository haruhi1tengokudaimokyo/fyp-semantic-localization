"""Real ACE + measured RGB-D motion replay, controlled recovery fault injection.
All thresholds fixed before examining recovery results; physical times unavailable.
These controlled faults are NOT a real kidnapped-robot or pedestrian benchmark.
"""

import argparse, csv, json, time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from motion.recovery import RecoveryGate


def errors(est, gt):
    valid = np.isfinite(est).all((1, 2))
    te = np.full(len(est), np.inf)
    re = te.copy()
    te[valid] = np.linalg.norm(est[valid, :3, 3] - gt[valid, :3, 3], axis=1)
    re[valid] = np.degrees(
        Rotation.from_matrix(
            est[valid, :3, :3].transpose(0, 2, 1) @ gt[valid, :3, :3]
        ).magnitude()
    )
    return te, re


def main(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    c = np.load(Path(a.ace_run) / "predictions.npz", allow_pickle=False)
    candidate = c["T_world_camera"]
    frames = c["frame"]
    m = np.load(Path(a.motion) / "motion.npz", allow_pickle=False)
    assert np.array_equal(frames, m["frame"])
    quality = np.array(
        [
            int(r["inliers"])
            for r in csv.DictReader((Path(a.ace_run) / "per_frame.csv").open())
        ]
    )
    n = len(frames)
    shift = np.eye(4)
    shift[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    shift[0, 3] = 1.0
    statefault = np.eye(4)
    statefault[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    statefault[:3, 3] = [2, 0, 0]
    cases = {
        "clean": {},
        "camera_outage_30frames": {"drop": list(range(300, 330))},
        "injected_state_drift": {"state_fault": 400},
        "isolated_wrong_absolute": {"corrupt": [200, 500, 800]},
        "coherent_wrong_absolute_burst": {"corrupt": list(range(600, 605))},
    }
    saved = {}
    start = time.perf_counter()
    for name, case in cases.items():
        gate = RecoveryGate()
        estimated = []
        naive = []
        rows = []
        last_naive = None
        for i in range(n):
            dropout = i in case.get("drop", [])
            prev_dropout = i - 1 in case.get("drop", [])
            absolute = None if dropout else candidate[i].copy()
            mv = bool(m["valid"][i]) and not (dropout or prev_dropout)
            if absolute is not None and i in case.get("corrupt", []):
                absolute = shift @ absolute
            if i == case.get("state_fault"):
                if gate.T is not None:
                    gate.T = statefault @ gate.T
                if last_naive is not None:
                    last_naive = statefault @ last_naive
            if absolute is not None and quality[i] >= 50:
                last_naive = absolute.copy()
            elif last_naive is not None and mv:
                last_naive = last_naive @ m["T_previous_current"][i]
            step = gate.step(m["T_previous_current"][i], mv, absolute, int(quality[i]))
            estimated.append(
                np.full((4, 4), np.nan)
                if step["T_world_camera"] is None
                else step.pop("T_world_camera")
            )
            naive.append(
                np.full((4, 4), np.nan) if last_naive is None else last_naive.copy()
            )
            step.pop("T_world_camera", None)
            rows.append(
                dict(
                    frame=i,
                    motion_available=mv,
                    absolute_available=absolute is not None,
                    **step
                )
            )
        estimated = np.stack(estimated)
        naive = np.stack(naive)
        np.savez_compressed(
            out / (name + ".npz"),
            T_world_camera_gated=estimated,
            T_world_camera_single_frame=naive,
            frame=frames,
        )
        (out / (name + ".online_diagnostics.json")).write_text(
            json.dumps(rows, indent=2)
        )
        saved[name] = (estimated, naive, rows)
    # All online replay finishes before any query GT pose is accessed.
    root = Path(a.scene)
    truth = np.stack(
        [
            np.loadtxt(root / "test/poses" / str(f).replace(".color.png", ".pose.txt"))
            for f in frames
        ]
    )
    summary = {}
    for name, (est, naive, rows) in saved.items():
        methods = {}
        for method, T in [("temporal_gate", est), ("single_frame_overwrite", naive)]:
            te, re = errors(T, truth)
            good = (te <= 0.05) & (re <= 5)
            methods[method] = dict(
                recall_5cm_5deg=float(good.mean()),
                median_translation_m=float(np.median(te[np.isfinite(te)])),
                median_rotation_deg=float(np.median(re[np.isfinite(re)])),
                valid_pose_fraction=float(np.isfinite(te).mean()),
                wrong_over_25cm_frames=int(((te > 0.25) & np.isfinite(te)).sum()),
                translation_errors_m=[float(v) if np.isfinite(v) else None for v in te],
                rotation_errors_deg=[float(v) if np.isfinite(v) else None for v in re],
            )
            events = {}
            for label, origin in [
                ("startup", 0),
                (
                    "post_camera_outage",
                    330 if name == "camera_outage_30frames" else None,
                ),
                ("post_state_drift", 400 if name == "injected_state_drift" else None),
                (
                    "post_wrong_burst",
                    605 if name == "coherent_wrong_absolute_burst" else None,
                ),
            ]:
                if origin is None:
                    continue
                hits = [j for j in range(origin, n - 2) if good[j : j + 3].all()]
                events[label] = (
                    None
                    if not hits
                    else dict(
                        first_three_correct_start_frame=hits[0],
                        latency_frames=hits[0] - origin,
                    )
                )
            methods[method]["offline_recovery_events"] = events
        methods["gate_diagnostics"] = dict(
            accepted_absolute=sum(r["accepted_absolute"] for r in rows),
            recovery_transitions=sum(r["recovered"] for r in rows),
            lost_status_frames=sum(r["status"] == "LOST" for r in rows),
            recovering_status_frames=sum(r["status"] == "RECOVERING" for r in rows),
        )
        summary[name] = methods
    protocol = dict(
        absolute_source="actual locally trained ACE predictions; no GT field in cache",
        motion_source="actual registered Kinect RGB-D GFTT/LK/PnP, no synthetic motion or test poses",
        frames=n,
        gate=dict(
            min_inliers=50,
            translation_gate_m=0.1,
            rotation_gate_deg=10,
            required_consistent=3,
        ),
        faults=cases,
        GT_online=False,
        physical_timestamps=False,
        latency_units="frames only",
        control_scope="controlled packet loss / injected state and absolute-constraint faults, not genuine physical kidnapping or pedestrian dynamics",
        no_online_mapping=True,
        covariance="unavailable; no score converted to covariance",
        replay_wall_seconds=time.perf_counter() - start,
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        {
            k: {
                m: {p: q for p, q in v.items() if "errors_" not in p}
                for m, v in vals.items()
            }
            for k, vals in summary.items()
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scene")
    p.add_argument("ace_run")
    p.add_argument("motion")
    p.add_argument("output")
    main(p.parse_args())
