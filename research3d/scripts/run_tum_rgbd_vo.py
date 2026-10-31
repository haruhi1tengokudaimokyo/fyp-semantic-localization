"""Measured RGB-D GFTT/LK/PnP odometry on complete TUM Freiburg3 sequences.
No GT or map enters motion estimation. Missing depth/failed steps retain original
frame positions with an explicit lost mask; held poses are not claimed tracked.
"""

import argparse, json, time
from pathlib import Path
import cv2, numpy as np
from scipy.spatial.transform import Rotation, Slerp
from motion.metrics import evaluate


def entries(path):
    return [
        (float(v[0]), v[1])
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.startswith("#")
        for v in [line.split()]
    ]


def associate(rgb, depth):
    possibilities = []
    times = np.array([r[0] for r in depth])
    for i, (t, _) in enumerate(rgb):
        lo = np.searchsorted(times, t - 0.02)
        hi = np.searchsorted(times, t + 0.02)
        possibilities.extend((abs(t - times[j]), i, j) for j in range(lo, hi))
    mapping = {}
    used = set()
    for _, i, j in sorted(possibilities):
        if i not in mapping and j not in used:
            mapping[i] = j
            used.add(j)
    return mapping


def estimate(previous, current, depth, K):
    points = cv2.goodFeaturesToTrack(
        previous,
        600,
        0.01,
        8,
        mask=((depth > 0.1) & (depth < 8)).astype(np.uint8) * 255,
        blockSize=7,
    )
    if points is None:
        return np.eye(4), False, 0
    tracked, st, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, points, None, winSize=(21, 21), maxLevel=3
    )
    if tracked is None:
        return np.eye(4), False, 0
    back, sb, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, tracked, None, winSize=(21, 21), maxLevel=3
    )
    if back is None:
        return np.eye(4), False, 0
    keep = (
        (st[:, 0] > 0)
        & (sb[:, 0] > 0)
        & (np.linalg.norm(back[:, 0] - points[:, 0], axis=1) < 1)
    )
    u = points[:, 0][keep]
    v = tracked[:, 0][keep]
    ij = np.rint(u).astype(int)
    z = depth[ij[:, 1], ij[:, 0]]
    P = np.c_[(u[:, 0] - K[0, 2]) * z / K[0, 0], (u[:, 1] - K[1, 2]) * z / K[1, 1], z]
    if len(P) < 20:
        return np.eye(4), False, 0
    ok, rv, tv, ix = cv2.solvePnPRansac(
        P,
        v,
        K,
        None,
        iterationsCount=1000,
        reprojectionError=2,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or ix is None or len(ix) < 20:
        return np.eye(4), False, 0 if ix is None else len(ix)
    j = ix[:, 0]
    rv, tv = cv2.solvePnPRefineLM(P[j], v[j], K, None, rv, tv)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rv)[0]
    T[:3, 3] = tv[:, 0]
    return np.linalg.inv(T), bool(np.isfinite(T).all()), len(ix)


def main(a):
    root = Path(a.dataset)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((root / "acceptance.json").read_text())
    assert meta["all_png_verified"]
    rgb = entries(root / "rgb.txt")
    depth = entries(root / "depth.txt")
    mapping = associate(rgb, depth)
    K = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]], dtype=float)
    cv2.setNumThreads(4)
    cv2.setRNGSeed(1305)
    previous = None
    previous_depth = None
    T = np.eye(4)
    trajectory = []
    lost = []
    rows = []
    deltas = []
    for i, (timestamp, path) in enumerate(rgb):
        tick = time.perf_counter()
        gray = cv2.imread(str(root / path), 0)
        assert gray is not None
        delta = np.eye(4)
        ok = False
        nin = 0
        if previous is not None and previous_depth is not None:
            delta, ok, nin = estimate(previous, gray, previous_depth, K)
        if ok:
            T = T @ delta
        trajectory.append(T.copy())
        deltas.append(delta)
        lost.append(not ok if i else False)
        rows.append(
            dict(
                frame=i,
                timestamp=timestamp,
                valid=ok,
                inliers=nin,
                depth_associated=i in mapping,
                runtime_ms=(time.perf_counter() - tick) * 1000,
            )
        )
        previous = gray
        previous_depth = (
            cv2.imread(str(root / depth[mapping[i]][1]), cv2.IMREAD_UNCHANGED).astype(
                float
            )
            / 5000
            if i in mapping
            else None
        )
        if i % 100 == 0:
            print("FRAME", i, "/", len(rgb), "lost", sum(lost), flush=True)
    poses = np.stack(trajectory)
    stamps = np.array([r[0] for r in rgb])
    lost = np.array(lost)
    np.savez_compressed(
        out / "trajectory.npz",
        T_world_camera=poses,
        T_previous_current=np.stack(deltas),
        timestamp_s=stamps,
        lost=lost,
    )
    (out / "per_frame.json").write_text(json.dumps(rows, indent=2))
    # Evaluation-only boundary: first access to mocap after all online estimates are saved.
    gt = np.loadtxt(root / "groundtruth.txt")
    order = np.argsort(gt[:, 0])
    gt = gt[order]
    _, unique = np.unique(gt[:, 0], return_index=True)
    gt = gt[unique]
    ix = np.searchsorted(gt[:, 0], stamps)
    valid = (ix > 0) & (ix < len(gt))
    safe = np.clip(ix, 1, len(gt) - 1)
    valid &= (gt[safe, 0] - gt[safe - 1, 0]) < 0.1
    q = stamps[valid]
    truth = np.repeat(np.eye(4)[None], len(q), 0)
    truth[:, :3, 3] = np.stack([np.interp(q, gt[:, 0], gt[:, j]) for j in [1, 2, 3]], 1)
    truth[:, :3, :3] = Slerp(gt[:, 0] - gt[0, 0], Rotation.from_quat(gt[:, 4:8]))(
        q - gt[0, 0]
    ).as_matrix()
    metrics = evaluate(poses[valid], truth, lost[valid])
    metrics["tracking_availability_all_rgb_frames"] = float(1 - lost[1:].mean())
    metrics["failure_duration_seconds"] = float(np.diff(stamps)[lost[1:]].sum())
    metrics["timestamp_note"] = (
        "Actual RGB timestamps; all original RGB frames preserved; GT only interpolated offline inside bracketing intervals<0.1s"
    )
    metrics["raw_rgb_frames"] = len(rgb)
    metrics["gt_evaluable_frames"] = int(valid.sum())
    metrics["runtime_mean_ms"] = float(np.mean([r["runtime_ms"] for r in rows]))
    metrics["rgb_depth_associated_frames"] = len(mapping)
    result = dict(
        status="completed",
        sequence=meta["sequence"],
        metrics=metrics,
        GT_online=False,
        real_people=True,
        full_sequence=True,
        prior_map_available=False,
        task="RGB-D visual odometry robustness; not full prior-map localization",
        K=K.tolist(),
        depth_scale=5000,
        association="Official-style greedy minimum timestamp difference, unique pairs <=0.02s; unmatched RGB frames retained",
        calibration_source=meta["official_formats"],
        dynamics=meta["dynamics"],
    )
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("output")
    main(p.parse_args())
