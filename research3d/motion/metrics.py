"""Frame-indexed SE(3) metrics. No scale alignment or synthetic time claims."""

import numpy as np
from scipy.spatial.transform import Rotation


def align_se3(points, reference):
    if points.shape != reference.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected paired Nx3")
    x = points - points.mean(0)
    y = reference - reference.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    fix = np.eye(3)
    fix[-1, -1] = np.linalg.det(vt.T @ u.T)
    R = vt.T @ fix @ u.T
    t = reference.mean(0) - R @ points.mean(0)
    return R, t


def evaluate(estimated, truth, lost_mask=None):
    est = np.asarray(estimated, float)
    gt = np.asarray(truth, float)
    if (
        est.shape != gt.shape
        or est.ndim != 3
        or est.shape[1:] != (4, 4)
        or len(est) < 3
    ):
        raise ValueError("Paired Nx4x4 poses required")
    if not np.isfinite(est).all() or not np.isfinite(gt).all():
        raise ValueError("Nonfinite trajectory; do not silently drop failures")
    R, t = align_se3(est[:, :3, 3], gt[:, :3, 3])
    aligned = est.copy()
    aligned[:, :3, :3] = R @ est[:, :3, :3]
    aligned[:, :3, 3] = est[:, :3, 3] @ R.T + t
    # First-pose gauge is independent of subsequent test trajectory alignment.
    anchor = gt[0] @ np.linalg.inv(est[0])
    anchored = anchor @ est
    te = np.linalg.norm(aligned[:, :3, 3] - gt[:, :3, 3], axis=1)
    ae = np.linalg.norm(anchored[:, :3, 3] - gt[:, :3, 3], axis=1)
    path = float(np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1).sum())
    rpe = {}
    for delta in [1, 5, 10]:
        if delta >= len(est):
            continue
        de = np.linalg.inv(est[:-delta]) @ est[delta:]
        dg = np.linalg.inv(gt[:-delta]) @ gt[delta:]
        error = np.linalg.inv(dg) @ de
        tx = np.linalg.norm(error[:, :3, 3], axis=1)
        rx = np.degrees(Rotation.from_matrix(error[:, :3, :3]).magnitude())
        rpe[str(delta)] = dict(
            translation_rmse_m=float(np.sqrt(np.mean(tx**2))),
            rotation_rmse_deg=float(np.sqrt(np.mean(rx**2))),
            pairs=len(tx),
        )
    lost = (
        np.zeros(len(est), bool) if lost_mask is None else np.asarray(lost_mask, bool)
    )
    if lost.shape != (len(est),):
        raise ValueError("Lost mask mismatch")
    runs = []
    run = 0
    for bad in lost[1:]:
        if bad:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return dict(
        n=len(est),
        alignment="SE3, no scale correction",
        ate_se3_aligned_rmse_m=float(np.sqrt(np.mean(te**2))),
        ate_first_pose_anchored_rmse_m=float(np.sqrt(np.mean(ae**2))),
        path_length_m=path,
        endpoint_first_pose_anchored_error_m=float(ae[-1]),
        endpoint_drift_per_meter=float(ae[-1] / path) if path > 0 else None,
        rpe_frame_delta=rpe,
        tracking_availability=float(1 - lost[1:].mean()),
        lost_frames=int(lost[1:].sum()),
        max_failure_run_frames=max(runs, default=0),
        failure_duration_seconds=None,
        timestamp_note="Frame deltas only; no physical seconds assigned when original timestamps are absent",
    )
