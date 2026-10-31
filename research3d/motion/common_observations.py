"""Continuous propagation with switchable, temporally checked map observations.
Inputs contain no query GT. Reprojection-based correction weights are explicit
heuristics, not calibrated pose covariance; correlated RGB modules never receive
an independent-sensor Bayesian interpretation. Fixed constants precede H results.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from motion.recovery import RecoveryGate, discrepancy


def blend(predicted, absolute, alpha):
    if predicted is None or alpha == 1.0:
        return absolute.copy()
    out = predicted.copy()
    out[:3, 3] = (1 - alpha) * predicted[:3, 3] + alpha * absolute[:3, 3]
    rot = Rotation.from_matrix(predicted[:3, :3].T @ absolute[:3, :3]).as_rotvec()
    out[:3, :3] = predicted[:3, :3] @ Rotation.from_rotvec(alpha * rot).as_matrix()
    return out


def estimate(
    relative,
    motion_valid,
    observations,
    modules,
    multiple_candidates=False,
    weighted=False,
):
    n = len(relative)
    assert len(motion_valid) == n and all(len(observations[k]) == n for k in modules)
    gates = {
        k: RecoveryGate(min_absolute_inliers=50 if k == "ace" else 12) for k in modules
    }
    T = None
    poses = []
    diagnostics = []
    for i in range(n):
        valid = bool(motion_valid[i])
        delta = relative[i]
        pred = None if T is None else (T @ delta if valid else T.copy())
        accepted = []
        rejected = 0
        for priority, module in enumerate(modules):
            pool = observations[module][i]
            pool = [
                h
                for h in pool
                if h is not None
                and np.isfinite(np.array(h["T_world_camera"])).all()
                and h["inliers"] >= gates[module].min_absolute_inliers
            ]
            if not multiple_candidates and pool:
                pool = pool[:1]
            if pool:
                if pred is None:
                    h = max(pool, key=lambda v: v.get("selection_score", v["inliers"]))
                else:

                    def distance(h):
                        dt, dr = discrepancy(pred, np.array(h["T_world_camera"]))
                        return dt / 0.25 + dr / 15

                    h = min(pool, key=distance)
                candidate = np.array(h["T_world_camera"], float)
                inliers = h["inliers"]
            else:
                h = None
                candidate = None
                inliers = 0
            step = gates[module].step(delta, valid, candidate, inliers)
            if h is not None and step["accepted_absolute"]:
                distance = (
                    0
                    if pred is None
                    else sum(
                        v / g for v, g in zip(discrepancy(pred, candidate), (0.25, 15))
                    )
                )
                accepted.append((distance, priority, module, h, step["recovered"]))
            elif h is not None:
                rejected += 1
        if accepted:
            _, _, module, h, recovered = min(accepted, key=lambda v: (v[0], v[1]))
            absolute = np.array(h["T_world_camera"], float)
            residual = h.get("median_reprojection_px")
            alpha = 1.0
            if weighted and residual is not None and np.isfinite(residual):
                alpha = float(np.clip(1 / (1 + max(0, float(residual))), 0.25, 1.0))
            # A three-frame consistent recovery explicitly resets the global gauge.
            T = blend(None if recovered else pred, absolute, alpha)
            row = dict(
                selected_module=module,
                accepted_absolute=True,
                recovered=bool(recovered),
                correction_alpha=alpha,
                reprojection_px=residual,
                candidate_count=sum(len(observations[k][i]) for k in modules),
                rejected_module_observations=rejected,
                status="TRACKING",
            )
        else:
            T = pred
            row = dict(
                selected_module=None,
                accepted_absolute=False,
                recovered=False,
                correction_alpha=None,
                candidate_count=sum(len(observations[k][i]) for k in modules),
                rejected_module_observations=rejected,
                status="DEAD_RECKONING" if pred is not None and valid else "LOST",
            )
        row.update(frame=i, motion_valid=valid, pose_covariance=None)
        diagnostics.append(row)
        poses.append(np.full((4, 4), np.nan) if T is None else T.copy())
    return np.stack(poses), diagnostics
