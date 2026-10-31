"""Offline residual/uncertainty audit using GT motion only after estimation.
3D matching coverage is an empirical consistency diagnostic, NOT a pose covariance
calibration claim: learned depth/flow errors can be correlated.
"""

import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import chi2, spearmanr
from motion.adapters import load_macvo_constraints


def unproject(uv, d, K):
    # Upstream NED coordinates: depth, horizontal-right, vertical-down.
    return np.stack(
        [
            d,
            (uv[:, 0] - K[:, 0, 2]) * d / K[:, 0, 0],
            (uv[:, 1] - K[:, 1, 2]) * d / K[:, 1, 1],
        ],
        1,
    )


def main(a):
    root = Path(a.run)
    z = np.load(root / "observations.npz")
    traj = np.load(root / "trajectory.npz")
    C = np.eye(4)
    C[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    gt = C @ traj["T_world_camera_gt"] @ C.T
    pose = C @ traj["T_world_camera_raw"] @ C.T
    i = z["edge/match2frame1/mapping"]
    j = z["edge/match2frame2/mapping"]
    K = z["frames//K"]
    p = unproject(z["match//pixel1_uv"], z["match//pixel1_d"][:, 0], K[i])
    q = unproject(z["match//pixel2_uv"], z["match//pixel2_d"][:, 0], K[j])
    dgt = np.linalg.inv(gt[j]) @ gt[i]
    R = dgt[:, :3, :3]
    e = (R @ p[..., None])[..., 0] + dgt[:, :3, 3] - q
    cov = R @ z["match//obs1_covTc"] @ R.swapaxes(1, 2) + z["match//obs2_covTc"]
    eig = np.linalg.eigvalsh(cov)
    valid = np.isfinite(cov).all((1, 2)) & (eig[:, 0] > 0)
    m = np.full(len(e), np.nan)
    m[valid] = np.einsum("ni,ni->n", e[valid], np.linalg.solve(cov[valid], e[valid]))
    sigma = np.trace(cov, axis1=1, axis2=2)
    err = np.linalg.norm(e, axis=1)
    # Actual optimized pixel+disparity residuals against the stored fixed source points.
    point = z["points//pos_Tw"][z["edge/match2point/mapping"]]
    inv = np.linalg.inv(pose[j])
    pc = (inv[:, :3, :3] @ point[..., None])[..., 0] + inv[:, :3, 3]
    uv = np.stack(
        [
            K[j, 0, 0] * pc[:, 1] / pc[:, 0] + K[j, 0, 2],
            K[j, 1, 1] * pc[:, 2] / pc[:, 0] + K[j, 1, 2],
        ],
        1,
    )
    res_uv = uv - z["match//pixel2_uv"]
    res_disp = (
        K[j, 0, 0] * z["frames//baseline"][j] / pc[:, 0] - z["match//pixel2_disp"][:, 0]
    )
    np.savez_compressed(
        root / "residuals.npz",
        frame1=i,
        frame2=j,
        optimized_pixel_residual_px=res_uv,
        optimized_disparity_residual_px=res_disp,
        GT_motion_3d_match_residual_m=e,
        combined_observation_covariance_m2=cov,
        observation_mahalanobis_squared=m,
    )
    constraints = load_macvo_constraints(root)
    fuse_rejected = False
    try:
        constraints[0].for_timed_estimator()
    except ValueError:
        fuse_rejected = True
    report = dict(
        matches=len(e),
        optimized_pixel_residual_median_px=float(
            np.median(np.linalg.norm(res_uv, axis=1))
        ),
        optimized_pixel_residual_p95_px=float(
            np.percentile(np.linalg.norm(res_uv, axis=1), 95)
        ),
        optimized_disparity_abs_median_px=float(np.median(np.abs(res_disp))),
        GT_motion_match_error_median_m=float(np.median(err)),
        GT_motion_match_error_p95_m=float(np.percentile(err, 95)),
        positive_definite_fraction=float(valid.mean()),
        observation_mahalanobis_median=float(np.nanmedian(m)),
        empirical_chi2_3_coverage={
            str(alpha): float(np.mean(m[valid] <= chi2.ppf(alpha, 3)))
            for alpha in [0.5, 0.9, 0.95, 0.99]
        },
        spearman_predicted_trace_vs_squared_error=float(
            spearmanr(sigma, err**2).statistic
        ),
        scope="3D matching observation consistency with offline GT motion; NOT pose covariance or a full independent Gaussian calibration test",
        pose_covariance_available=False,
        unified_interface=dict(
            indexed_constraints=len(constraints),
            valid_constraints=sum(x.tracking_valid for x in constraints),
            fake_physical_timestamp_fusion_rejected=fuse_rejected,
        ),
    )
    (root / "residual_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    main(p.parse_args())
