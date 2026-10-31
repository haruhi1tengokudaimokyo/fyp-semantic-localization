"""Conventional stereo VO baseline: SGBM depth + temporal LK + PnP-RANSAC.
No global map updates, IMU, wheel, GT depth, or GT motion priors.
"""

import argparse, json, time, resource
from pathlib import Path
import cv2, numpy as np
from scipy.spatial.transform import Rotation
from motion.metrics import evaluate


def main(a):
    cv2.setNumThreads(4)
    cv2.setRNGSeed(1305)
    root = Path(a.dataset)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    left = sorted((root / "image_left").glob("*.png"))[: a.frames]
    right = sorted((root / "image_right").glob("*.png"))[: a.frames]
    assert len(left) == len(right) == a.frames
    K = np.array([[320.0, 0, 320], [0, 320.0, 240], [0, 0, 1.0]])
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=96,
        blockSize=5,
        P1=8 * 25,
        P2=32 * 25,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    poses = [np.eye(4)]
    lost = [False]
    rows = []
    prev = None
    prev_depth = None
    start = time.perf_counter()
    for i, (lp, rp) in enumerate(zip(left, right)):
        t = time.perf_counter()
        L = cv2.imread(str(lp), 0)
        R = cv2.imread(str(rp), 0)
        disp = matcher.compute(L, R).astype(float) / 16
        depth = np.zeros_like(disp)
        valid = disp > 1.0
        depth[valid] = 80.0 / disp[valid]
        count = 0
        good_count = 0
        bad = False
        if prev is not None:
            mask = ((prev_depth > 0) & (prev_depth < 50)).astype("uint8") * 255
            p0 = cv2.goodFeaturesToTrack(
                prev,
                maxCorners=600,
                qualityLevel=0.01,
                minDistance=8,
                mask=mask,
                blockSize=7,
            )
            if p0 is not None:
                p1, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev,
                    L,
                    p0,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        30,
                        0.01,
                    ),
                )
                back, bs, _ = cv2.calcOpticalFlowPyrLK(
                    L, prev, p1, None, winSize=(21, 21), maxLevel=3
                )
                f = (
                    (status.ravel() > 0)
                    & (bs.ravel() > 0)
                    & (
                        np.linalg.norm(back.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)
                        < 1.0
                    )
                )
                uv = p0.reshape(-1, 2)[f]
                uv1 = p1.reshape(-1, 2)[f]
                ij = np.rint(uv).astype(int)
                z = prev_depth[ij[:, 1], ij[:, 0]]
                xyz = np.c_[(uv[:, 0] - 320) * z / 320, (uv[:, 1] - 240) * z / 320, z]
                count = len(xyz)
                if count >= 20:
                    ok, rv, tv, inliers = cv2.solvePnPRansac(
                        xyz,
                        uv1,
                        K,
                        None,
                        iterationsCount=100,
                        reprojectionError=2.0,
                        confidence=0.999,
                        flags=cv2.SOLVEPNP_EPNP,
                    )
                    if ok and inliers is not None and len(inliers) >= 20:
                        good_count = len(inliers)
                        rv, tv = cv2.solvePnPRefineLM(
                            xyz[inliers[:, 0]], uv1[inliers[:, 0]], K, None, rv, tv
                        )
                        T = np.eye(4)
                        T[:3, :3] = cv2.Rodrigues(rv)[0]
                        T[:3, 3] = tv.ravel()
                        poses.append(poses[-1] @ np.linalg.inv(T))
                    else:
                        bad = True
                else:
                    bad = True
            else:
                bad = True
            if bad:
                poses.append(poses[-1].copy())
            lost.append(bad)
        rows.append(
            dict(
                frame=i,
                runtime_ms=(time.perf_counter() - t) * 1000,
                correspondences=count,
                pnp_inliers=good_count,
                lost=bad,
            )
        )
        prev = L
        prev_depth = depth
    # Offline evaluation only, after all estimated poses exist.
    raw = np.loadtxt(root / "pose_left.txt")[: len(poses)]
    gt = np.tile(np.eye(4), (len(raw), 1, 1))
    gt[:, :3, :3] = Rotation.from_quat(raw[:, 3:]).as_matrix()
    gt[:, :3, 3] = raw[:, :3]
    C = np.eye(4)
    C[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    gt = C.T @ gt @ C
    poses = np.array(poses)
    lost = np.array(lost)
    m = evaluate(poses, gt, lost)
    m.update(
        method="SGBM + LK + EPNP-RANSAC + LM refinement",
        wall_seconds=time.perf_counter() - start,
        runtime_mean_ms=float(np.mean([x["runtime_ms"] for x in rows])),
        runtime_p95_ms=float(np.percentile([x["runtime_ms"] for x in rows], 95)),
        maxrss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        parameters=dict(
            num_disparities=96,
            block_size=5,
            min_inliers=20,
            ransac_px=2,
            max_depth_m=50,
        ),
        pose_covariance=None,
    )
    np.savez_compressed(
        out / "trajectory.npz",
        T_world_camera=poses,
        T_world_camera_gt=gt,
        lost_before_interpolation=lost,
        frame_idx=np.arange(len(poses)),
    )
    (out / "summary.json").write_text(json.dumps(m, indent=2))
    (out / "per_frame.json").write_text(json.dumps(rows, indent=2))
    print(m)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("output")
    p.add_argument("--frames", type=int, default=200)
    main(p.parse_args())
