"""Real 7Scenes RGB-D motion, registering raw depth into RGB using official ACE calibration.
Uses no test poses. Depth is a real Kinect sensor, not a GT motion source.
"""

import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np


def main(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(a.scene)
    sys.path.insert(0, str(Path(a.ace) / "datasets"))
    from setup_7scenes import d_to_rgb

    cv2.setNumThreads(4)
    cv2.setRNGSeed(1305)
    files = sorted((root / "test/rgb").glob("*.color.png"))
    K = np.array([[525.0, 0, 320], [0, 525, 240], [0, 0, 1]])
    y, x = np.mgrid[:480, :640]
    rays = np.stack([(x - 320) / 585, (y - 240) / 585, np.ones_like(x)], -1)
    deltas = [np.eye(4)]
    valids = [False]
    rows = []
    previous = None
    previous_depth = None
    for i, p in enumerate(files):
        start = time.perf_counter()
        gray = cv2.imread(str(p), 0)
        source = p.resolve()
        raw = cv2.imread(
            str(source.with_name(source.name.replace(".color.png", ".depth.png"))),
            cv2.IMREAD_UNCHANGED,
        )
        assert raw is not None
        valid = (raw > 0) & (raw < 10000)
        points = rays[valid] * (raw[valid, None] / 1000)
        rgb = points @ d_to_rgb[:3, :3].T + d_to_rgb[:3, 3]
        uv = np.rint(rgb[:, :2] / rgb[:, 2:] * 525 + [320, 240]).astype(int)
        mask = (
            (rgb[:, 2] > 0.1)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < 640)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < 480)
        )
        zbuffer = np.full(640 * 480, np.inf)
        np.minimum.at(zbuffer, uv[mask, 1] * 640 + uv[mask, 0], rgb[mask, 2])
        depth = zbuffer.reshape(480, 640)
        depth[~np.isfinite(depth)] = 0
        ok = False
        nin = 0
        delta = np.eye(4)
        if previous is not None:
            corners = cv2.goodFeaturesToTrack(
                previous,
                600,
                0.01,
                8,
                mask=(previous_depth > 0.1).astype(np.uint8) * 255,
                blockSize=7,
            )
            if corners is not None:
                tracked, st, _ = cv2.calcOpticalFlowPyrLK(
                    previous, gray, corners, None, winSize=(21, 21), maxLevel=3
                )
                back, sb, _ = cv2.calcOpticalFlowPyrLK(
                    gray, previous, tracked, None, winSize=(21, 21), maxLevel=3
                )
                keep = (
                    (st[:, 0] > 0)
                    & (sb[:, 0] > 0)
                    & (np.linalg.norm(back[:, 0] - corners[:, 0], axis=1) < 1)
                )
                u = corners[:, 0][keep]
                v = tracked[:, 0][keep]
                ij = np.rint(u).astype(int)
                d = previous_depth[ij[:, 1], ij[:, 0]]
                P = np.c_[(u[:, 0] - 320) * d / 525, (u[:, 1] - 240) * d / 525, d]
                if len(P) >= 20:
                    success, rv, tv, inliers = cv2.solvePnPRansac(
                        P,
                        v,
                        K,
                        None,
                        iterationsCount=1000,
                        reprojectionError=2.0,
                        confidence=0.999,
                        flags=cv2.SOLVEPNP_EPNP,
                    )
                    nin = 0 if inliers is None else len(inliers)
                    if success and nin >= 20:
                        ii = inliers[:, 0]
                        rv, tv = cv2.solvePnPRefineLM(P[ii], v[ii], K, None, rv, tv)
                        T = np.eye(4)
                        T[:3, :3] = cv2.Rodrigues(rv)[0]
                        T[:3, 3] = tv[:, 0]
                        delta = np.linalg.inv(T)
                        ok = bool(np.isfinite(delta).all())
            deltas.append(delta)
            valids.append(ok)
        rows.append(
            dict(
                frame=p.name,
                inliers=nin,
                valid=ok,
                runtime_ms=(time.perf_counter() - start) * 1000,
                registered_depth_fraction=float((depth > 0).mean()),
            )
        )
        previous = gray
        previous_depth = depth
        if i % 100 == 0:
            print("FRAME", i, "valid", ok, "inliers", nin, flush=True)
    np.savez_compressed(
        out / "motion.npz",
        T_previous_current=np.stack(deltas),
        valid=np.array(valids),
        frame=np.array([p.name for p in files]),
    )
    (out / "per_frame.json").write_text(json.dumps(rows, indent=2))
    (out / "protocol.json").write_text(
        json.dumps(
            dict(
                sensor="real RGB and Kinect depth; no IMU/wheel or GT poses",
                depth_registration="raw depth K585, official ACE d_to_rgb extrinsic, RGB K525, nearest projected pixel and nearest-Z buffer",
                calibration_caveat="ACE default calibration, approximate device calibration",
                frames=len(files),
                available_motion_pairs=int(sum(valids)),
                timestamp_scope="frame index only, physical timestamps unavailable",
                runtime_mean_ms=float(np.mean([r["runtime_ms"] for r in rows])),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scene")
    p.add_argument("output")
    p.add_argument("--ace", required=True)
    main(p.parse_args())
