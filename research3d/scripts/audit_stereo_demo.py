"""Audit real official demo files, pairing and epipolar behavior without GT depth."""

import argparse, hashlib, json
from pathlib import Path
import cv2, numpy as np


def main(a):
    root = Path(a.root)
    left = sorted((root / "image_left").glob("*.png"))
    right = sorted((root / "image_right").glob("*.png"))
    pose = np.loadtxt(root / "pose_left.txt")
    assert len(left) == len(right) == len(pose) > 200
    assert [p.name.split("_")[0] for p in left] == [p.name.split("_")[0] for p in right]
    assert pose.shape[1] == 7 and np.isfinite(pose).all()
    norm = np.linalg.norm(pose[:, 3:], axis=1)
    assert np.max(abs(norm - 1)) < 1e-3
    sampled = []
    sift = cv2.SIFT_create(nfeatures=1200)
    bf = cv2.BFMatcher()
    for i in np.linspace(0, len(left) - 1, 12).astype(int):
        L = cv2.imread(str(left[i]), cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(str(right[i]), cv2.IMREAD_GRAYSCALE)
        assert L.shape == R.shape == (480, 640)
        kp1, d1 = sift.detectAndCompute(L, None)
        kp2, d2 = sift.detectAndCompute(R, None)
        m = bf.knnMatch(d1, d2, k=2)
        good = [x for x, y in m if x.distance < 0.7 * y.distance]
        dy = np.array(
            [abs(kp1[x.queryIdx].pt[1] - kp2[x.trainIdx].pt[1]) for x in good]
        )
        sampled.append(
            dict(
                frame=int(i),
                matches=len(good),
                median_vertical_difference_px=float(np.median(dy)),
                fraction_within_1px=float(np.mean(dy < 1)),
            )
        )
    files = []
    for p in left + right + [root / "pose_left.txt"]:
        files.append(
            dict(
                path=str(p.relative_to(root)),
                bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            )
        )
    report = dict(
        source="MAC-VO official Google Drive demo, selected TartanAir sequence",
        frames=len(left),
        K=[[320, 0, 320], [0, 320, 240], [0, 0, 1]],
        baseline_m=0.25,
        calibration_source="official TartanAir_StereoSequence loader, fixed benchmark calibration",
        pair_ids_verified=True,
        quaternion_max_norm_error=float(np.max(abs(norm - 1))),
        has_original_time_file=(root / "imu/cam_time.npy").exists(),
        online_modalities=["left RGB", "right RGB"],
        imu_provided=False,
        wheel_provided=False,
        gt_depth_provided=False,
        epipolar_samples=sampled,
        files=files,
        license_note="Official MAC-VO demo links TartanAir; use for non-commercial research, original data terms apply. Not republished as a derivative dataset.",
    )
    Path(a.output).write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k != "files"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("output")
    main(p.parse_args())
