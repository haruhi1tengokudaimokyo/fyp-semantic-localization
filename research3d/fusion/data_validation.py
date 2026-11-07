"""Independent on-disk QA for the stereo capture contract (no GT online)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def _rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _safe(root, relative):
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("path escapes dataset root")
    return path


def _pose(value):
    pose = np.asarray(value, float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("invalid pose shape or values")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("invalid homogeneous pose")
    if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6) or abs(np.linalg.det(pose[:3, :3]) - 1) > 1e-6:
        raise ValueError("pose rotation must be proper SO(3), not reflection")
    return pose


def stereo_depth_check(left, right, K, baseline):
    """Rectified depth reprojection using independent right depth.

    Uses only smooth interior left depth and visible right correspondences;
    occlusion is explicitly filtered. Requires coverage and disparity rather
    than allowing an empty/zero-baseline projection to pass. Values are fixed
    data-QA limits, not localization test-set tuned thresholds.
    """
    yy, xx = np.mgrid[2:left.shape[0]-2:4, 2:left.shape[1]-2:4]
    depth = left[yy, xx]
    disparity = K[0, 0] * baseline / np.maximum(depth, 1e-9)
    rx = xx - disparity
    ix = np.rint(rx).astype(int)
    valid = (depth > .15) & (depth < 30) & (ix >= 0) & (ix < right.shape[1])
    smooth = np.ones_like(valid)
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        smooth &= abs(left[yy + dy, xx + dx] - depth) < .03
    valid &= smooth
    rz = right[yy, np.clip(ix, 0, right.shape[1]-1)]
    # Reject occluded correspondence only, not arbitrary residual outliers.
    visible = valid & (rz >= depth - .05) & (rz > .15) & (rz < 30)
    residual = abs(rz[visible] - depth[visible])
    return dict(samples=int(visible.sum()), eligible_samples=int(valid.sum()),
                visible_fraction=float(visible.sum() / max(1, valid.sum())),
                p95_depth_residual_m=float(np.percentile(residual, 95)) if len(residual) else None,
                median_disparity_px=float(np.median(disparity[visible])) if len(residual) else None)


def structural_depth_check(depth, instance_rgb, colors, K, pose, house):
    """Project measured structural pixels into the independent source polygons.

    The rendered instance dictionary identifies actual floor/wall/ceiling
    pixels; furniture is never relabeled as architecture. Compare them against
    original house polygons in metres, with finite bounds rather than arbitrary
    fitted planes. The 15cm limit is inherited from prior ProcTHOR qualification
    to allow polygon reference-surface vs finite physical-wall thickness.
    """
    yy, xx = np.mgrid[4:depth.shape[0]-4:4, 4:depth.shape[1]-4:4]
    raster = instance_rgb.astype(np.int64)
    codes = (raster[..., 0] << 16) + (raster[..., 1] << 8) + raster[..., 2]
    sampled = codes[yy, xx]
    interior = np.ones_like(sampled, bool)
    for dx, dy in [(3, 0), (-3, 0), (0, 3), (0, -3)]:
        interior &= sampled == codes[yy+dy, xx+dx]
    z = depth[yy, xx]
    local = np.stack([(xx-K[0, 2])*z/K[0, 0], (yy-K[1, 2])*z/K[1, 1], z], axis=-1)
    world = local @ pose[:3, :3].T + pose[:3, 3]
    # Inverse canonical map (X,Y,Z)=(Unity x,z,y), no learned/fitted alignment.
    points = world[..., [0, 2, 1]]
    errors, counts = [], {}
    for label in ("room", "Ceiling_room", "wall"):
        accepted_codes = [(int(c[0]) << 16) + (int(c[1]) << 8) + int(c[2])
                          for key, c in colors.items() if key.split("|")[0] == label]
        visible = interior & np.isin(sampled, accepted_codes) & (z > .1) & (z < 30.)
        p = points[visible]
        counts[label] = int(len(p))
        if not len(p):
            continue
        polygons = []
        if label == "wall":
            polygons = [np.array([[v[k] for k in "xyz"] for v in wall["polygon"]], float)
                        for wall in house["walls"] if not wall.get("empty", False)]
        else:
            for room in house["rooms"]:
                polygon = np.array([[v[k] for k in "xyz"] for v in room["floorPolygon"]], float)
                if label == "Ceiling_room":
                    heights = [v["y"] for w in house["walls"] if w.get("roomId") == room["id"] for v in w["polygon"]]
                    polygon[:, 1] = max(heights)
                polygons.append(polygon)
        residual = np.full(len(p), np.inf)
        for polygon in polygons:
            # Axis-aligned bounds reject other parallel walls/rooms. House
            # polygons themselves provide the measured reference plane.
            normal = np.cross(polygon[1]-polygon[0], polygon[2]-polygon[0])
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            distance = abs((p - polygon[0]) @ (normal / norm))
            bounded = np.all((p >= polygon.min(0)-.15) & (p <= polygon.max(0)+.15), axis=1)
            residual = np.minimum(residual, np.where(bounded, distance, np.inf))
        errors.extend(residual.tolist())
    values = np.asarray(errors)
    finite = np.isfinite(values)
    return dict(samples=int(len(values)), structural_sample_counts=counts,
                finite_correspondence_fraction=float(finite.mean()) if len(values) else 0.,
                p95_plane_distance_m=float(np.percentile(values[finite], 95)) if finite.any() else None)


def validate_capture(root: Path):
    errors, metrics = [], {}
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        cal = json.loads((root / "calibration.json").read_text())
        calibration_sha = hashlib.sha256((root / "calibration.json").read_bytes()).hexdigest()
        K = np.asarray(cal["K"], float)
        baseline = float(cal["baseline_m"])
        if not 0 < baseline < .4:
            raise ValueError("baseline must be nonzero")
        relative = _pose(cal["T_left_right"])
        if not np.allclose(relative[:3, 3], [baseline, 0, 0], atol=1e-7):
            raise ValueError("rectified baseline extrinsic mismatch")
        if not cal["native_resolution"] or cal["width"] < 320 or cal["height"] < 240:
            raise ValueError("native resolution policy violated")
        if cal["depth_convention"] != "optical_axis_z_metres; OFFLINE_ONLY":
            raise ValueError("unrecognized metric depth convention")
        expected_f = cal["height"] / (2 * np.tan(np.deg2rad(cal["vertical_fov_deg"]) / 2))
        if not np.allclose(K, [[expected_f, 0, (cal["width"]-1)/2], [0, expected_f, (cal["height"]-1)/2], [0, 0, 1]]):
            raise ValueError("intrinsics inconsistent with native vertical FoV")
        annotation = json.loads((root / "map/annotation_3d.json").read_text())
        house = json.loads((root / "map/procthor_house.json").read_text())
        types = {x["type"] for x in annotation["planes"]}
        if not {"wall", "floor", "ceiling"}.issubset(types):
            raise ValueError("structural map lacks room envelope")
        metrics["structure_plane_counts"] = {t: sum(p["type"] == t for p in annotation["planes"]) for t in sorted(types)}
        all_positions = {}
        for split in ("reference", "query"):
            online = _rows(root / "online" / split / "frames.jsonl")
            offline = _rows(root / "offline" / f"{split}_poses.jsonl")
            if len(online) != len(offline) or len(online) < 3 or len(online) != manifest["captured_frames"][split]:
                raise ValueError(f"{split}: row count mismatch")
            times = np.array([r["timestamp_s"] for r in online])
            if not np.allclose(np.diff(times), 1 / cal["fps"], atol=1e-8):
                raise ValueError(f"{split}: nonuniform/nonmonotonic acquisition time")
            sequences = [r["event_sequence_id"] for r in online]
            if len(set(sequences)) != len(sequences) or np.any(np.diff(sequences) <= 0):
                raise ValueError("reused/nonmonotonic simulator Event")
            poses, reprojections, structure_checks, images_seen = [], [], [], set()
            for row, off in zip(online, offline):
                allowed = {"source_id", "building_id", "map_id", "camera_id", "query_id", "frame_id", "timestamp_s",
                           "capture_time_s", "render_completed_time_s", "event_sequence_id", "same_event_stereo",
                           "left_rgb", "right_rgb", "left_sha256", "right_sha256"}
                if set(row) != allowed:
                    raise ValueError("online manifest fields differ from strict allowlist (possible GT leakage)")
                if row["capture_time_s"] != row["timestamp_s"]:
                    raise ValueError("synthetic capture clocks differ")
                for key in ("source_id", "building_id", "map_id", "camera_id", "query_id", "frame_id", "timestamp_s"):
                    if row[key] != off[key]:
                        raise ValueError("online/offline identity mismatch")
                if not isinstance(off.get("provenance"), dict) or off["provenance"].get("calibration_sha256") != calibration_sha:
                    raise ValueError("offline provenance must be an object bound to calibration SHA")
                if row["map_id"] != cal["map_id"] or row["map_id"] != manifest["map_id"] or not row["same_event_stereo"]:
                    raise ValueError("map ID / synchronization mismatch")
                left_pose, right_pose = _pose(off["T_map_camera"]), _pose(off["T_map_right"])
                if not np.allclose(np.linalg.inv(left_pose) @ right_pose, relative, atol=1e-5):
                    raise ValueError("actual eye geometry is not the declared stereo rig")
                depths = []
                for eye in ("left", "right"):
                    image_path = _safe(root, row[f"{eye}_rgb"])
                    if image_path in images_seen:
                        raise ValueError("image path reused for different samples/eyes")
                    images_seen.add(image_path)
                    with Image.open(image_path) as image:
                        image.load()
                        if image.mode != "RGB" or image.size != (cal["width"], cal["height"]):
                            raise ValueError("invalid RGB dimensions/mode")
                    if hashlib.sha256(image_path.read_bytes()).hexdigest() != row[f"{eye}_sha256"]:
                        raise ValueError("RGB checksum mismatch")
                    depth = np.load(_safe(root, off[f"{eye}_depth"]), allow_pickle=False)
                    if depth.shape != (cal["height"], cal["width"]) or not np.isfinite(depth).all() or (depth > 0).mean() < .95:
                        raise ValueError("depth shape/finite/positive coverage failure")
                    if ((depth > .1) & (depth < 30.)).mean() < .5:
                        raise ValueError("insufficient usable indoor metric depth coverage")
                    depths.append(depth)
                reprojection = stereo_depth_check(*depths, K, baseline)
                if reprojection["samples"] < 100 or reprojection["visible_fraction"] < .5 or reprojection["p95_depth_residual_m"] > .05:
                    raise ValueError("stereo metric depth reprojection QA failed: " + str(reprojection))
                reprojections.append(reprojection)
                with Image.open(_safe(root, off["left_instance"])) as image:
                    instance = np.asarray(image.convert("RGB"))
                if instance.shape != (cal["height"], cal["width"], 3):
                    raise ValueError("instance raster shape mismatch")
                colors = json.loads(_safe(root, off["instance_dictionary"]).read_text())
                structure = structural_depth_check(depths[0], instance, colors, K, left_pose, house)
                if (structure["samples"] < 100 or structure["finite_correspondence_fraction"] < .95
                        or structure["p95_plane_distance_m"] > .15):
                    raise ValueError("independent structural projection QA failed: " + str(structure))
                structure_checks.append(structure)
                poses.append(left_pose)
            poses = np.array(poses)
            # Compare rig centers (left eye shifts while turning).
            centers = poses[:, :3, 3] + poses[:, :3, 0] * baseline / 2
            all_positions[split] = centers
            steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            rotations = np.einsum("nji,njk->nik", poses[:-1, :3, :3], poses[1:, :3, :3])
            angles = np.rad2deg(np.arccos(np.clip((np.trace(rotations, axis1=1, axis2=2)-1)/2, -1, 1)))
            cfg = manifest["config"]
            if steps.max() > cfg["translation_speed_m_s"] / cal["fps"] + 1e-5:
                raise ValueError("teleport discontinuity / translation speed limit violated")
            if angles.max() > cfg["angular_speed_deg_s"] / cal["fps"] + .001:
                raise ValueError("rotation discontinuity / angular speed limit violated")
            if steps.sum() <= 0:
                raise ValueError("trajectory has no translation")
            metrics[split] = dict(frames=len(online), duration_s=float(times[-1] - times[0]),
                                  path_length_m=float(steps.sum()), max_step_m=float(steps.max()),
                                  max_yaw_step_deg=float(angles.max()),
                                  worst_reprojection_p95_m=max(x["p95_depth_residual_m"] for x in reprojections),
                                  worst_structural_p95_m=max(x["p95_plane_distance_m"] for x in structure_checks),
                                  structural_samples=sum(x["samples"] for x in structure_checks))
        # Avoid O(N*M) memory for full trajectories.
        separation = min(float(np.linalg.norm(all_positions["query"] - p, axis=1).min()) for p in all_positions["reference"])
        metrics["reference_query_min_rig_center_separation_m"] = separation
        if separation < .001:
            raise ValueError("reference/query capture positions overlap")
        metrics["formal_sequence_complete"] = not manifest["pilot_truncated"]
        metrics["semantic_supervision"] = "source structural annotation only; no raster-class supervision claimed"
    except (ValueError, KeyError, OSError, TypeError) as exc:
        errors.append(str(exc))
    return dict(schema="fyp.stereo_qa.v1", passed=not errors, errors=errors, metrics=metrics,
                scope="stereo pixels, metric depth cross-view consistency, source structural polygon projection, poses and split identity; not localization accuracy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    audit = validate_capture(args.root)
    print(json.dumps(audit, indent=2))
    raise SystemExit(0 if audit["passed"] else 1)


if __name__ == "__main__":
    main()
