"""Strict, GT-free online contracts. Calibration and map identity are mandatory."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

IDENTITY = ("source_id", "building_id", "map_id", "camera_id")
FORBIDDEN = {"gt", "ground_truth", "T_gt", "gt_pose", "T_ground_truth",
             "T_before_gt", "is_correct", "oracle", "ground_truth_pose"}


def no_truth(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN or key.lower().startswith("gt_") or "ground_truth" in key.lower():
                raise ValueError(f"GT field forbidden in online input: {key}")
            no_truth(child)
    elif isinstance(value, list):
        for child in value:
            no_truth(child)


def pose(value):
    t = np.asarray(value, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError("pose must be finite 4x4 camera-to-map/odom SE(3)")
    if not np.allclose(t[3], [0, 0, 0, 1], atol=1e-7, rtol=0):
        raise ValueError("invalid homogeneous last row")
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5, rtol=0) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError("rotation must be proper SO(3), not reflection")
    return t


def discrepancy(a, b):
    a, b = pose(a), pose(b)
    return (float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            float(np.rad2deg(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude())))


def identity(row):
    values = tuple(row[k] for k in IDENTITY)
    if any(not isinstance(x, str) or not x for x in values):
        raise ValueError("source/building/map/camera IDs must be nonempty strings")
    return values


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("expected finite numeric value")
    return float(value)


def provenance(row):
    for key in ("image_sha256", "model_sha256", "calibration_sha256"):
        v = row["provenance"][key]
        if not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError(f"invalid {key}")


def validate_observation(row):
    no_truth(row)
    identity(row)
    if type(row.get("schema_version")) is not int or row["schema_version"] != 1 or row.get("source") not in {"ace", "spvloc"}:
        raise ValueError("unsupported observation schema/source")
    if not isinstance(row["query_id"], str) or not row["query_id"]:
        raise ValueError("empty query_id")
    capture = number(row["capture_time_s"])
    if number(row["completed_time_s"]) < capture:
        raise ValueError("completion precedes capture")
    provenance(row)
    if row["status"] not in {"VALID", "FAILED", "NO_CANDIDATE", "SKIPPED"}:
        raise ValueError("invalid observation status")
    pool = row["candidates"]
    if not isinstance(pool, list) or (bool(pool) != (row["status"] == "VALID")):
        raise ValueError("status/candidate disagreement")
    if any(type(x["rank"]) is not int for x in pool) or [x["rank"] for x in pool] != list(range(1, len(pool) + 1)):
        raise ValueError("candidate ranks must be ordered, contiguous and 1-based")
    if row["source"] == "ace" and len(pool) > 1:
        raise ValueError("current ACE exporter supplies one independent hypothesis")
    if len(pool) > 3:
        raise ValueError("SPVLoc experiment declares at most Top-3")
    for candidate in pool:
        pose(candidate["T_map_camera"])
        q = candidate["quality"]
        if row["source"] == "ace":
            n = q["pnp_inliers"]
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise ValueError("ACE quality is a nonnegative integer PnP inlier count")
        else:
            number(q["viewport_score"])
        if "probability_correct" in q and not q.get("calibration_artifact_sha256"):
            raise ValueError("uncalibrated matching score is not a probability")
    return row


def validate_motion(row):
    no_truth(row)
    identity(row)
    if type(row.get("schema_version")) is not int or row["schema_version"] != 1 or row.get("source") != "macvo":
        raise ValueError("motion requires MAC-VO schema v1")
    if not isinstance(row["query_id"], str) or not row["query_id"]:
        raise ValueError("motion query_id required")
    if number(row["completed_time_s"]) < number(row["capture_time_s"]):
        raise ValueError("motion completion precedes capture")
    if not isinstance(row["valid"], bool) or not isinstance(row["segment_id"], str):
        raise ValueError("motion validity and continuity segment required")
    if row["valid"]:
        pose(row["T_odom_camera"])
    elif row.get("T_odom_camera") is not None:
        raise ValueError("lost motion must not silently interpolate a pose")
    provenance(row)
    return row


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        for row in values:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)
