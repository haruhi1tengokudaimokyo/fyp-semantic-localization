from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import hashlib
import json
import numpy as np


def immutable(a, shape=None):
    a = np.array(a, dtype=np.float64, copy=True)
    if shape and a.shape != shape:
        raise ValueError(f"Expected {shape}, got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("Non-finite array")
    a.flags.writeable = False
    return a


def se3(a):
    a = immutable(a, (4, 4))
    if not np.allclose(a[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Invalid homogeneous row")
    if not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=2e-3) or not np.isclose(
        np.linalg.det(a[:3, :3]), 1, atol=2e-3
    ):
        raise ValueError("Invalid SO(3)")
    return a


def covariance(a):
    if a is None:
        return None
    a = immutable(a, (6, 6))
    if not np.allclose(a, a.T) or np.linalg.eigvalsh(a).min() < -1e-10:
        raise ValueError("Covariance must be symmetric PSD")
    return a


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    K: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "K", immutable(self.K, (3, 3)))
        if (
            self.width <= 0
            or self.height <= 0
            or self.K[0, 0] <= 0
            or self.K[1, 1] <= 0
        ):
            raise ValueError("Invalid camera")


@dataclass(frozen=True)
class SensorObservation:
    """No ground truth field. stamp_ns=None for datasets lacking sensor timestamps."""

    frame_id: str
    camera: Camera
    rgb_path: Path
    stamp_ns: Optional[int] = None
    sensor_frame: str = "camera_rgb_optical"


@dataclass(frozen=True)
class MotionEstimate:
    T_previous_current: np.ndarray
    previous_stamp_ns: int
    current_stamp_ns: int
    covariance_se3: Optional[np.ndarray]
    source: str

    def __post_init__(self):
        object.__setattr__(self, "T_previous_current", se3(self.T_previous_current))
        object.__setattr__(self, "covariance_se3", covariance(self.covariance_se3))
        if self.current_stamp_ns <= self.previous_stamp_ns:
            raise ValueError("Nonmonotonic motion timestamps")


@dataclass(frozen=True)
class SemanticObservation:
    frame_id: str
    class_probabilities: np.ndarray
    class_names: tuple
    provenance: str

    def __post_init__(self):
        p = immutable(self.class_probabilities)
        if (
            p.ndim != 3
            or p.shape[-1] != len(self.class_names)
            or (p < 0).any()
            or not np.allclose(p.sum(-1), 1, atol=1e-4)
        ):
            raise ValueError("Expected normalized HWC probabilities")
        object.__setattr__(self, "class_probabilities", p)


@dataclass(frozen=True)
class MapCorrespondence:
    pixels_xy: np.ndarray
    points_world: np.ndarray
    reprojection_error_px: np.ndarray

    def __post_init__(self):
        p = immutable(self.pixels_xy)
        q = immutable(self.points_world)
        e = immutable(self.reprojection_error_px)
        if (
            p.ndim != 2
            or p.shape[1] != 2
            or q.shape != (len(p), 3)
            or e.shape != (len(p),)
        ):
            raise ValueError("Correspondence shapes")
        object.__setattr__(self, "pixels_xy", p)
        object.__setattr__(self, "points_world", q)
        object.__setattr__(self, "reprojection_error_px", e)


@dataclass(frozen=True)
class PoseHypothesis:
    T_world_camera: np.ndarray
    score: float
    score_kind: str
    covariance_se3: Optional[np.ndarray] = None

    def __post_init__(self):
        object.__setattr__(self, "T_world_camera", se3(self.T_world_camera))
        object.__setattr__(self, "covariance_se3", covariance(self.covariance_se3))


@dataclass(frozen=True)
class PoseConstraint:
    hypothesis: PoseHypothesis
    map_id: str
    frame_id: str
    source: str


@dataclass(frozen=True)
class LocalizationDiagnostics:
    runtime_ms: float
    inliers: int
    correspondences: int
    median_reprojection_px: float
    status: str


@dataclass(frozen=True)
class LocalizationState:
    frame_id: str
    hypotheses: tuple
    diagnostics: LocalizationDiagnostics
    # A research estimate is not automatically an accepted tracking correction.
    accepted: bool = False


@dataclass(frozen=True)
class ReadOnlyMap:
    geometry_path: Path
    geometry_sha256: str
    map_id: str
    reference_ids: tuple
    reference_catalog_path: Optional[Path] = None
    reference_catalog_sha256: Optional[str] = None

    def load(self):
        if (
            hashlib.sha256(self.geometry_path.read_bytes()).hexdigest()
            != self.geometry_sha256
        ):
            raise ValueError("Map digest mismatch")
        with np.load(self.geometry_path, allow_pickle=False) as z:
            return {k: immutable(z[k]) for k in z.files}

    def references(self):
        """Verified train-only posed RGB observations, in the geometry map frame."""
        if self.reference_catalog_path is None:
            raise ValueError("This map has no reference catalog")
        content = self.reference_catalog_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != self.reference_catalog_sha256:
            raise ValueError("Reference catalog digest mismatch")
        rows = json.loads(content)
        if tuple(r["reference_id"] for r in rows) != self.reference_ids:
            raise ValueError("Reference ID mismatch")
        result = []
        for row in rows:
            if row["split"] != "train":
                raise ValueError("Map reference must come from train split")
            observation = SensorObservation(
                row["reference_id"],
                Camera(row["width"], row["height"], row["K"]),
                Path(row["rgb_path"]),
            )
            result.append((observation, se3(row["T_world_camera"])))
        return tuple(result)
