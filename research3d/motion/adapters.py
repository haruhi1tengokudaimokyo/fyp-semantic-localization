"""Typed, timestamp-honest motion constraints for downstream state estimation."""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from knownmap.contracts import se3, MotionEstimate


@dataclass(frozen=True)
class IndexedMotionConstraint:
    previous_frame: int
    current_frame: int
    T_previous_current: np.ndarray
    tracking_valid: bool
    source: str
    previous_stamp_ns: Optional[int] = None
    current_stamp_ns: Optional[int] = None
    covariance_se3: None = None

    def __post_init__(self):
        object.__setattr__(self, "T_previous_current", se3(self.T_previous_current))
        if self.current_frame <= self.previous_frame:
            raise ValueError("Nonmonotonic frame indices")
        if (self.previous_stamp_ns is None) != (self.current_stamp_ns is None):
            raise ValueError("Both physical timestamps required")
        if (
            self.current_stamp_ns is not None
            and self.current_stamp_ns <= self.previous_stamp_ns
        ):
            raise ValueError("Nonmonotonic physical timestamps")

    def for_timed_estimator(self):
        if not self.tracking_valid:
            raise ValueError("Cannot fuse a lost-track constraint")
        if self.current_stamp_ns is None:
            raise ValueError(
                "Dataset has no physical sensor timestamps; fusion prohibited"
            )
        return MotionEstimate(
            self.T_previous_current,
            self.previous_stamp_ns,
            self.current_stamp_ns,
            None,
            self.source,
        )


def load_macvo_constraints(directory):
    from pathlib import Path

    root = Path(directory)
    with np.load(root / "trajectory.npz", allow_pickle=False) as z:
        poses = z["T_world_camera_raw"]
        lost = z["lost_before_interpolation"]
    return tuple(
        IndexedMotionConstraint(
            i,
            i + 1,
            np.linalg.inv(poses[i]) @ poses[i + 1],
            not bool(lost[i] or lost[i + 1]),
            "MACVO_FP32_camera_optical",
        )
        for i in range(len(poses) - 1)
    )
