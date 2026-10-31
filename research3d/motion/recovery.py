"""Frame-indexed multi-observation recovery gate; no GT or fabricated covariance.
Coherent wrong global modes can defeat temporal consistency. This limitation is
explicitly evaluated, rather than treating three consistent poses as a guarantee.
"""

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from knownmap.contracts import se3


def discrepancy(A, B):
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3])), float(
        np.degrees(Rotation.from_matrix(A[:3, :3].T @ B[:3, :3]).magnitude())
    )


@dataclass
class RecoveryGate:
    translation_gate_m: float = 0.10
    rotation_gate_deg: float = 10.0
    required_consistent: int = 3
    min_absolute_inliers: int = 50

    def __post_init__(self):
        self.T = None
        self.pending = None
        self.pending_count = 0
        self.status = "UNINITIALIZED"

    def step(self, motion, motion_valid, absolute, inliers):
        if motion_valid:
            motion = se3(motion)
        predicted = (
            None
            if self.T is None
            else (self.T @ motion if motion_valid else self.T.copy())
        )
        candidate = (
            se3(absolute)
            if absolute is not None
            and np.isfinite(absolute).all()
            and inliers >= self.min_absolute_inliers
            else None
        )
        accepted = False
        recovered = False
        if candidate is not None:
            if predicted is not None and all(
                v <= g
                for v, g in zip(
                    discrepancy(predicted, candidate),
                    (self.translation_gate_m, self.rotation_gate_deg),
                )
            ):
                self.T = candidate.copy()
                self.status = "TRACKING"
                self.pending = None
                self.pending_count = 0
                accepted = True
            else:
                consistent = False
                if self.pending is not None and motion_valid:
                    consistent = all(
                        v <= g
                        for v, g in zip(
                            discrepancy(self.pending @ motion, candidate),
                            (self.translation_gate_m, self.rotation_gate_deg),
                        )
                    )
                self.pending_count = self.pending_count + 1 if consistent else 1
                self.pending = candidate.copy()
                if self.pending_count >= self.required_consistent:
                    self.T = candidate.copy()
                    self.status = "TRACKING"
                    accepted = True
                    recovered = True
                    self.pending = None
                    self.pending_count = 0
                else:
                    self.T = predicted
                    self.status = "RECOVERING"
        else:
            self.pending = None
            self.pending_count = 0
            self.T = predicted
            self.status = (
                "DEAD_RECKONING" if predicted is not None and motion_valid else "LOST"
            )
        return dict(
            T_world_camera=None if self.T is None else self.T.copy(),
            accepted_absolute=accepted,
            recovered=recovered,
            status=self.status,
            pending_count=self.pending_count,
            pose_covariance=None,
        )
