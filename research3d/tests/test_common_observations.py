"""Online fusion invariants and exact replay parity on real completed ACE/MACVO caches."""

import csv, os, unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from motion.common_observations import estimate, blend

BASE = Path(os.environ.get("FYP3D_BASE", "/nonexistent"))


class BlendTests(unittest.TestCase):
    def test_gauge_equivariant_weighted_correction(self):
        A = np.eye(4)
        A[:3, 3] = [1, 2, 3]
        B = np.eye(4)
        B[:3, :3] = Rotation.from_euler("xyz", [0.2, 0.1, -0.3]).as_matrix()
        B[:3, 3] = [3, 1, 4]
        G = np.eye(4)
        G[:3, :3] = Rotation.from_euler("xyz", [0.7, -0.2, 0.4]).as_matrix()
        G[:3, 3] = [12, -5, 3]
        np.testing.assert_allclose(
            blend(G @ A, G @ B, 0.4), G @ blend(A, B, 0.4), atol=1e-10
        )

    def test_no_absolute_observation_cannot_invent_global_gauge(self):
        d = np.repeat(np.eye(4)[None], 8, 0)
        d[1:, 0, 3] = 0.1
        P, rows = estimate(
            d,
            np.r_[False, np.ones(7, bool)],
            {"geometry": [[] for _ in d]},
            ["geometry"],
        )
        self.assertTrue(np.isnan(P).all())
        self.assertFalse(any(r["accepted_absolute"] for r in rows))


@unittest.skipUnless(
    (BASE / "runs/formal/common/continuous_joint/global_predictions.npz").exists(),
    "completed real common-scene cache required",
)
class RealReplayTests(unittest.TestCase):
    def setUp(self):
        root = BASE / "runs/formal/common"
        ace = np.load(root / "ace_seed2089/evaluation/predictions.npz")
        quality = list(
            csv.DictReader((root / "ace_seed2089/evaluation/per_frame.csv").open())
        )
        z = np.load(root / "macvo_paper600/trajectory.npz")
        T = z["T_world_camera_raw"]
        lost = z["lost_before_interpolation"]
        self.delta = np.repeat(np.eye(4)[None], len(T), 0)
        self.delta[1:] = np.linalg.inv(T[:-1]) @ T[1:]
        self.valid = np.r_[False, ~(lost[:-1] | lost[1:])]
        self.obs = {
            "ace": [
                [
                    dict(
                        T_world_camera=T.tolist(),
                        inliers=int(q["inliers"]),
                        median_reprojection_px=float(q["median_reprojection_px"]),
                    )
                ]
                for T, q in zip(ace["T_world_camera"], quality)
            ]
        }

    def test_existing_ace_gate_cache_float_precision_parity(self):
        P, _ = estimate(self.delta, self.valid, self.obs, ["ace"])
        baseline = np.load(
            BASE / "runs/formal/common/continuous_joint/global_predictions.npz"
        )["MACVO_ACE_gate"]
        self.assertTrue(np.array_equal(np.isfinite(P), np.isfinite(baseline)))
        # Original cache propagated float32 matrices; the new interface uses float64.
        np.testing.assert_allclose(P, baseline, rtol=0, atol=1e-6, equal_nan=True)

    def test_actual_cache_global_gauge_equivariance(self):
        P, _ = estimate(self.delta, self.valid, self.obs, ["ace"], weighted=True)
        G = np.eye(4)
        G[:3, :3] = Rotation.from_euler("xyz", [0.5, -0.3, 0.7]).as_matrix()
        G[:3, 3] = [3, -7, 2]
        transformed = {
            "ace": [
                [
                    {
                        **h,
                        "T_world_camera": (G @ np.array(h["T_world_camera"])).tolist(),
                    }
                    for h in pool
                ]
                for pool in self.obs["ace"]
            ]
        }
        Q, _ = estimate(self.delta, self.valid, transformed, ["ace"], weighted=True)
        np.testing.assert_allclose(Q, G @ P, atol=1e-8, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
