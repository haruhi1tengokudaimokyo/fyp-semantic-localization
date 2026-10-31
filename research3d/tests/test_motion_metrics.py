import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from motion.metrics import evaluate


class TrajectoryMetrics(unittest.TestCase):
    def test_rigid_gauge_and_no_scale_cheat(self):
        gt = np.tile(np.eye(4), (30, 1, 1))
        gt[:, :3, 3] = np.stack(
            [
                np.linspace(0, 3, 30),
                np.sin(np.linspace(0, 3, 30)),
                np.cos(np.linspace(0, 3, 30)),
            ],
            -1,
        )
        A = np.eye(4)
        A[:3, :3] = Rotation.from_euler("xyz", [0.2, 0.3, 0.4]).as_matrix()
        A[:3, 3] = [3, 2, 1]
        m = evaluate(A @ gt, gt)
        self.assertLess(m["ate_se3_aligned_rmse_m"], 1e-12)
        self.assertLess(m["rpe_frame_delta"]["1"]["translation_rmse_m"], 1e-12)
        wrong = gt.copy()
        wrong[:, :3, 3] *= 2
        self.assertGreater(evaluate(wrong, gt)["ate_se3_aligned_rmse_m"], 0.1)

    def test_failures_kept_in_denominator(self):
        gt = np.tile(np.eye(4), (20, 1, 1))
        gt[:, 0, 3] = np.arange(20)
        lost = np.zeros(20, bool)
        lost[4:8] = True
        m = evaluate(gt, gt, lost)
        self.assertEqual(m["lost_frames"], 4)
        self.assertEqual(m["max_failure_run_frames"], 4)
        self.assertAlmostEqual(m["tracking_availability"], 15 / 19)


if __name__ == "__main__":
    unittest.main()
