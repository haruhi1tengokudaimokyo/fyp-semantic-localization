import hashlib, tempfile, unittest
from pathlib import Path
import numpy as np
from knownmap.contracts import *


class Contracts(unittest.TestCase):
    def test_frames(self):
        with self.assertRaises(ValueError):
            se3(np.diag([1, 1, -1, 1]))
        with self.assertRaises(ValueError):
            MotionEstimate(np.eye(4), 2, 1, None, "camera")

    def test_covariance(self):
        with self.assertRaises(ValueError):
            covariance(-np.eye(6))
        self.assertIsNone(covariance(None))

    def test_readonly_digest(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "map.npz"
            np.savez(p, points_world=np.ones((4, 3)))
            m = ReadOnlyMap(
                p, hashlib.sha256(p.read_bytes()).hexdigest(), "test", ("ref0",)
            )
            v = m.load()
            with self.assertRaises(ValueError):
                v["points_world"][0, 0] = 3
            p.write_bytes(b"mutated")
            with self.assertRaises(ValueError):
                m.load()

    def test_no_gt_interface(self):
        self.assertFalse(
            any(
                "pose" in name or "gt" in name or "truth" in name
                for name in SensorObservation.__dataclass_fields__
            )
        )

    def test_semantic_probs(self):
        with self.assertRaises(ValueError):
            SemanticObservation(
                "x", np.zeros((3, 3, 2)), ("wall", "door"), "prediction"
            )


if __name__ == "__main__":
    unittest.main()
