"""Actual TUM RGB/depth regression and person exclusion failure semantics."""

import importlib.util
import os
from pathlib import Path
import unittest
import cv2
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_tum_rgbd_vo.py"
spec = importlib.util.spec_from_file_location("tum_vo", SCRIPT)
vo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vo)
BASE = Path(os.environ.get("FYP3D_BASE", "/nonexistent"))


@unittest.skipUnless(
    (BASE / "datasets/tum_rgbd/sitting_static/rgb.txt").is_file(),
    "real TUM sequence required",
)
class PersonExclusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = BASE / "datasets/tum_rgbd/sitting_static"
        rgb, depth = vo.entries(root / "rgb.txt"), vo.entries(root / "depth.txt")
        mapping = vo.associate(rgb, depth)
        cls.i = next(i for i in range(100, len(rgb) - 1) if i in mapping)
        cls.previous = cv2.imread(str(root / rgb[cls.i][1]), 0)
        cls.current = cv2.imread(str(root / rgb[cls.i + 1][1]), 0)
        cls.depth = (
            cv2.imread(str(root / depth[mapping[cls.i]][1]), -1).astype(float) / 5000
        )
        cls.K = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.0]])

    def test_empty_previous_person_mask_preserves_original_measured_motion(self):
        cv2.setRNGSeed(1305)
        a = vo.estimate(self.previous, self.current, self.depth, self.K)
        cv2.setRNGSeed(1305)
        b = vo.estimate(
            self.previous,
            self.current,
            self.depth,
            self.K,
            np.zeros_like(self.previous, dtype=bool),
        )
        self.assertTrue(a[1])
        self.assertEqual(a[1:], b[1:])
        np.testing.assert_array_equal(a[0], b[0])

    def test_no_static_reference_features_returns_explicit_lost(self):
        T, ok, count = vo.estimate(
            self.previous,
            self.current,
            self.depth,
            self.K,
            np.ones_like(self.previous, dtype=bool),
        )
        self.assertFalse(ok)
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(T, np.eye(4))

    def test_current_person_regions_cannot_enter_pnp(self):
        T, ok, count = vo.estimate(
            self.previous,
            self.current,
            self.depth,
            self.K,
            current_person=np.ones_like(self.current, dtype=bool),
        )
        self.assertFalse(ok)
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(T, np.eye(4))


if __name__ == "__main__":
    unittest.main()
