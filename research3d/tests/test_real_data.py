"""Run with FYP3D_BASE; checks real data and official preprocessing parity."""

import json, os, sys, unittest
from pathlib import Path
import numpy as np
from knownmap.contracts import Camera, SensorObservation, ReadOnlyMap
from knownmap.ace import load_image

BASE = Path(os.environ.get("FYP3D_BASE", "/nonexistent"))


@unittest.skipUnless(
    (BASE / "datasets/7scenes/7scenes_heads").exists(), "real dataset not installed"
)
class RealData(unittest.TestCase):
    def test_official_preprocessing(self):
        import torch

        sys.path.insert(0, str(BASE / "sources/ace"))
        from dataset import CamLocDataset

        root = BASE / "datasets/7scenes/7scenes_heads/test"
        data = CamLocDataset(root, mode=0)
        cam = Camera(640, 480, [[525, 0, 320], [0, 525, 240], [0, 0, 1]])
        for i in [0, 37, 401, 999]:
            image, *_ = data[i]
            ours = load_image(SensorObservation(str(i), cam, data.rgb_files[i]))
            self.assertTrue(torch.equal(image.float(), ours.to(image.dtype).float()))

    def test_train_test_disjoint_map(self):
        d = json.loads(
            (BASE / "datasets/7scenes/7scenes_heads/acceptance.json").read_text()
        )
        self.assertEqual(d["counts"]["train"]["frames"], 1000)
        self.assertEqual(d["counts"]["test"]["frames"], 1000)
        self.assertFalse(
            set(d["counts"]["train"]["sequences"])
            & set(d["counts"]["test"]["sequences"])
        )
        root = BASE / "runs/heads_geometry"
        m = json.loads((root / "map.json").read_text())
        self.assertTrue(all(ref.startswith("seq-02/") for ref in m["reference_ids"]))
        map = ReadOnlyMap(
            root / "geometry.npz",
            m["geometry_sha256"],
            m["map_id"],
            tuple(m["reference_ids"]),
            root / "references.json",
            m["reference_catalog_sha256"],
        )
        self.assertGreater(len(map.load()["points_world"]), 100000)
        self.assertEqual(len(map.references()), 100)
        self.assertTrue(all(obs.rgb_path.exists() for obs, pose in map.references()))

    def test_independent_mesh_checks(self):
        for name in ["replicacad_empty", "hssd_stage_only"]:
            d = json.loads((BASE / "runs" / name / "mesh_depth_check.json").read_text())
            self.assertTrue(d["passed"])
            self.assertGreater(d["n"], 1000)
            self.assertLess(d["p95_abs_depth_m"], 0.001)


if __name__ == "__main__":
    unittest.main()
