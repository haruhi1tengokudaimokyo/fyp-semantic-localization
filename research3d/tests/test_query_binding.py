"""Real-cache adapter checks: new kidnap images, same map, no stale H queries."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from knownmap.query_binding import bind_continuous_queries

BASE = Path(os.environ.get("FYP3D_BASE", "/nonexistent"))


@unittest.skipUnless(
    (BASE / "datasets/common_hssd105_kidnap/acceptance.json").is_file(),
    "actual rendered camera teleport cache required",
)
class QueryBindingTests(unittest.TestCase):
    def setUp(self):
        self.dataset = BASE / "datasets/common_hssd105_kidnap"
        self.maproot = BASE / "runs/formal/common/observations/bow"
        self.protocol = json.loads((self.maproot / "map_protocol.json").read_text())

    def test_actual_new_queries_preserve_map_and_strip_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            result = bind_continuous_queries(self.dataset, directory, self.protocol)
            queries = json.loads((Path(directory) / "queries.json").read_text())
            old = json.loads((self.maproot / "queries.json").read_text())
            self.assertEqual(len(queries), 600)
            self.assertTrue(all(set(q) == {"rgb", "K", "frame_id"} for q in queries))
            self.assertTrue(all(q["rgb"] != r["rgb"] for q, r in zip(queries, old)))
            self.assertEqual(result["map_sha256"], self.protocol["map_sha256"])
            self.assertEqual(result["references"], self.protocol["references"])
            self.assertTrue(result["no_query_GT_online"])
            truth = json.loads(
                (Path(directory) / "ground_truth_eval_only.json").read_text()
            )
            self.assertEqual(len(truth), 600)
            self.assertIn("T_world_camera", truth[300])

    def test_reject_cross_scene_map(self):
        protocol = {**self.protocol, "references": [dict(rgb="/different_scene.png")]}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AssertionError, "Cross-scene"):
                bind_continuous_queries(self.dataset, directory, protocol)

    def test_reject_reordered_reference_coordinates(self):
        protocol = {**self.protocol, "references": self.protocol["references"][::-1]}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AssertionError, "reordered"):
                bind_continuous_queries(self.dataset, directory, protocol)


if __name__ == "__main__":
    unittest.main()
