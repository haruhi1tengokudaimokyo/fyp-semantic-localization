"""Contract tests use synthetic filesystem fixtures, never scientific results."""
import copy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from research3d.fusion.training import (
    HierarchicalSchedule, NotReady, View, plan, source_manifest, validate_config, validation_macro,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs/fusion/training_budget.json"


class EqualBudgetTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text())

    def test_three_arms_identical_updates_init_and_attempted_budget(self):
        result = plan(self.config, 123)
        self.assertEqual(result["optimizer_updates"], 8575)
        self.assertEqual(result["attempted_queries"], 343000)
        self.assertEqual(result["arms"]["P+Z"]["expected_attempted_queries_per_source"], 171500)
        self.assertEqual(result["validation_sources"], ["procthor", "zind"])
        self.assertEqual(result["bn_policy"], "frozen_running_statistics_trainable_affine")

    def test_exact_source_quota_each_optimizer_update(self):
        views = [View("p", "a", 0, 0, 0), View("z", "b", 1, 0, 0)]
        sampler = HierarchicalSchedule(views, ["p", "z"], 40, 31, 123)
        samples = list(sampler.generate())
        self.assertEqual(len(samples), 31 * 40)
        for index in range(0, len(samples), 40):
            self.assertEqual(Counter(v.source_id for v in samples[index:index+40]), {"p": 20, "z": 20})
        self.assertEqual(samples, list(sampler.generate()))
        self.assertNotEqual(samples, list(sampler.generate(1)))

    def test_building_is_not_weighted_by_its_number_of_views(self):
        views = [View("p", "small", 0, 0, 0)] + [View("p", "large", 1, i, i+1) for i in range(100)]
        result = list(HierarchicalSchedule(views, ["p"], 40, 500, 12).generate())
        fraction = sum(v.building_id == "small" for v in result) / len(result)
        self.assertTrue(.48 < fraction < .52, fraction)

    def test_derived_perspective_duplicates_do_not_reweight_panorama(self):
        views = [View("p", "a", 0, 0, i) for i in range(100)] + [View("p", "a", 0, 1, 101)]
        result = list(HierarchicalSchedule(views, ["p"], 40, 500, 17).generate())
        self.assertTrue(.48 < sum(v.room_id == 0 for v in result) / len(result) < .52)

    def test_unsupported_precision_and_changed_bn_fail_closed(self):
        for name, value in (("precision", "bf16-mixed"), ("bn_policy", "train_native_batch40"),
                            ("optimizer_updates", 8576), ("num_workers", 4)):
            config = copy.deepcopy(self.config)
            config[name] = value
            with self.assertRaises(NotReady):
                validate_config(config)

    def test_empty_source_and_nonintegral_quota_are_rejected(self):
        view = View("p", "a", 0, 0, 0)
        with self.assertRaises(NotReady):
            HierarchicalSchedule([view], ["p", "z"], 40, 1, 0)
        with self.assertRaises(NotReady):
            HierarchicalSchedule([view], ["p", "z"], 3, 1, 0)

    def test_validation_macro_does_not_favour_large_buildings_or_sources(self):
        value, sources = validation_macro({("p", "one"): [2.] * 100,
                                           ("p", "two"): [6.], ("z", "three"): [10.]}, ["p", "z"])
        self.assertEqual(sources, {"p": 4., "z": 10.})
        self.assertEqual(value, 7.)
        with self.assertRaises(NotReady):
            validation_macro({("p", "one"): [1.]}, ["p", "z"])
        with self.assertRaises(NotReady):
            validation_macro({("p", "one"): [float("nan")]}, ["p"])


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = {"root": str(self.root), "manifest": str(self.root / "manifest.json"),
                       "prepared_cache": "pickle"}

    def write(self, path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))

    def procthor(self):
        rows = []
        for sid, split in enumerate(("train", "val", "test")):
            folder = self.root / f"scene_{sid:05d}"
            self.write(folder / "qualification.json", {"status": "DATA_READY_PILOT"})
            self.write(folder / "panorama_qualification.json", {"status": "PANORAMA_READY"})
            self.write(folder / "annotation_3d.json", {"fixture": "not scientific input"})
            (folder / "2D_rendering").mkdir()
            self.write(self.root / "pickle" / f"temp_{sid:05d}", {"fixture": True})
            rows.append(dict(scene_id=sid, split=split, official_house_index=1,
                             source_geometry_sha256=f"fixture-hash-{sid}"))
        self.write(Path(self.source["manifest"]), {"status": "DATA_READY", "houses": rows})
        return rows

    def test_verified_procthor_source_preserves_official_split_local_indices(self):
        self.procthor()
        rows = source_manifest("procthor", self.source)
        self.assertEqual([r["building_id"] for r in rows], ["train:1", "val:1", "test:1"])

    def test_same_geometry_in_different_splits_rejected(self):
        rows = self.procthor()
        rows[1]["source_geometry_sha256"] = rows[0]["source_geometry_sha256"]
        self.write(Path(self.source["manifest"]), {"status": "DATA_READY", "houses": rows})
        with self.assertRaisesRegex(NotReady, "leakage"):
            source_manifest("procthor", self.source)

    def test_declared_building_group_cannot_cross_splits(self):
        rows = self.procthor()
        rows[0]["building_group_id"] = rows[1]["building_group_id"] = "same-base-cad"
        self.write(Path(self.source["manifest"]), {"status": "DATA_READY", "houses": rows})
        with self.assertRaisesRegex(NotReady, "leakage"):
            source_manifest("procthor", self.source)

    def test_panorama_or_cache_missing_rejects_instead_of_writing_data(self):
        self.procthor()
        (self.root / "pickle/temp_00000").unlink()
        with self.assertRaisesRegex(NotReady, "cache missing"):
            source_manifest("procthor", self.source)
        self.assertFalse((self.root / "pickle/temp_00000").exists())

    def test_raw_zind_download_does_not_become_data_ready(self):
        self.write(self.root / "download_status.json", {"stage": "COMPLETE", "complete_houses": 1575})
        with self.assertRaisesRegex(NotReady, "qualified conversion manifest"):
            source_manifest("zind", self.source)

    def test_zind_conversion_without_projection_evidence_is_rejected(self):
        self.source["official_partition"] = str(self.root / "partition.json")
        self.write(Path(self.source["official_partition"]), {"train": ["0000"], "val": ["0001"], "test": ["0002"]})
        self.write(Path(self.source["manifest"]), {
            "status": "DATA_READY", "converter": "spvloc.tools.zind_to_s3d", "offline_only_gt": True,
            "scenes": [{"scene_id": 0, "original_home_id": "0000", "split": "train", "checks": {}}]})
        with self.assertRaisesRegex(NotReady, "QA is incomplete"):
            source_manifest("zind", self.source)

    def test_zind_floor_cannot_be_relabelled_into_another_home_split(self):
        self.source["official_partition"] = str(self.root / "partition.json")
        self.write(Path(self.source["official_partition"]), {"train": ["0000"], "test": ["0001"]})
        self.write(Path(self.source["manifest"]), {
            "status": "DATA_READY", "converter": "spvloc.tools.zind_to_s3d", "offline_only_gt": True,
            "scenes": [{"scene_id": 0, "original_home_id": "0000", "split": "test", "checks": {}}]})
        with self.assertRaisesRegex(NotReady, "home/floor split mismatch"):
            source_manifest("zind", self.source)


if __name__ == "__main__":
    unittest.main()
