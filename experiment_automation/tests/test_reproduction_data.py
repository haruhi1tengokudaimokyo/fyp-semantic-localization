"""Isolated 12-Scenes split/corruption/ID safety fixtures; never scientific results."""

import importlib.util, io, json, tempfile, unittest, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "reproduction/prepare_12scenes.py"
spec = importlib.util.spec_from_file_location("prepare12", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class SplitTests(unittest.TestCase):
    def archive(self, root, missing=False, traversal=False):
        p = root / "fixture.zip"
        image = io.BytesIO()
        Image.new("RGB", (640, 480), (80, 120, 150)).save(image, format="JPEG")
        with zipfile.ZipFile(p, "w") as z:
            if traversal:
                z.writestr("../info.txt", "invalid")
            lines = ["a"] * 8
            lines[3] = "height = 480"
            lines[7] = "a b 525 d e f g 527"
            z.writestr("apt1/kitchen/info.txt", "\n".join(lines))
            # Match upstream split.txt parser: its second token contains frame id.
            z.writestr("apt1/kitchen/split.txt", "test frame-000002)\n")
            for i in range(3):
                z.writestr(
                    f"apt1/kitchen/data/frame-{i:06d}.color.jpg", image.getvalue()
                )
                if missing and i == 1:
                    continue
                T = np.eye(4)
                T[0, 3] = i
                if i == 0:
                    T[0, 0] = np.inf
                text = io.StringIO()
                np.savetxt(text, T)
                z.writestr(f"apt1/kitchen/data/frame-{i:06d}.pose.txt", text.getvalue())
        return p

    def test_invalid_test_pose_does_not_move_train_frame_into_test(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = self.archive(root)
            with patch.object(
                prepare.shutil,
                "disk_usage",
                return_value=type("Space", (), {"free": 100 * 1024**3})(),
            ):
                prepare.prepare(p, "apt1", root / "out", expected_scenes=["kitchen"])
            d = json.loads(
                (root / "out/12scenes_apt1_kitchen/acceptance.json").read_text()
            )
            self.assertEqual(d["test_frames"], 1)
            self.assertEqual(d["train_frames"], 1)
            self.assertEqual(d["frames"][0]["frame"], "frame-000001")
            self.assertEqual(d["frames"][1]["split"], "train")
            self.assertEqual(d["focal_length"], 526)
            self.assertEqual(d["excluded_invalid_pose_frames"][0]["split"], "test")

    def test_missing_pose_fails_instead_of_positional_mispairing(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = self.archive(root, missing=True)
            with patch.object(
                prepare.shutil,
                "disk_usage",
                return_value=type("Space", (), {"free": 100 * 1024**3})(),
            ):
                with self.assertRaises(AssertionError):
                    prepare.prepare(
                        p, "apt1", root / "out", expected_scenes=["kitchen"]
                    )

    def test_partial_official_group_cannot_be_counted_as_complete(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = self.archive(root)
            with patch.object(
                prepare.shutil,
                "disk_usage",
                return_value=type("Space", (), {"free": 100 * 1024**3})(),
            ):
                with self.assertRaisesRegex(
                    AssertionError, "Incomplete/extra official scene group"
                ):
                    prepare.prepare(p, "apt1", root / "out")

    def test_unsafe_archive_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = self.archive(root, traversal=True)
            with self.assertRaises(AssertionError):
                prepare.prepare(p, "apt1", root / "out", expected_scenes=["kitchen"])


if __name__ == "__main__":
    unittest.main()
