"""Official ACE 12-Scenes RGB-only setup with audited ID joins and invalid poses.

Preserve split.txt ordinal boundary BEFORE excluding invalid poses. Do not sort
images and poses independently and hope they correspond. Pseudo-GT is separate.
"""

import argparse, hashlib, json, shutil, zipfile, zlib
from pathlib import Path
import numpy as np
from PIL import Image


def atomic(path, data):
    p = Path(path)
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(data, indent=2))
    t.replace(p)


def prepare(archive, group, output, expected_scenes=None):
    official_scenes = {
        "apt1": ["kitchen", "living"],
        "apt2": ["bed", "kitchen", "living", "luke"],
        "office1": ["gates362", "gates381", "lounge", "manolis"],
        "office2": ["5a", "5b"],
    }
    full_group = expected_scenes is None
    expected_scenes = official_scenes[group] if full_group else expected_scenes
    archive, out = Path(archive), Path(output)
    source = out / "12scenes_source"
    source.mkdir(parents=True, exist_ok=True)
    scientific = []
    with zipfile.ZipFile(archive) as z:
        for m in z.infolist():
            path = Path(m.filename)
            assert not path.is_absolute() and ".." not in path.parts
            if m.is_dir() or not (
                path.name in ["info.txt", "split.txt"]
                or path.name.endswith((".color.jpg", ".pose.txt"))
            ):
                continue
            assert (
                m.filename not in scientific
            ), "Ambiguous duplicate scientific archive member"
            assert not ((m.external_attr >> 16) & 0o170000) == 0o120000
            target = source / path
            target.parent.mkdir(parents=True, exist_ok=True)
            valid = target.is_file() and target.stat().st_size == m.file_size
            if valid:
                valid = (zlib.crc32(target.read_bytes()) & 0xFFFFFFFF) == m.CRC
            if not valid:
                assert shutil.disk_usage(out).free > 30 * 1024**3 + m.file_size
                temp = target.with_suffix(target.suffix + ".tmp")
                with z.open(m) as src, temp.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                assert temp.stat().st_size == m.file_size
                temp.replace(target)
            scientific.append(m.filename)
    actual = {
        Path(n).parts[1]
        for n in scientific
        if Path(n).name == "info.txt" and Path(n).parts[0] == group
    }
    assert actual == set(
        expected_scenes
    ), "Incomplete/extra official scene group; never qualify a partial source ZIP"
    receipts = []
    for info in sorted((source / group).glob("*/info.txt")):
        folder = info.parent
        data = folder / "data"
        assert data.is_dir()
        lines = info.read_text().splitlines()
        height = int(lines[3].split()[2])
        f = lines[7].split()
        focal = (float(f[2]) + float(f[7])) / 2
        split = int((folder / "split.txt").read_text().splitlines()[0].split()[1][8:-1])
        images = sorted(data.glob("*.color.jpg"))
        assert 0 < split < len(images)
        records = []
        excluded = []
        dest = out / ("12scenes_" + group + "_" + folder.name)
        for i, image in enumerate(images):
            stem = image.name.removesuffix(".color.jpg")
            pose = data / (stem + ".pose.txt")
            assert pose.is_file(), ("Missing pose for official RGB frame", image)
            T = np.loadtxt(pose)
            variant = "test" if i < split else "train"
            if T.shape != (4, 4) or not np.isfinite(T).all():
                excluded.append(
                    dict(
                        frame=stem,
                        split=variant,
                        reason="nonfinite/corrupt pose, not localization error",
                    )
                )
                continue
            assert (
                np.allclose(T[3], [0, 0, 0, 1])
                and abs(np.linalg.det(T[:3, :3]) - 1) < 1e-3
            )
            with Image.open(image) as im:
                assert im.height == height
                im.verify()
            for kind in ["rgb", "poses", "calibration"]:
                (dest / variant / kind).mkdir(parents=True, exist_ok=True)
            for kind, original in [("rgb", image), ("poses", pose)]:
                target = dest / variant / kind / original.name
                if target.exists():
                    assert target.resolve() == original.resolve()
                else:
                    target.symlink_to(original.resolve())
            (dest / variant / "calibration" / (stem + ".calibration.txt")).write_text(
                str(focal)
            )
            records.append(
                dict(
                    frame=stem,
                    split=variant,
                    rgb_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                )
            )
        result = dict(
            status="prepared",
            scene=dest.name,
            pose_version="original D-SLAM; not Pseudo-GT",
            official_split_ordinal_boundary=split,
            official_total_rgb=len(images),
            train_frames=sum(r["split"] == "train" for r in records),
            test_frames=sum(r["split"] == "test" for r in records),
            excluded_invalid_pose_frames=excluded,
            frames=records,
            focal_length=focal,
            image_height=height,
            depth="Not extracted, matching official ACE --depth none; original archive retained",
            filtering_policy="Only invalid pose exclusion prescribed by official setup; no performance-based filtering",
            source="https://graphics.stanford.edu/projects/reloc/",
            official_setup="nianticlabs/ace datasets/setup_12scenes.py, --poses original --depth none",
        )
        assert result["train_frames"] > 0 and result["test_frames"] > 0
        atomic(dest / "acceptance.json", result)
        receipts.append({k: v for k, v in result.items() if k != "frames"})
    assert receipts, "No expected official scene directories found"
    result = dict(
        status="prepared",
        group=group,
        scenes=receipts,
        scientific_members_crc_checked=len(scientific),
        pose_version="original",
        complete_RGB_pose_archive_selection=True,
        complete_official_scene_group=full_group,
        expected_scene_names=expected_scenes,
    )
    atomic(out / (group + "_acceptance.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive")
    p.add_argument("group")
    p.add_argument("output")
    a = p.parse_args()
    prepare(a.archive, a.group, a.output)
