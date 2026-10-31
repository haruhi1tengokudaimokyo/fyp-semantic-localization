"""Single-scene equivalent of ACE setup_7scenes.py (calibrated poses).
Keeps original depth; NEVER treats it as registered RGB depth.
"""

import argparse, hashlib, json, sys, zipfile
from pathlib import Path
import numpy as np


def prepare(archive, root, ace):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        z.extractall(root / "7scenes_source")
    src = root / "7scenes_source" / "heads"
    excluded = []
    verified = 0
    for p in sorted(src.glob("*.zip")):
        with zipfile.ZipFile(p) as z:
            for member in z.infolist():
                if member.is_dir():
                    continue
                if not member.filename.endswith(
                    (".color.png", ".depth.png", ".pose.txt")
                ):
                    try:
                        z.read(member)
                    except zipfile.BadZipFile:
                        excluded.append(
                            {
                                "archive": p.name,
                                "member": member.filename,
                                "reason": "CRC mismatch in unused non-scientific file",
                            }
                        )
                    continue
                z.extract(member, src)
                verified += 1
    sys.path.insert(0, str(Path(ace) / "datasets"))
    from setup_7scenes import d_to_rgb, focal_length

    out = root / "7scenes_heads"
    counts = {}
    for split, name in [("TrainSplit.txt", "train"), ("TestSplit.txt", "test")]:
        seqs = [
            "seq-" + s.strip()[8:].zfill(2)
            for s in (src / split).read_text().splitlines()
            if s.strip()
        ]
        n = 0
        for sub in ["rgb", "poses", "calibration"]:
            (out / name / sub).mkdir(parents=True, exist_ok=True)
        for seq in seqs:
            for img in sorted((src / seq).glob("*.color.png")):
                base = img.name.removesuffix(".color.png")
                prefix = seq + "-" + base
                link = out / name / "rgb" / (prefix + ".color.png")
                if not link.exists():
                    link.symlink_to(img.resolve())
                pose = np.loadtxt(src / seq / (base + ".pose.txt")) @ np.linalg.inv(
                    d_to_rgb
                )
                np.savetxt(out / name / "poses" / (prefix + ".pose.txt"), pose)
                (out / name / "calibration" / (prefix + ".calibration.txt")).write_text(
                    str(focal_length) + "\n"
                )
                n += 1
        counts[name] = {"sequences": seqs, "frames": n}
    assert not (set(counts["train"]["sequences"]) & set(counts["test"]["sequences"]))
    h = hashlib.sha256()
    with open(archive, "rb") as stream:
        while data := stream.read(1024**2):
            h.update(data)
    report = {
        "archive_sha256": h.hexdigest(),
        "outer_zip_crc_verified": True,
        "scientific_members_crc_verified": verified,
        "excluded_unused_crc_failures": excluded,
        "counts": counts,
        "pose": "T_world_rgb = T_world_depth @ inv(T_rgb_depth)",
        "rgb_K": [[525, 0, 320], [0, 525, 240], [0, 0, 1]],
        "depth_K": [[585, 0, 320], [0, 585, 240], [0, 0, 1]],
        "depth_unit": "mm, 65535 invalid",
        "calibration_caveat": "default intrinsics and external Kinect extrinsic per official ACE; not exact per-device calibration",
    }
    (out / "acceptance.json").write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive")
    p.add_argument("root")
    p.add_argument("--ace", required=True)
    a = p.parse_args()
    prepare(a.archive, a.root, a.ace)
