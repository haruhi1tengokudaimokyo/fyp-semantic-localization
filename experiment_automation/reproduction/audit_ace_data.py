"""Exact official 7-Scenes split, frame pairing and RGB/depth-pose convention audit."""

import argparse, ast, hashlib, json, re
from pathlib import Path
import numpy as np
from PIL import Image

D = np.array(
    [
        [
            0.99996518012567637,
            0.0026765126468950343,
            -0.0079041012313000904,
            -0.025558943178152542,
        ],
        [
            -0.00274093112813167,
            0.99996302803027592,
            -0.0081504520778013286,
            0.00010109636268061706,
        ],
        [
            0.0078819942130445332,
            0.0081718328771890631,
            0.99993554558014031,
            0.0020318321729487039,
        ],
        [0, 0, 0, 1],
    ]
)


def audit(root, output):
    setup = Path("/data/fyp_3d/sources/ace/datasets/setup_7scenes.py")
    assignment = next(
        n
        for n in ast.parse(setup.read_text()).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "d_to_rgb" for t in n.targets)
    )
    official_D = np.array(ast.literal_eval(assignment.value.args[0]), float)
    assert np.allclose(D, official_D, atol=1e-15, rtol=0)
    root = Path(root)
    scenes = []
    for scene in [
        "chess",
        "fire",
        "heads",
        "office",
        "pumpkin",
        "redkitchen",
        "stairs",
    ]:
        source = root / "7scenes_source" / scene
        data = root / ("7scenes_" + scene)
        result = dict(
            scene=scene,
            pose_version="original D-SLAM calibrated to RGB; not Pseudo-GT",
            splits={},
        )
        identities = {}
        for split, filename in [("train", "TrainSplit.txt"), ("test", "TestSplit.txt")]:
            sequences = [
                "seq-" + line.strip()[8:].zfill(2)
                for line in (source / filename).read_text().splitlines()
                if line.strip()
            ]
            expected = {
                seq + "-" + p.name
                for seq in sequences
                for p in (source / seq).glob("*.color.png")
            }
            rgb = sorted((data / split / "rgb").glob("*.color.png"))
            assert {p.name for p in rgb} == expected, (
                scene,
                split,
                "official split differs",
            )
            hashes = []
            for p in rgb:
                stem = p.name.removesuffix(".color.png")
                pose = data / split / "poses" / (stem + ".pose.txt")
                calib = data / split / "calibration" / (stem + ".calibration.txt")
                seq = p.name[:6]
                original = (
                    source / seq / (p.name[7:].replace(".color.png", ".pose.txt"))
                )
                T = np.loadtxt(pose)
                T0 = np.loadtxt(original)
                assert T.shape == (4, 4) and np.isfinite(T).all()
                assert np.allclose(T, T0 @ np.linalg.inv(official_D), atol=1e-6), (
                    scene,
                    stem,
                    "wrong pose convention",
                )
                assert (
                    np.allclose(T[3], [0, 0, 0, 1])
                    and abs(np.linalg.det(T[:3, :3]) - 1) < 1e-3
                )
                assert float(np.loadtxt(calib)) == 525
                with Image.open(p) as im:
                    assert im.size == (640, 480)
                    im.verify()
                hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
            identities[split] = set(hashes)
            result["splits"][split] = dict(
                frames=len(rgb),
                sequences=sequences,
                ordered_rgb_sha256=hashlib.sha256("".join(hashes).encode()).hexdigest(),
            )
        # Report exact repeated imagery; never remove test images to improve scores.
        result["train_test_exact_rgb_overlap"] = len(
            identities["train"] & identities["test"]
        )
        result["split_sequence_overlap"] = list(
            set(result["splits"]["train"]["sequences"])
            & set(result["splits"]["test"]["sequences"])
        )
        assert not result["split_sequence_overlap"]
        scenes.append(result)
        print("AUDITED", scene, flush=True)
    result = dict(
        status="audited",
        scenes=scenes,
        all_frames_decoded_or_verified=True,
        calibration_525_verified=True,
        original_calibrated_pose_verified=True,
        test_frames_removed=0,
        cleaning_policy="Record any corruption/overlap; fail on mispairing. Preserve official test set; no test-error based filtering.",
        official_setup_sha256=hashlib.sha256(setup.read_bytes()).hexdigest(),
    )
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(result, indent=2))
    t.replace(p)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("output")
    a = p.parse_args()
    audit(a.root, a.output)
