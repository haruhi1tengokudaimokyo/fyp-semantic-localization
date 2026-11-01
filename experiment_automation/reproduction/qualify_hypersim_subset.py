"""Aggregate only actually accepted frozen-plan frames; fail on split/image leakage."""

import argparse, collections, hashlib, json
from pathlib import Path
import numpy as np


def main(plan_path, dataset, output, revision="v1"):
    plan = json.loads(Path(plan_path).read_text())["hypersim"]
    root = Path(dataset)
    digest = hashlib.sha256(Path(plan_path).read_bytes()).hexdigest()
    counts = collections.Counter()
    classes = {k: np.zeros(40, np.int64) for k in ["train", "val", "test"]}
    hashes = {}
    catalog = []
    scenes = []
    for selected in plan["frozen_plan"]:
        scene = selected["scene"]
        scene_root = root / scene / "v2" if revision == "v2" else root / scene
        accept = json.loads((scene_root / "acceptance.json").read_text())
        if revision == "v2":
            assert accept["calibration_revision"] == "hypersim_camera_v2"
            assert accept["geometry_round_trip"]["quantization_bound_violations"] == 0
        frames = json.loads((scene_root / "supervision_manifest.json").read_text())
        assert (
            accept["frozen_plan_sha256"] == digest
            and accept["all_selected_members_crc_verified"]
        )
        assert accept["partition"] == selected["split"] and len(frames) == len(
            selected["planned_frames"]
        )
        expected = {(r["camera"], r["frame"]) for r in selected["planned_frames"]}
        assert {(r["camera"], r["frame"]) for r in frames} == expected
        for r in frames:
            assert "T_world_camera" not in r
            key = r["rgb_decoded_sha256"]
            old = hashes.get(key)
            assert (
                old is None or old == selected["split"]
            ), "Exact decoded RGB content crosses held-out split; quarantine, do not silently drop test frames"
            hashes[key] = selected["split"]
            catalog.append(
                {
                    **r,
                    "split": selected["split"],
                    "source_family_proxy": selected["source_family_proxy"],
                }
            )
        counts[selected["split"]] += len(frames)
        classes[selected["split"]] += accept["semantic_pixel_counts_1_to_40"]
        scenes.append(
            dict(
                scene=scene,
                split=selected["split"],
                frames=len(frames),
                geometry_round_trip=accept["geometry_round_trip"],
            )
        )
    assert (
        len(scenes) == plan["planned_scenes"] and len(catalog) == plan["planned_frames"]
    )
    result = dict(
        status="qualified_supervision_subset",
        calibration_revision=revision,
        acquired_frames=len(catalog),
        acquired_scenes=len(scenes),
        splits=dict(counts),
        semantic_pixel_counts_1_to_40={k: v.tolist() for k, v in classes.items()},
        source_family_groups=len(
            {s["source_family_proxy"] for s in plan["frozen_plan"]}
        ),
        physical_building_identity_verified=False,
        exact_decoded_RGB_cross_split_overlap=0,
        full_original_mesh_available=False,
        localization_executed=False,
        frozen_plan_sha256=digest,
        scenes=scenes,
        scope="Frozen custom source-family-disjoint supervision subset, not official-v1 benchmark or verified physical-building generalization. No original full mesh claim.",
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [("manifest.json", catalog), ("acceptance.json", result)]:
        target = output / name
        t = target.with_suffix(".tmp")
        t.write_text(json.dumps(value, indent=2))
        t.replace(target)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["scenes", "semantic_pixel_counts_1_to_40"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("plan")
    p.add_argument("dataset")
    p.add_argument("output")
    p.add_argument("--revision", choices=["v1", "v2"], default="v1")
    a = p.parse_args()
    main(a.plan, a.dataset, a.output, a.revision)
