"""Real publisher metadata audit and a frozen prospective subset, never acquired-frame counts."""

import argparse, collections, csv, hashlib, json, subprocess
from pathlib import Path


def main(output):
    base = Path("/data/fyp_3d/sources")
    source = base / "hypersim"
    metadata = source / "evermotion_dataset/analysis/metadata_images_split_scene_v1.csv"
    rows = list(csv.DictReader(metadata.open()))
    released = [r for r in rows if r["included_in_public_release"] == "True"]
    by_scene = collections.defaultdict(list)
    for r in released:
        by_scene[r["scene_name"]].append(r)
    families = collections.defaultdict(list)
    for scene, frames in by_scene.items():
        families["_".join(scene.split("_")[:2])].append((len(frames), scene))
    selected = []
    # One scene per conservative source-collection family; no claim that the family
    # ID establishes physical building identity. Preserve original split as metadata.
    groups = sorted(
        families, key=lambda g: hashlib.sha256(("1305:" + g).encode()).hexdigest()
    )
    for i, group in enumerate(groups):
        _, scene = max(families[group])
        frames = sorted(
            by_scene[scene], key=lambda r: (r["camera_name"], int(r["frame_id"]))
        )
        take = min(256, len(frames))
        indices = [j * len(frames) // take for j in range(take)]
        split = (
            "train"
            if i < int(0.7 * len(groups))
            else "val" if i < int(0.85 * len(groups)) else "test"
        )
        selected.append(
            dict(
                source_family_proxy=group,
                scene=scene,
                split=split,
                planned_frames=[
                    dict(
                        camera=r["camera_name"],
                        frame=int(r["frame_id"]),
                        official_v1_partition=r["split_partition_name"],
                    )
                    for r in [frames[j] for j in indices]
                ],
            )
        )
    paths = {
        "Hypersim": source,
        "ADE20K_SegFormer": base / "SegFormer",
        "HPointLoc": base / "HPointLoc",
    }
    result = dict(
        status="metadata_audited",
        source_commits={
            name: subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
            for name, path in paths.items()
        },
        hypersim=dict(
            public_metadata_frames=len(released),
            public_metadata_scenes=len(by_scene),
            excluded_reasons=dict(
                collections.Counter(
                    r["exclude_reason"]
                    for r in rows
                    if r["included_in_public_release"] != "True"
                )
            ),
            frozen_plan=selected,
            planned_scenes=len(selected),
            planned_frames=sum(len(s["planned_frames"]) for s in selected),
            acquired_frames=0,
            metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
            split_policy="Custom source-family-disjoint prospective split, NOT official v1 evaluation; original partition recorded per frame",
            physical_building_identity_verified=False,
            asset_reuse="Official README explicitly notes cross-scene asset reuse; rendered-content/geometry overlap checks remain required",
            depth="Euclidean optical-centre distance in meters; convert using actual M_cam_from_uv and half-pixel centres; poses in asset units require metadata_scene meters_per_asset_unit",
            full_original_mesh_acquired=False,
            raw_modalities_required=[
                "color",
                "semantic",
                "semantic_instance",
                "depth_meters",
                "camera_keyframe_frame_indices",
                "camera_keyframe_positions",
                "camera_keyframe_orientations",
                "metadata_scene",
            ],
        ),
        ade20k=dict(
            acquired_frames=0,
            official_train_val_split_required=True,
            reduce_zero_label=True,
            ignore_index=255,
            raw_label_domain="0=void,1..150=classes; convert to255,0..149 without uint8 underflow",
            original_protocol="SegFormer B0 official 512x512,160k iterations,AdamW lr6e-5,poly schedule,warmup1500,150classes; HSSD40epoch experiments are independent baselines",
        ),
        hpointloc=dict(
            acquired_frames=0,
            derived_from="Matterport3D, according to official README; repo MIT does not establish source-data authorization",
            formats="HDF5 rgb/rgb_base,depth/depth_base,gps/gps_base,quat/quat_base,semantic_base",
            online_input_policy="gps/quaternion arrays are evaluation poses, not an online GPS sensor; never expose query GT to estimators",
            adapter_issues=[
                "upstream extractor references undefined w,h,fx,fy,cx,cy",
                "np.save appends .npy to a .png target; must verify generated index filenames",
                "scene graph instances and meshes not established by image HDF5 alone; not automatically SG2Loc compatible",
            ],
        ),
        scope="Source/metadata and prospective plan only; no synthetic counts of downloaded data, no new segmentation training launched",
    )
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(result, indent=2))
    t.replace(p)
    print(
        json.dumps(
            {k: v for k, v in result["hypersim"].items() if k != "frozen_plan"},
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output")
    main(p.parse_args().output)
