"""Inspect actually accepted v3 catalogs; report quality and leakage without training."""

import collections, csv, hashlib, io, json
from pathlib import Path

BASE = Path("/data/fyp_3d")


def build(tasks):
    rows = []
    hashes = {}
    overlaps = []
    class_counts = collections.Counter()
    for task in tasks:
        tid = task["task_id"]
        if not tid.startswith("R.data.hypersim.subset.v3."):
            continue
        scene = tid.split(".")[-1]
        path = BASE / "datasets/hypersim_formal_subset" / scene / "v3/acceptance.json"
        row = dict(
            scene=scene,
            status=task["status"],
            attempt=task["attempt"],
            accepted_frames=0,
            partition=None,
            calibration="v3",
            failure_evidence=None,
        )
        if task["status"] == "COMPLETED":
            a = json.loads(path.read_text())
            frames = json.loads((path.parent / "supervision_manifest.json").read_text())
            assert a["calibration_revision"] == "hypersim_camera_v3" and a[
                "acquired_frames"
            ] == len(frames)
            probes = a["geometry_round_trip"]["probes"]
            valid = [r["valid_depth_fraction"] for r in frames]
            row.update(
                accepted_frames=len(frames),
                partition=a["partition"],
                cameras=len({r["camera"] for r in frames}),
                width=a["width"],
                height=a["height"],
                geometry_probed_pixels=a["geometry_round_trip"]["pixels"],
                geometry_violations=a["geometry_round_trip"][
                    "quantization_bound_violations"
                ],
                max_probe_p95_world_error_m=max(p["p95_m"] for p in probes),
                max_probe_normalized_error=a["geometry_round_trip"][
                    "normalized_error_max"
                ],
                min_valid_depth_fraction=min(valid),
                mean_valid_depth_fraction=sum(valid) / len(valid),
                unknown_semantic_pixels=a["unknown_label_pixels"],
                source_orientation_gram_error_max=a[
                    "source_orientation_gram_error_max"
                ],
                fx_min=min(r["K"][0][0] for r in frames),
                fx_max=max(r["K"][0][0] for r in frames),
                fy_min=min(r["K"][1][1] for r in frames),
                fy_max=max(r["K"][1][1] for r in frames),
                crc_verified_bytes=a["downloaded_scientific_bytes"],
                acceptance_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            class_counts.update(
                {
                    str(i + 1): v
                    for i, v in enumerate(a["semantic_pixel_counts_1_to_40"])
                }
            )
            for f in frames:
                h = f["rgb_decoded_sha256"]
                old = hashes.get(h)
                identity = dict(
                    scene=scene,
                    camera=f["camera"],
                    frame=f["frame"],
                    partition=a["partition"],
                )
                if old and old["partition"] != a["partition"]:
                    overlaps.append(dict(first=old, second=identity, decoded_sha256=h))
                hashes[h] = identity
        elif task["status"] == "FAILED":
            row["failure_evidence"] = (
                f'/data/fyp_3d/automation/artifacts/{tid}.attempt{task["attempt"]}.failure.json'
            )
        rows.append(row)
    result = dict(
        dataset="Hypersim custom frozen source-family subset",
        calibration="v3",
        accepted_scenes=sum(r["status"] == "COMPLETED" for r in rows),
        accepted_frames=sum(r["accepted_frames"] for r in rows),
        states=dict(collections.Counter(r["status"] for r in rows)),
        exact_decoded_RGB_cross_split_overlaps=overlaps,
        class_pixel_counts_1_to_40=dict(class_counts),
        rows=rows,
        physical_building_identity_verified=False,
        used_for_new_training=False,
        scope="Live data-quality snapshot; not a qualified training split until cleaned_qualification.v3 accepts all prospective scene decisions. No localization outcomes used.",
    )
    stream = io.StringIO()
    keys = list(dict.fromkeys(k for r in rows for k in r))
    w = csv.DictWriter(stream, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
    return result, stream.getvalue()
