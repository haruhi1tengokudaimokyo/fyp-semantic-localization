"""Validate the experiment protocol and freeze building lineage, without training."""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

from .contracts import digest, sha256_file, write_json, number


def validate_config(c):
    if type(c["schema_version"]) is not int or c["schema_version"] != 1 or c["evaluation"]["test_tuning_allowed"]:
        raise ValueError("unsupported or leaking protocol")
    dev = c["data"]["development_scene_ids"]
    if len(dev) != len(set(dev)) or set(dev) & set(c["data"]["forbidden_for_system_tuning_scene_ids"]):
        raise ValueError("development overlaps previously evaluated test buildings")
    if c["frontends"]["macvo"]["frozen"] is not True:
        raise ValueError("MAC-VO is frozen in this phase")
    if any(x["gt_online"] for x in c["frontends"].values()):
        raise ValueError("query GT forbidden online")
    if c["resources"]["cloud_rental_authorized"] or c["resources"]["automatic_training"]:
        raise ValueError("this setup is not authorization for cloud rental or automatic training")
    s = c["data"]["stereo"]
    if not 0 < s["baseline_m"] < 1 or min(s["width"], s["height"]) < 256:
        raise ValueError("true stereo with native resolution required")
    if c["evaluation"]["absolute_alignment"] != "none" or c["evaluation"]["missing_predictions"] != "count_as_failures":
        raise ValueError("global evaluation must preserve failures and map coordinates")
    defaults = c["replay_defaults"]
    for name, override in c["variants"].items():
        v = {**defaults, **override}
        if type(v["top_k"]) is not int or v["top_k"] not in [1, 3] or type(v["confirmation_frames"]) is not int or v["confirmation_frames"] < 1:
            raise ValueError(f"invalid replay settings: {name}")
        for key in ("max_observation_age_s", "hypothesis_ttl_s", "max_dead_reckoning_s", "conflicts_before_lost"):
            if number(v[key]) <= 0:
                raise ValueError(f"{key} must be positive")
        if v["max_observation_age_s"] > v["max_dead_reckoning_s"]:
            raise ValueError("maximum observation age exceeds permitted dead reckoning")
        if set(v["source_priority"]) != {"ace", "spvloc"} or len(v["source_priority"]) != 2:
            raise ValueError("each source needs one priority")
        if len(set(v["absolute_sources"])) != len(v["absolute_sources"]) or not set(v["absolute_sources"]) <= set(v["source_priority"]):
            raise ValueError("invalid source subset")
        for source in ("ace", "spvloc"):
            for kind in ("tracking_gate", "recovery_gate"):
                gates = v[kind][source]
                if len(gates) != 2 or not all(number(x) > 0 for x in gates) or gates[1] > 180:
                    raise ValueError("gate must be positive metres/degrees with rotation <=180")
        if number(v["quality"]["ace"]["min_pnp_inliers"]) < 0:
            raise ValueError("negative PnP inlier threshold")
        number(v["quality"]["spvloc"]["min_viewport_score"])
    if defaults["tracking_gate"]["ace"] == defaults["tracking_gate"]["spvloc"]:
        raise ValueError("separate source error scales required")
    return c


def freeze_lineage(c, source_root):
    """Exclude every old candidate, not only accepted test houses, from new test."""
    data = Path(c["data"]["source_root"])
    accepted = json.loads(Path(c["data"]["accepted_manifest"]).read_text())
    if accepted["status"] != "DATA_READY":
        raise ValueError("old ProcTHOR data not accepted")
    by_id = {row["scene_id"]: row for row in accepted["houses"]}
    dev = []
    for scene_id in c["data"]["development_scene_ids"]:
        row = by_id[scene_id]
        if row["split"] not in {"train", "val"}:
            raise ValueError("test building selected for development")
        house_path = data / f"scene_{scene_id:05d}" / "procthor_house.json"
        house = json.loads(house_path.read_text())
        geo = {k: house[k] for k in ["rooms", "walls", "doors", "windows", "objects"]}
        if digest(geo) != row["source_geometry_sha256"]:
            raise ValueError("building identity changed")
        dev.append(dict(source_id="procthor", building_id=f"{row['split']}/{row['official_house_index']}",
                        scene_id=scene_id, original_split=row["split"],
                        base_building_geometry_sha256=row["source_geometry_sha256"],
                        house_file=str(house_path), house_file_sha256=sha256_file(house_path),
                        role="system_development"))
    excluded_geometry = {r["source_geometry_sha256"] for r in accepted["houses"]}
    excluded_test = {r["official_house_index"] for r in accepted["houses"] if r["split"] == "test"}
    candidate_path = data / "building_candidates_100_20_20.json"
    if candidate_path.exists():
        for row in json.loads(candidate_path.read_text())["houses"]:
            excluded_geometry.add(row["source_geometry_sha256"])
            if row["split"] == "test":
                excluded_test.add(row["official_house_index"])
    target = c["data"]["new_final_test_buildings"]
    selected = []
    source = Path(source_root) / "test.jsonl.gz"
    with gzip.open(source, "rt") as stream:
        for index, line in enumerate(stream):
            h = json.loads(line)
            if index in excluded_test or len(h["rooms"]) < 2 or not h["doors"] or not h["windows"]:
                continue
            geometry = {k: h[k] for k in ["rooms", "walls", "doors", "windows", "objects"]}
            sha = digest(geometry)
            if sha in excluded_geometry:
                continue
            excluded_geometry.add(sha)
            selected.append(dict(source_id="procthor", building_id=f"test/{index}", official_house_index=index,
                                 original_split="test", base_building_geometry_sha256=sha,
                                 role="sealed_system_test_not_captured"))
            if len(selected) == target:
                break
    if len(selected) != target:
        raise ValueError("not enough independent test houses")
    return dict(schema_version=1, selection_used_localization_results=False,
                development=dev, sealed_system_test=selected,
                original_manifest_sha256=sha256_file(c["data"]["accepted_manifest"]),
                publisher_test_gzip_sha256=sha256_file(source),
                status="LINEAGE_FROZEN_NOT_DATA_READY")


def inspect(c):
    free = shutil.disk_usage("/data").free / 1024**3
    required = c["resources"]["min_data_free_gib"]
    manifest = Path(c["data"]["accepted_manifest"])
    raw_zind = Path("/data/fyp_3d/datasets/zind_original/download_status.json")
    zind = json.loads(raw_zind.read_text()) if raw_zind.exists() else {}
    # Only nonsecret status values: downloaded URL/credential records never exported.
    return dict(config_valid=True, disk_free_gib=free, reserve_gib=required,
                new_data_budget_gib=max(0, min(free - required, c["resources"]["max_new_data_peak_gib"])),
                existing_procthor_manifest_sha256=sha256_file(manifest) if manifest.exists() else None,
                zind_download_status=zind.get("stage", zind.get("status", "UNKNOWN")),
                zind_complete_houses=zind.get("complete_houses"), zind_total_houses=zind.get("total_houses"),
                readiness={"configuration": "VALIDATED", "shared_continuous_stereo": "NOT_YET_ACCEPTED",
                           "same_map_ACE_heads": "NOT_TRAINED", "shared_frontend_observations": "NOT_YET_ACCEPTED",
                           "ZInD_SPVLoc_conversion": "NOT_YET_ACCEPTED", "fusion_effectiveness": "NOT_EVALUATED"},
                no_training_started=True, no_background_services_changed=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["validate", "preflight", "freeze-lineage"])
    p.add_argument("--config", type=Path, default=Path("configs/fusion/experiment.json"))
    p.add_argument("--source-root", type=Path, default=Path("/data/fyp_3d/sources/procthor-10k"))
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    c = validate_config(json.loads(a.config.read_text()))
    if a.output and a.output.exists():
        raise FileExistsError("refuse to overwrite frozen protocol/evidence")
    result = {"status": "CONFIG_VALID", "config_sha256": sha256_file(a.config)}
    if a.command == "freeze-lineage":
        result = freeze_lineage(c, a.source_root)
    elif a.command == "preflight":
        result = inspect(c)
    if a.output:
        write_json(a.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
