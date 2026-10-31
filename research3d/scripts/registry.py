"""Build machine-readable dataset registry from actual acquisition/acceptance files."""

import argparse, json
from pathlib import Path


def main(base, output):
    b = Path(base)
    rows = []

    def add(name, official, path, sensors, license, tasks, caveat, geometry, semantics):
        p = b / path if path else None
        acq = (
            json.loads((p / "acquisition.json").read_text())
            if p and (p / "acquisition.json").exists()
            else None
        )
        integ = (
            json.loads((p / "integrity.json").read_text())
            if p and (p / "integrity.json").exists()
            else None
        )
        row = dict(
            dataset=name,
            official=official,
            license=license,
            downloaded=bool(p and p.exists()),
            local_path=str(p) if p else None,
            actual_download_urls=[x["url"] for x in acq["files"]] if acq else [],
            revision=acq["revision"] if acq else None,
            bytes_downloaded=sum(x["bytes"] for x in acq["files"]) if acq else None,
            extracted_bytes=(
                sum(
                    x.stat().st_size
                    for x in p.rglob("*")
                    if x.is_file() and not x.is_symlink()
                )
                if p
                else None
            ),
            integrity=integ,
            sensors=sensors,
            geometry=geometry,
            semantics=semantics,
            dynamic_people="not verified",
            allowed_experiments=tasks,
            caveat=caveat,
        )
        rows.append(row)
        return row

    r = add(
        "ReplicaCAD interactive",
        "https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset",
        "datasets/replicacad",
        dict(
            RGB="rendered and verified",
            Depth="rendered and verified",
            Pose="actual sensor transform",
            Calibration="synthetic pinhole",
            IMU=False,
            Wheel=False,
        ),
        "CC-BY-4.0",
        [
            "fixed 3D rendering",
            "coordinate validation",
            "synthetic known-map localization",
        ],
        "Apartment rearrangements do not equal independent buildings. Official config also lists some absent optional variation navmeshes; tested apt_0/empty_stage assets load.",
        "original GLB stages and objects",
        "object IDs available, structural pixels mostly unknown",
    )
    r = add(
        "HSSD-Habitat",
        "https://huggingface.co/datasets/hssd/hssd-hab",
        "datasets/hssd",
        dict(
            RGB="3 scenes x 24 views verified",
            Depth="verified, independent stage ray test",
            Pose="actual sensor transform",
            Calibration="synthetic pinhole",
            IMU=False,
            Wheel=False,
        ),
        "CC-BY-NC-4.0",
        [
            "3D rendering",
            "audited object-class supervision",
            "scene-wise data preparation",
        ],
        "Missing architectural class coverage: see hssd_semantic_audit.json. Distinct scene IDs do not establish independent physical buildings.",
        "original GLB stages and 117 rigid object templates",
        "native configured class IDs; repeated instances not distinguished; openings may lack semantic_id",
    )
    r = add(
        "7-Scenes Heads",
        "https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/",
        "datasets/7scenes",
        dict(
            RGB=True,
            Depth="raw unregistered Kinect mm",
            Pose="KinectFusion offline GT; RGB pose adapted per ACE",
            Calibration="ACE defaults/extrinsic, not exact per-device",
            IMU=False,
            Wheel=False,
        ),
        "Microsoft 7-Scenes non-commercial license",
        ["single-scene RGB relocalization", "train-only RGB-D geometry"],
        "Only official train/test sequences used. Single scene supports no cross-building claim. RGB depth unregistered. Two unused Thumbs.db CRC errors are explicitly excluded.",
        "train-depth-derived point cloud, no test poses used",
        "not provided",
    )
    accept = b / "datasets/7scenes/7scenes_heads/acceptance.json"
    if accept.exists():
        r["acceptance"] = json.loads(accept.read_text())
    ar = b / "downloads/heads_range.zip.json"
    if ar.exists():
        d = json.loads(ar.read_text())
        r.update(
            actual_download_urls=[d["url"]],
            bytes_downloaded=d["bytes"],
            archive_sha256=d["sha256"],
        )
    for name, url, sensors, tasks, caveat in [
        (
            "OpenLORIS-Scene",
            "https://huggingface.co/datasets/shixuesong/openloris-scene",
            dict(
                Camera="requires bag topic inspection",
                IMU="requires bag inspection",
                Wheel="requires bag inspection",
            ),
            ["VO/VIO/fusion after sensor audit"],
            "Not downloaded. No verified prior map; LiDAR topics forbidden online. GT provenance must be checked.",
        ),
        (
            "VIODE",
            "https://github.com/kminoda/VIODE",
            dict(
                Camera="not locally verified", IMU="not locally verified", Wheel=False
            ),
            ["dynamic VIO after verification"],
            "Not downloaded; no verified common 3D prior map.",
        ),
        (
            "TUM RGB-D",
            "https://cvg.cit.tum.de/data/datasets/rgbd-dataset",
            dict(
                RGB="not locally verified",
                Depth="not locally verified",
                Pose="not locally verified",
                IMU=False,
                Wheel=False,
            ),
            ["dynamic RGB-D tracking after verification"],
            "Not downloaded; VO evaluation does not imply global localization.",
        ),
        (
            "InCrowd-VI",
            "https://vault.cloudlab.zhaw.ch/vault-data/incrowd-vi",
            dict(
                Camera="not locally verified", IMU="not locally verified", Wheel=False
            ),
            ["future dynamic VIO"],
            "Not downloaded; later stage, no verified map.",
        ),
    ]:
        add(
            name,
            url,
            None,
            sensors,
            "not locally audited",
            tasks,
            caveat,
            "not locally verified",
            "not locally verified",
        )
    loris = next(x for x in rows if x["dataset"] == "OpenLORIS-Scene")
    loris["license"] = "CC BY-ND 4.0, per official mirrored README"
    for key, filename in [
        ("remote_header_audit", "openloris_remote_topics.json"),
        ("remote_tar_inventory", "openloris_office_tar_members.json"),
    ]:
        file = b / "runs" / filename
        if file.exists():
            loris[key] = json.loads(file.read_text())
    loris["gt_source"] = (
        "Official project: office uses OptiTrack, other scenes use offline LiDAR SLAM. GT is evaluation-only."
    )
    loris["metadata_only_downloaded"] = "remote_header_audit" in loris
    Path(output).write_text(
        json.dumps(
            {
                "datasets": rows,
                "download_policy": "/data checked before downloads; no gated dataset bypasses; no unprovided sensors invented",
            },
            indent=2,
        )
    )
    print([(x["dataset"], x["downloaded"], x["bytes_downloaded"]) for x in rows])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("output")
    a = p.parse_args()
    main(a.base, a.output)
