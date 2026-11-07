"""Export and train a map-bound ACE head from *reference* captures only.

This is an engineering pilot (1M feature buffer, 16 official epochs), not the
8M/7-Scenes benchmark. No query pose file is accepted or loaded by this module.
The official model, losses and schedule are run by train_ace_local.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

from research3d.knownmap.contracts import se3


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_image(root, value):
    root = Path(root).resolve(strict=True)
    p = Path(value)
    p = (root / p).resolve(strict=True) if not p.is_absolute() else p.resolve(strict=True)
    if not p.is_relative_to(root) or p.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("RGB must be an image inside the capture root")
    return p


def camera_checked(value):
    width, height = int(value["width"]), int(value["height"])
    K = np.asarray(value["K"], float)
    if K.shape == (9,):
        K = K.reshape(3, 3)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("invalid camera intrinsics")
    if width <= 0 or height != 480 or width > 4096:
        raise ValueError("ACE pilot uses native image height 480; explicit resampling requires a new protocol")
    expected = np.array([[K[0, 0], 0, width / 2], [0, K[0, 0], height / 2], [0, 0, 1]])
    if K[0, 0] <= 0 or not np.allclose(K, expected, rtol=0, atol=1e-4):
        raise ValueError("official scalar-focal ACE dataset needs centered, square-pixel intrinsics")
    return dict(width=width, height=height, K=K.tolist())


def check_query_online(row):
    forbidden = {"pose", "gt", "ground_truth", "T_map_camera", "T_world_camera", "T_map_camera_optical", "depth", "segmentation"}
    if forbidden.intersection(row):
        raise ValueError("query online manifest includes supervision")


def export_scene(dataset, output, min_reference_frames=100):
    dataset, output = Path(dataset).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite existing ACE export")
    if output.is_relative_to(dataset):
        raise ValueError("training export must be outside capture root")
    map_path = dataset / "map_manifest.json"
    manifest = json.loads(map_path.read_text())
    map_id, map_hash = manifest["map_id"], manifest["map_sha256"]
    if not isinstance(map_id, str) or not map_id or len(map_hash) != 64:
        raise ValueError("map identity/hash missing")
    int(map_hash, 16)
    if manifest.get("world_path"):
        world = (dataset / manifest["world_path"]).resolve(strict=True)
        if sha256(world) != map_hash:
            raise ValueError("world/map SHA256 differs from manifest")
    refs_path, queries_path = dataset / "reference/frames.jsonl", dataset / "query/frames.jsonl"
    refs, queries = read_jsonl(refs_path), read_jsonl(queries_path)
    if len(refs) < min_reference_frames or not queries:
        raise ValueError("not enough reference/query frames for declared pilot")
    ids, paths, image_hashes, validated = set(), set(), set(), []
    camera = None
    for row in refs:
        if row.get("map_id", map_id) != map_id:
            raise ValueError("reference from another map")
        qid = str(row["query_id"])
        if qid in ids:
            raise ValueError("duplicate reference ID")
        ids.add(qid)
        p = safe_image(dataset, row["image_path"])
        if p in paths:
            raise ValueError("reused reference image path")
        paths.add(p)
        c = camera_checked(row.get("camera", manifest.get("camera")))
        if camera is not None and c != camera:
            raise ValueError("reference camera/calibration changed")
        camera = c
        with Image.open(p) as image:
            image.load()
            if image.size != (c["width"], c["height"]):
                raise ValueError("reference image size/calibration mismatch")
        T = se3(row["T_map_camera"])
        h = sha256(p)
        if row.get("image_sha256", h) != h:
            raise ValueError("reference image SHA mismatch")
        image_hashes.add(h)
        validated.append((row, p, h, T))
    query_ids, query_hashes = set(), set()
    for row in queries:
        check_query_online(row)
        if row.get("map_id", map_id) != map_id:
            raise ValueError("query from another map")
        qid = str(row["query_id"])
        if qid in ids or qid in query_ids:
            raise ValueError("reference/query ID overlap or repeated query")
        query_ids.add(qid)
        p = safe_image(dataset, row["image_path"])
        h = sha256(p)
        if p in paths or h in image_hashes:
            raise ValueError("reference/query RGB overlap")
        if row.get("image_sha256", h) != h:
            raise ValueError("query image SHA mismatch")
        if camera_checked(row.get("camera", manifest.get("camera"))) != camera:
            raise ValueError("reference/query calibration mismatch")
        with Image.open(p) as image:
            image.load()
            if image.size != (camera["width"], camera["height"]):
                raise ValueError("query image size/calibration mismatch")
        query_hashes.add(h)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + output.name + ".partial-", dir=output.parent))
    try:
        for sub in ("rgb", "poses", "calibration"):
            (stage / "train" / sub).mkdir(parents=True)
        for index, (row, p, h, T) in enumerate(validated):
            stem = f"frame-{index:06d}"
            (stage / "train/rgb" / (stem + p.suffix.lower())).symlink_to(p)
            np.savetxt(stage / "train/poses" / (stem + ".pose.txt"), T, fmt="%.12g")
            np.savetxt(stage / "train/calibration" / (stem + ".calibration.txt"), [camera["K"][0][0]], fmt="%.12g")
        training_files = [dict(path=str(p.relative_to(stage)), sha256=sha256(p))
                          for p in sorted((stage / "train").rglob("*")) if p.is_file()]
        audit = dict(
            schema="robot_ace_export_v1", status="REFERENCE_EXPORT_VERIFIED", map_id=map_id,
            map_sha256=map_hash, map_manifest_sha256=sha256(map_path), dataset_root=str(dataset),
            reference_manifest_sha256=sha256(refs_path), query_online_manifest_sha256=sha256(queries_path),
            reference_count=len(refs), query_count=len(queries), unique_reference_images=len(image_hashes),
            unique_query_images=len(query_hashes), reference_query_hash_overlap=0, camera=camera,
            coordinate_convention="camera optical (right/down/forward) to metric world/map, c2w SE3",
            query_gt_read=False, reference_gt_role="offline mapping/training only",
            reference_files=[dict(query_id=r["query_id"], path=str(p), sha256=h) for r, p, h, _ in validated],
            training_files=training_files,
            protocol=dict(buffer_features=1000000, epochs=16, batch_size=5120, seed=2089,
                          scope="new Gazebo map engineering pilot, not official 8M benchmark"),
        )
        atomic_json(stage / "export_audit.json", audit)
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return audit


def validate_export(scene):
    scene = Path(scene).resolve(strict=True)
    audit = json.loads((scene / "export_audit.json").read_text())
    dataset = Path(audit["dataset_root"])
    for rel, key in (("map_manifest.json", "map_manifest_sha256"), ("reference/frames.jsonl", "reference_manifest_sha256"), ("query/frames.jsonl", "query_online_manifest_sha256")):
        if sha256(dataset / rel) != audit[key]:
            raise ValueError("capture manifest changed since export: " + rel)
    manifest = json.loads((dataset / "map_manifest.json").read_text())
    if manifest.get("world_path") and sha256(dataset / manifest["world_path"]) != audit["map_sha256"]:
        raise ValueError("world geometry changed since export")
    for row in audit["reference_files"]:
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError("reference RGB changed since export")
    for row in audit["training_files"]:
        if sha256(scene / row["path"]) != row["sha256"]:
            raise ValueError("exported training file changed since export")
    return audit


def train_scene(scene, output, source, python=sys.executable):
    scene, output, source = Path(scene).resolve(strict=True), Path(output).resolve(), Path(source).resolve(strict=True)
    audit = validate_export(scene)
    if output.exists():
        raise FileExistsError("refusing to overwrite ACE training")
    if output.is_relative_to(Path(audit["dataset_root"])):
        raise ValueError("training artifacts must stay outside capture root")
    for p in (source / "ace_encoder_pretrained.pt", source / "ace_trainer.py"):
        if not p.is_file():
            raise FileNotFoundError(p)
    output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[2]
    command = [str(python), "-m", "research3d.scripts.train_ace_local", str(scene), str(output),
               "--ace", str(source), "--seed", "2089", "--buffer-features", "1000000", "--buffer-storage", "cuda"]
    atomic_json(output / "protocol.json", dict(audit=audit, command=command, status="STARTING"))
    with (output / "train.log").open("w") as log:
        result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        atomic_json(output / "status.json", dict(status="FAILED", returncode=result.returncode))
        raise RuntimeError("official ACE training failed; inspect train.log")
    head = output / "heads_local.pt"
    resources = json.loads((output / "training_resources.json").read_text())
    if resources["epochs"] != 16 or resources["buffer_features"] != 1000000:
        raise ValueError("trainer protocol does not match pilot declaration")
    validate_export(scene)
    binding = dict(schema="robot_ace_model_v1", status="TRAINED_NOT_LOCALIZATION_VALIDATED", map_id=audit["map_id"],
                   map_sha256=audit["map_sha256"], dataset_root=audit["dataset_root"], camera=audit["camera"],
                   head_path=str(head), head_sha256=sha256(head), encoder_sha256=sha256(source / "ace_encoder_pretrained.pt"),
                   trainer_sha256=sha256(source / "ace_trainer.py"), export_audit_sha256=sha256(scene / "export_audit.json"),
                   reference_manifest_sha256=audit["reference_manifest_sha256"], training_resources=resources,
                   covariance_policy="DSAC hard inlier count is not a calibrated pose covariance")
    atomic_json(output / "model_binding.json", binding)
    atomic_json(output / "status.json", dict(status="TRAINED_NOT_LOCALIZATION_VALIDATED", head_sha256=binding["head_sha256"]))
    return binding


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    e = sub.add_parser("export")
    e.add_argument("--dataset", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--min-reference-frames", type=int, default=100)
    t = sub.add_parser("train")
    t.add_argument("--scene", required=True)
    t.add_argument("--output", required=True)
    t.add_argument("--source", required=True)
    t.add_argument("--python", default=sys.executable)
    args = p.parse_args()
    if args.action == "export":
        result = export_scene(args.dataset, args.output, args.min_reference_frames)
        print(json.dumps({k: v for k, v in result.items() if k not in {"reference_files", "training_files"}}, indent=2))
    else:
        print(json.dumps(train_scene(args.scene, args.output, args.source, args.python), indent=2))


if __name__ == "__main__":
    main()
