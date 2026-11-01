"""Complete released RGB/semantic/depth supervision for one official scene.

Uses the publisher repository's partial-ZIP design and URL list, with strict
Range/CRC/hash checks, atomic members and resume validation. No original mesh
or completed paper reproduction is claimed.
"""

import argparse, csv, hashlib, importlib.util, json, shutil, time, zipfile, zlib
from pathlib import Path
import h5py, numpy as np, requests
from PIL import Image
from knownmap.dataset_geometry import (
    hypersim_intrinsics,
    radial_distance_to_z,
    gl_asset_pose_to_cv_meters,
    reduce_semantic_labels,
)

SOURCE = Path("/data/fyp_3d/sources/hypersim")
spec = importlib.util.spec_from_file_location(
    "official_hypersim_partial", SOURCE / "contrib/99991/download.py"
)
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)


class StrictWebFile(official.WebFile):
    def __init__(self, url, session):
        self.url, self.session, self.offset = url, session, 0
        with session.get(
            url, headers={"Range": "bytes=0-0"}, stream=True, timeout=(30, 90)
        ) as r:
            r.raise_for_status()
            assert r.status_code == 206
            self.size = int(r.headers["Content-Range"].split("/")[-1])
            self.etag = r.headers.get("ETag")

    def read(self, n=None):
        n = self.available() if n is None or n < 0 else min(n, self.available())
        if n <= 0:
            return b""
        assert n < 256 * 1024**2, "Unexpected giant ZIP read"
        end = self.offset + n - 1
        headers = {"Range": f"bytes={self.offset}-{end}"}
        if self.etag:
            headers["If-Range"] = self.etag
        with self.session.get(self.url, headers=headers, timeout=(30, 120)) as r:
            r.raise_for_status()
            assert r.status_code == 206
            assert (
                r.headers["Content-Range"] == f"bytes {self.offset}-{end}/{self.size}"
            )
            value = r.content
            assert len(value) == n
        self.offset += len(value)
        return value


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def array(p):
    with h5py.File(p) as f:
        return f["dataset"][:]


def main(scene, output, plan_path=None):
    tick = time.monotonic()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        r
        for r in csv.DictReader(
            (SOURCE / "evermotion_dataset/analysis/metadata_images.csv").open()
        )
        if r["scene_name"] == scene and r["included_in_public_release"] == "True"
    ]
    assert rows
    released_total = len(rows)
    if plan_path:
        plan = json.loads(Path(plan_path).read_text())["hypersim"]
        selection = next(s for s in plan["frozen_plan"] if s["scene"] == scene)
        keys = {(r["camera"], r["frame"]) for r in selection["planned_frames"]}
        rows = [r for r in rows if (r["camera_name"], int(r["frame_id"])) in keys]
        assert len(rows) == len(
            keys
        ), "Frozen plan includes unreleased or duplicate frames"
    metadata = next(
        r
        for r in csv.DictReader(
            (SOURCE / "contrib/mikeroberts3000/metadata_camera_parameters.csv").open()
        )
        if r["scene_name"] == scene
    )
    K, width, height = hypersim_intrinsics(metadata)
    wanted = {scene + "/_detail/metadata_scene.csv"}
    for r in rows:
        cam, frame = r["camera_name"], int(r["frame_id"])
        for name in [
            "camera_keyframe_frame_indices",
            "camera_keyframe_positions",
            "camera_keyframe_orientations",
        ]:
            wanted.add(f"{scene}/_detail/{cam}/{name}.hdf5")
        wanted.add(
            f"{scene}/images/scene_{cam}_final_preview/frame.{frame:04d}.tonemap.jpg"
        )
        for name in ["depth_meters", "semantic", "semantic_instance"]:
            wanted.add(
                f"{scene}/images/scene_{cam}_geometry_hdf5/frame.{frame:04d}.{name}.hdf5"
            )
    first = rows[0]
    position_name = f"{scene}/images/scene_{first['camera_name']}_geometry_hdf5/frame.{int(first['frame_id']):04d}.position.hdf5"
    wanted.add(position_name)
    url = next(u for u in official.URLS if u.endswith("/" + scene + ".zip"))
    session = requests.Session()
    remote = StrictWebFile(url, session)
    members = []
    with zipfile.ZipFile(remote) as z:
        available = set(z.namelist())
        assert wanted <= available, sorted(wanted - available)
        for i, name in enumerate(sorted(wanted)):
            rel = Path(name)
            assert not rel.is_absolute() and ".." not in rel.parts
            entry = z.getinfo(name)
            target = out / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            good = target.is_file() and target.stat().st_size == entry.file_size
            if good:
                good = (zlib.crc32(target.read_bytes()) & 0xFFFFFFFF) == entry.CRC
            if not good:
                assert (
                    shutil.disk_usage(out).free > 30 * 1024**3 + entry.file_size
                ), "Preserve data disk reserve"
                temp = target.with_suffix(target.suffix + ".tmp")
                with z.open(entry) as src, temp.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                assert temp.stat().st_size == entry.file_size
                temp.replace(target)
            members.append(
                dict(
                    path=name,
                    bytes=entry.file_size,
                    crc32=entry.CRC,
                    sha256=sha(target),
                )
            )
            if i % 20 == 0:
                print("SCIENTIFIC_MEMBER", i, "/", len(wanted), flush=True)
    params = dict(
        list(csv.reader((out / scene / "_detail/metadata_scene.csv").open()))[1:]
    )
    scale = float(params["meters_per_asset_unit"])
    assert np.isclose(scale, float(metadata["settings_units_info_meters_scale"]))
    cameras = {}
    for cam in {r["camera_name"] for r in rows}:
        root = out / scene / "_detail" / cam
        ids = array(root / "camera_keyframe_frame_indices.hdf5").reshape(-1)
        positions = array(root / "camera_keyframe_positions.hdf5")
        rotations = array(root / "camera_keyframe_orientations.hdf5")
        assert len(ids) == len(set(ids)) == len(positions) == len(rotations)
        cameras[cam] = {
            int(frame): (p, R) for frame, p, R in zip(ids, positions, rotations)
        }
    catalog = []
    truth = []
    counts = np.zeros(40, np.int64)
    unknown = 0
    rgb_hashes = []
    geometry = None
    for r in rows:
        cam, frame = r["camera_name"], int(r["frame_id"])
        prefix = (
            out
            / scene
            / "images"
            / ("scene_" + cam + "_geometry_hdf5")
            / f"frame.{frame:04d}"
        )
        depth = array(str(prefix) + ".depth_meters.hdf5")
        semantic = array(str(prefix) + ".semantic.hdf5")
        instance = array(str(prefix) + ".semantic_instance.hdf5")
        assert depth.shape == semantic.shape == instance.shape == (height, width)
        labels = reduce_semantic_labels(semantic, "hypersim")
        valid_labels = labels != 255
        counts += np.bincount(labels[valid_labels], minlength=40)
        unknown += int((~valid_labels).sum())
        z, valid = radial_distance_to_z(depth, K)
        p, R = cameras[cam][frame]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        Tcv = gl_asset_pose_to_cv_meters(T, scale)
        rgb = (
            out
            / scene
            / "images"
            / ("scene_" + cam + "_final_preview")
            / f"frame.{frame:04d}.tonemap.jpg"
        )
        with Image.open(rgb) as image:
            assert image.size == (width, height)
            image.verify()
        with Image.open(rgb) as image:
            decoded = np.asarray(image.convert("RGB"))
            decoded_sha = hashlib.sha256(decoded.tobytes()).hexdigest()
        rgb_hashes.append(sha(rgb))
        catalog.append(
            dict(
                scene=scene,
                camera=cam,
                frame=frame,
                rgb=str(rgb),
                depth_radial_meters=str(prefix) + ".depth_meters.hdf5",
                semantic=str(prefix) + ".semantic.hdf5",
                instance=str(prefix) + ".semantic_instance.hdf5",
                K=K.tolist(),
                valid_depth_fraction=float(valid.mean()),
                rgb_decoded_sha256=decoded_sha,
            )
        )
        truth.append(
            dict(
                scene=scene,
                camera=cam,
                frame=frame,
                T_world_camera_cv_meters=Tcv.tolist(),
            )
        )
        if cam == first["camera_name"] and frame == int(first["frame_id"]):
            world = array(out / position_name) * scale
            v, u = np.indices(depth.shape)
            rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
            reconstructed = (rays * z[..., None]) @ Tcv[:3, :3].T + Tcv[:3, 3]
            mask = valid & np.isfinite(world).all(-1)
            errors = np.linalg.norm((reconstructed - world)[mask], axis=1)
            assert len(errors) > 1000
            geometry = dict(
                pixels=len(errors),
                median_m=float(np.median(errors)),
                p95_m=float(np.percentile(errors, 95)),
            )
            assert (
                geometry["p95_m"] < 0.005
            ), "Depth/pose/unit/pixel-centre round trip failed"
    result = dict(
        status="accepted_supervision_data",
        scene=scene,
        released_frames=len(rows),
        released_scene_frames_total=released_total,
        acquired_frames=len(catalog),
        width=width,
        height=height,
        K=K.tolist(),
        meters_per_asset_unit=scale,
        geometry_round_trip=geometry,
        semantic_pixel_counts_1_to_40=counts.tolist(),
        unknown_label_pixels=unknown,
        exact_duplicate_rgb_frames=len(rgb_hashes) - len(set(rgb_hashes)),
        all_selected_members_crc_verified=True,
        archive_url=url,
        archive_bytes=remote.size,
        archive_etag=remote.etag,
        downloaded_scientific_bytes=sum(m["bytes"] for m in members),
        full_original_mesh_available=False,
        localization_executed=False,
        roles="supervision/calibration audit only; GT poses in separate file, no online localization results",
        license="Hypersim dataset CC BY-SA 3.0; Apple repository software license and MIT contributed downloader separately retained",
        members=members,
        elapsed_seconds=time.monotonic() - tick,
        frozen_plan_sha256=sha(plan_path) if plan_path else None,
        partition=(
            selection["split"] if plan_path else "diagnostic_only_no_formal_split"
        ),
        source_family_proxy=selection["source_family_proxy"] if plan_path else None,
    )
    for name, value in [
        ("supervision_manifest.json", catalog),
        ("ground_truth_eval_only.json", truth),
        ("acceptance.json", result),
    ]:
        target = out / scene / name
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(target)
    print(json.dumps({k: v for k, v in result.items() if k != "members"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scene")
    p.add_argument("output")
    p.add_argument("--plan")
    a = p.parse_args()
    main(a.scene, a.output, a.plan)
