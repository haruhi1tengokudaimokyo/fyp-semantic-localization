"""Real publisher extra-camera geometry probe; does not requalify a scene."""

import argparse, csv, hashlib, json, zipfile, zlib
from pathlib import Path
import numpy as np, requests
from acquire_hypersim_scene import StrictWebFile, official, array
from knownmap.dataset_geometry import (
    hypersim_world_camera_model,
    hypersim_quantization_bound,
    radial_distance_to_z,
)

SOURCE = Path("/data/fyp_3d/sources/hypersim")


def main(scene, output):
    root = Path("/data/fyp_3d/datasets/hypersim_formal_subset") / scene
    meta = next(
        r
        for r in csv.DictReader(
            (SOURCE / "contrib/mikeroberts3000/metadata_camera_parameters.csv").open()
        )
        if r["scene_name"] == scene
    )
    scale = float(meta["settings_units_info_meters_scale"])
    results = []
    url = next(u for u in official.URLS if u.endswith("/" + scene + ".zip"))
    with zipfile.ZipFile(StrictWebFile(url, requests.Session())) as z:
        for cr in sorted((root / "_detail").glob("cam_*")):
            fs = sorted(
                (root / "images" / ("scene_" + cr.name + "_geometry_hdf5")).glob(
                    "*.depth_meters.hdf5"
                )
            )
            if not fs:
                continue
            f = fs[0]
            frame = int(f.name.split(".")[1])
            ix = list(
                array(cr / "camera_keyframe_frame_indices.hdf5").reshape(-1)
            ).index(frame)
            R = array(cr / "camera_keyframe_orientations.hdf5")[ix]
            p = array(cr / "camera_keyframe_positions.hdf5")[ix]
            position = Path(str(f).replace("depth_meters.hdf5", "position.hdf5"))
            member = str(position.relative_to(root.parent))
            entry = z.getinfo(member)
            if not position.exists():
                value = z.read(entry)
                assert zlib.crc32(value) & 0xFFFFFFFF == entry.CRC
                temp = position.with_suffix(".probe.tmp")
                temp.write_bytes(value)
                temp.replace(position)
            assert zlib.crc32(position.read_bytes()) & 0xFFFFFFFF == entry.CRC
            K, T, w, h = hypersim_world_camera_model(meta, R, p, scale)
            raw = array(position)
            depth = array(f)
            world = raw.astype(float) * scale
            zz, valid = radial_distance_to_z(depth, K)
            v, u = np.indices(depth.shape)
            rays = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(K).T
            recon = (rays * zz[..., None]) @ T[:3, :3].T + T[:3, 3]
            mask = valid & np.isfinite(world).all(-1)
            err = np.linalg.norm(recon - world, axis=-1)
            bound = hypersim_quantization_bound(raw, depth, scale)
            results.append(
                dict(
                    camera=cr.name,
                    frame=frame,
                    pixels=int(mask.sum()),
                    source_gram_error=float(np.max(np.abs(R.T @ R - np.eye(3)))),
                    violations=int((err[mask] > bound[mask]).sum()),
                    normalized_error_max=float(np.max(err[mask] / bound[mask])),
                    p95_error_m=float(np.percentile(err[mask], 95)),
                    position_sha256=hashlib.sha256(position.read_bytes()).hexdigest(),
                    K=K.tolist(),
                )
            )
    result = dict(
        status="geometry_probe_completed",
        scene=scene,
        all_probed_cameras_within_analytic_bound=all(
            r["violations"] == 0 for r in results
        ),
        results=results,
        localization_executed=False,
        qualification_changed=False,
        method="RQ of full publisher world-from-camera basis times publisher camera-ray matrix; no GT fit, no SVD rotation projection",
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    t = out.with_suffix(".tmp")
    t.write_text(json.dumps(result, indent=2))
    t.replace(out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scene")
    p.add_argument("output")
    a = p.parse_args()
    main(a.scene, a.output)
