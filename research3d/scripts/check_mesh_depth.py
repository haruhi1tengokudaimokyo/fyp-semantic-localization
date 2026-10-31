"""Independent original triangle-mesh intersection vs Habitat rasterized depth.
Use empty_stage: no objects/occluders, no map generated from the depth under test.
Native OpenCV integer pixel centers imply cx=(width-1)/2, cy=(height-1)/2.
"""

import argparse, json
from pathlib import Path
import numpy as np
import trimesh


def main(a):
    root = Path(a.run)
    mesh = trimesh.load(a.mesh, force="scene").to_geometry()
    errs = []
    records = []
    for p in sorted(root.glob("*.depth.npy")):
        depth = np.load(p)
        T = np.loadtxt(root / p.name.replace(".depth.npy", ".T_world_camera.txt"))
        h, w = depth.shape
        f = w / 2
        y, x = np.mgrid[16 : h - 16 : 24, 16 : w - 16 : 24]
        z = depth[y.ravel(), x.ravel()]
        valid = z > 0
        x = x.ravel()[valid]
        y = y.ravel()[valid]
        z = z[valid]
        rays = np.stack(
            [(x + 0.5 - w / 2) / f, (y + 0.5 - h / 2) / f, np.ones(len(x))], -1
        )
        world_rays = rays @ T[:3, :3].T
        origins = np.broadcast_to(T[:3, 3], world_rays.shape)
        locations, idx, _ = mesh.ray.intersects_location(
            origins, world_rays, multiple_hits=False
        )
        eye = (locations - T[:3, 3]) @ T[:3, :3]
        error = np.abs(eye[:, 2] - z[idx])
        errs.extend(error.tolist())
        records.append(
            dict(
                frame=p.name,
                rays=len(z),
                hits=len(idx),
                median_abs_depth_m=float(np.median(error)),
            )
        )
    e = np.array(errs)
    report = dict(
        source_mesh=str(a.mesh),
        frames=records,
        n=len(e),
        median_abs_depth_m=float(np.median(e)),
        p95_abs_depth_m=float(np.percentile(e, 95)),
        max_abs_depth_m=float(e.max()),
        fraction_under_1cm=float(np.mean(e < 0.01)),
        pixel_convention="integer centers, cx=(W-1)/2, cy=(H-1)/2",
        method="independent trimesh ray-triangle intersections in original mesh; actual saved sensor T_world_camera",
        pass_criterion="p95 < 0.01 m",
        passed=bool(np.percentile(e, 95) < 0.01),
    )
    (root / "mesh_depth_check.json").write_text(json.dumps(report, indent=2))
    print(report)
    if not report["passed"]:
        raise RuntimeError("Independent depth/geometry check failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("mesh")
    main(p.parse_args())
