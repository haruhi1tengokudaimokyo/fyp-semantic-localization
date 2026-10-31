"""Explicit wheel planar-motion diagnostic; keeps original 2D runs untouched.
Upstream Wheel3DCen imposes full 3D wheel-frame motion constraints, including
zero lateral/vertical movement assumptions appropriate only on level ground.
No GT-selected numerical parameter sweep or claimed paper reproduction.
"""

import argparse, json, shutil
from pathlib import Path


def main(a):
    src = Path(a.root)
    dst = src.with_name(src.name + "_wheel3d")
    dst.mkdir()
    for name in ["audit.json", "ground_truth_eval_only.npz", "imu_calibrated.npz"]:
        (dst / name).symlink_to(src / name)
    for mode in ["imu_wheel", "stereo_imu", "stereo_imu_wheel"]:
        d = dst / mode
        d.mkdir()
        for p in (src / mode).glob("config*.yaml"):
            if p.name == "config_init.imu_only_original.yaml":
                continue
            text = (
                p.read_text()
                .replace(str(src / mode), str(d))
                .replace("type: Wheel2DCen", "type: Wheel3DCen")
            )
            (d / p.name).write_text(text)
    (dst / "variant.json").write_text(
        json.dumps(
            dict(
                parent=str(src),
                change="Wheel2DCen -> Wheel3DCen only",
                reason="diagnose unobserved vertical drift via upstream full 3D nonholonomic wheel constraints",
                caveat="assumes wheel-frame lateral and vertical velocity approximately zero; diagnostic added after first baseline result, not a preregistered main benchmark",
            ),
            indent=2,
        )
    )
    print(dst)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    main(p.parse_args())
