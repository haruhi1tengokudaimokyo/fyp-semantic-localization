"""Controlled real-sensor visual outage: omit camera packets, preserve IMU/wheel.
Not dynamic-person segmentation, not synthetic IMU. Local derived bag only.
"""

import argparse, json, hashlib, shutil
from pathlib import Path
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore


def main(a):
    src = Path(a.root)
    dst = src.with_name(src.name + "_outage")
    dst.mkdir(exist_ok=True)
    if (dst / "audit.json").exists():
        saved = json.loads((dst / "audit.json").read_text())
        outage = saved["controlled_visual_outage"]
        assert (
            outage["start_relative_s"] == a.start and outage["end_relative_s"] == a.end
        )
        assert (dst / "online_sensors_camera_outage.bag").is_file()
        print("RESUME_DERIVED_OUTAGE", dst, flush=True)
        return
    audit = json.loads((src / "audit.json").read_text())
    bag = Path(src / "imu_wheel/config_system.yaml").read_text()
    import yaml

    system = yaml.safe_load("\n".join(bag.splitlines()[1:]))["sys"]
    inputbag = Path(system["path_bag"])
    outputbag = dst / "online_sensors_camera_outage.bag"
    store = get_typestore(Stores.ROS1_NOETIC)
    kept = {}
    removed = {}
    import os

    temporary = dst / (
        "online_sensors_camera_outage.attempt_" + str(os.getpid()) + ".tmp.bag"
    )
    with Reader(inputbag) as reader, Writer(temporary) as writer:
        start = reader.start_time
        connections = {
            c.id: writer.add_connection(c.topic, c.msgtype, typestore=store)
            for c in reader.connections
        }
        for c, t, raw in reader.messages():
            if "/image_raw" in c.topic and a.start <= (t - start) / 1e9 < a.end:
                removed[c.topic] = removed.get(c.topic, 0) + 1
                continue
            writer.write(connections[c.id], t, raw)
            kept[c.topic] = kept.get(c.topic, 0) + 1
    temporary.replace(outputbag)
    for name in ["ground_truth_eval_only.npz", "imu_calibrated.npz"]:
        if not (dst / name).exists():
            (dst / name).symlink_to(src / name)
    for mode in ["imu_wheel", "stereo_imu", "stereo_imu_wheel"]:
        d = dst / mode
        d.mkdir(exist_ok=True)
        for p in (src / mode).glob("config*.yaml"):
            if p.name == "config_init.imu_only_original.yaml":
                continue
            text = (
                p.read_text()
                .replace(str(src / mode), str(d))
                .replace(str(inputbag), str(outputbag))
            )
            (d / p.name).write_text(text)
    digest = hashlib.sha256()
    with outputbag.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            digest.update(block)
    audit.update(
        parent_derived_bag_sha256=audit["derived_bag_sha256"],
        derived_bag_sha256=digest.hexdigest(),
        online_topics=kept,
        controlled_visual_outage=dict(
            start_relative_s=a.start,
            end_relative_s=a.end,
            removed_camera_messages=removed,
            IMU_wheel_unchanged=True,
            source="original real sensor measurements; camera packet omission only",
            caveat="controlled visual availability stress test, not real dynamic pedestrian dataset",
        ),
    )
    (dst / "audit.json").write_text(json.dumps(audit, indent=2))
    print(dst, removed, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--start", type=float, default=8.0)
    p.add_argument("--end", type=float, default=12.0)
    main(p.parse_args())
