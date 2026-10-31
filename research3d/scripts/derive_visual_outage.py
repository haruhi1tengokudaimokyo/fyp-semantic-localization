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
    dst.mkdir()
    audit = json.loads((src / "audit.json").read_text())
    bag = Path(src / "imu_wheel/config_system.yaml").read_text()
    import yaml

    system = yaml.safe_load("\n".join(bag.splitlines()[1:]))["sys"]
    inputbag = Path(system["path_bag"])
    outputbag = dst / "online_sensors_camera_outage.bag"
    store = get_typestore(Stores.ROS1_NOETIC)
    kept = {}
    removed = {}
    with Reader(inputbag) as reader, Writer(outputbag) as writer:
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
    for name in ["ground_truth_eval_only.npz", "imu_calibrated.npz"]:
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
                .replace(str(inputbag), str(outputbag))
            )
            (d / p.name).write_text(text)
    audit.update(
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
