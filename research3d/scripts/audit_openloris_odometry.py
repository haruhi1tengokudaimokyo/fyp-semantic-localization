"""Resolve pure-wheel eligibility using original paper and actual ROS bag topics."""

import argparse, dataclasses, hashlib, json, urllib.request
from pathlib import Path
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


def main(a):
    root = Path(a.dataset)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    paper = out / "ICRA2020_OpenLORIS_Scene.pdf"
    url = "https://qi-she.net/pdf/ICRA2020_Scene.pdf"
    if not paper.exists():
        with urllib.request.urlopen(url, timeout=90) as response:
            content = response.read()
        assert content.startswith(b"%PDF")
        tmp = paper.with_suffix(".tmp")
        tmp.write_bytes(content)
        tmp.replace(paper)
    records = []
    for bag in sorted(root.glob("office1-*.bag")):
        with Reader(bag) as reader:
            connections = [c for c in reader.connections if c.topic == "/odom"]
            assert connections
            store = get_typestore(Stores.ROS1_NOETIC)
            for c in connections:
                store.register(get_types_from_msg(c.msgdef.data, c.msgtype))
            twists = []
            frames = set()
            for c, t, raw in reader.messages(connections=connections):
                m = store.deserialize_ros1(raw, c.msgtype)
                frames.add((m.header.frame_id, m.child_frame_id))
                twists.append(
                    [
                        m.twist.twist.linear.x,
                        m.twist.twist.linear.y,
                        m.twist.twist.angular.z,
                    ]
                )
            x = np.array(twists)
            assert len(x) > 10 and np.isfinite(x).all()
            records.append(
                dict(
                    bag=bag.name,
                    odom_messages=len(x),
                    frame_pairs=sorted(frames),
                    maximum_abs_twist_y=float(np.abs(x[:, 1]).max()),
                    connection_metadata=[
                        dict(c.ext._asdict()) for c in connections
                    ],
                    raw_encoder_topics=[
                        c.topic
                        for c in reader.connections
                        if "encoder" in c.topic.lower()
                    ],
                )
            )
    assert len(records) == 7
    result = dict(
        status="completed",
        paper_url=url,
        paper_sha256=hashlib.sha256(paper.read_bytes()).hexdigest(),
        paper_section="III.A, Robots and Sensors, page2",
        finding="Office sequences use Segway Deliverybot S1. Its base odometry combines wheel encoders with a chassis IMU through a proprietary filter. /odom is NOT established as raw encoder-only data.",
        pure_wheel_eligible=False,
        GT_or_LiDAR_fusion_reported_by_authors=False,
        conditional_input_name="vendor-fused wheel+chassis-IMU base odometry twist, plus separate T265 IMU",
        online_pose_fields_used=False,
        online_twist_fields_used=["linear.x", "angular.z"],
        raw_encoder_topics_available=any(r["raw_encoder_topics"] for r in records),
        bags=records,
        caveat="Vendor source is proprietary; no raw-encoder-only decomposition or internal covariance independence claim. Separate base-odometry experiments cannot be presented as pure Wheel+IMU.",
    )
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("output")
    main(p.parse_args())
