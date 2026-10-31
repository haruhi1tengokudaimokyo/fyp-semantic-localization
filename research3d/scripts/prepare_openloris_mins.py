"""Build a local GT/LiDAR-free ROS1 bag using measured OpenLORIS sensors.
The original is unchanged. Derived bag is local only (OpenLORIS CC BY-ND).
Gyro-time accel interpolation is bracketed, never extrapolated. Camera images
and wheel twist are copied without resampling. GT is exported separately.
"""

import argparse, copy, hashlib, json, shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stamp(h):
    return h.stamp.sec * 10**9 + h.stamp.nanosec


def mat(t):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat([getattr(t.rotation, k) for k in "xyzw"]).as_matrix()
    m[:3, 3] = [getattr(t.translation, k) for k in "xyz"]
    return m


def main(a):
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    store = get_typestore(Stores.ROS1_NOETIC)
    gyro = []
    acc = []
    gt = []
    cal = {}
    events = []
    counts = {}
    cameras = ["/t265/fisheye1/image_raw", "/t265/fisheye2/image_raw"]
    wheel = "/odom"
    with Reader(Path(a.bag)) as reader:
        for c in reader.connections:
            store.register(get_types_from_msg(c.msgdef.data, c.msgtype))
        for c, t, raw in reader.messages():
            counts[c.topic] = counts.get(c.topic, 0) + 1
            if c.topic in cameras + [wheel]:
                events.append((t, c.topic, c.msgtype, bytes(raw)))
            elif c.topic in ["/t265/gyro/sample", "/t265/accel/sample"]:
                m = store.deserialize_ros1(raw, c.msgtype)
                (gyro if "gyro" in c.topic else acc).append((stamp(m.header), m))
            elif c.topic == "/gt":
                for m in store.deserialize_ros1(raw, c.msgtype).transforms:
                    assert (
                        m.header.frame_id == "gt_map"
                        and m.child_frame_id == "base_link"
                    )
                    gt.append((stamp(m.header), mat(m.transform)))
            elif (
                c.topic == "/tf_static"
                or c.topic.endswith("/imu_info")
                or c.topic.endswith("/camera_info")
            ):
                cal[c.topic] = store.deserialize_ros1(raw, c.msgtype)
    assert all(
        counts.get(t, 0) > 0
        for t in cameras + [wheel, "/gt", "/t265/gyro/sample", "/t265/accel/sample"]
    )
    transforms = {
        m.child_frame_id: (m.header.frame_id, mat(m.transform))
        for m in cal["/tf_static"].transforms
    }

    def base_to(child):
        if child == "base_link":
            return np.eye(4)
        parent, T = transforms[child]
        return base_to(parent) @ T

    T_base_imu = base_to("t265_imu")
    T_imu_base = np.linalg.inv(T_base_imu)

    # Official OpenLORIS documentation specifies affine left multiplication.
    def intrinsic(sensor):
        key = next(
            k
            for k in cal
            if k.startswith("/t265/" + sensor) and k.endswith("/imu_info")
        )
        return np.asarray(cal[key].data).reshape(3, 4)

    G = intrinsic("gyro")
    A = intrinsic("accel")
    acc.sort(key=lambda x: x[0])
    gyro.sort(key=lambda x: x[0])
    ta = np.array([t for t, m in acc], np.int64)
    va = (
        np.array([[getattr(m.linear_acceleration, k) for k in "xyz"] for t, m in acc])
        @ A[:, :3].T
        + A[:, 3]
    )
    merged = []
    for t, m0 in gyro:
        if not ta[0] <= t <= ta[-1]:
            continue
        m = copy.deepcopy(m0)
        av = np.array(
            [
                np.interp(float(t - ta[0]), (ta - ta[0]).astype(float), va[:, j])
                for j in range(3)
            ]
        )
        gv = (
            G[:, :3] @ np.array([getattr(m.angular_velocity, k) for k in "xyz"])
            + G[:, 3]
        )
        for j, k in enumerate("xyz"):
            setattr(m.angular_velocity, k, float(gv[j]))
            setattr(m.linear_acceleration, k, float(av[j]))
        m.header.frame_id = "t265_imu"
        merged.append((t, gv, av))
        events.append(
            (
                t,
                "/imu_calibrated",
                m.__msgtype__,
                bytes(store.serialize_ros1(m, m.__msgtype__)),
            )
        )
    events.sort(key=lambda e: e[0])
    bagout = root / "online_sensors.bag"
    with Writer(bagout) as writer:
        connections = {}
        for t, topic, typ, raw in events:
            if topic not in connections:
                connections[topic] = writer.add_connection(topic, typ, typestore=store)
            writer.write(connections[topic], t, raw)
    with Reader(bagout) as verify:
        online = {c.topic: c.msgcount for c in verify.connections}
        assert set(online) == set(cameras + [wheel, "/imu_calibrated"])
    np.savez_compressed(
        root / "ground_truth_eval_only.npz",
        timestamp_ns=np.array([t for t, T in gt]),
        T_world_base=np.stack([T for t, T in gt]),
    )
    np.savez_compressed(
        root / "imu_calibrated.npz",
        timestamp_ns=np.array([t for t, g, v in merged]),
        angular_velocity=np.stack([g for t, g, v in merged]),
        linear_acceleration=np.stack([v for t, g, v in merged]),
    )
    # YAML files use OpenCV syntax accepted by the actual MINS parser.
    template = Path(a.source) / "mins/config/euroc_mav"
    sim = Path(a.source) / "mins/config/simulation"
    import yaml

    class IndentDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)

    def write(path, data):
        path.write_text(
            "%YAML:1.0\n---\n"
            + yaml.dump(
                data, Dumper=IndentDumper, sort_keys=False, default_flow_style=None
            )
        )

    def load(path):
        return yaml.safe_load("\n".join(path.read_text().splitlines()[1:]))

    for mode, camen, wheelen in [
        ("imu_wheel", False, True),
        ("stereo_imu", True, False),
        ("stereo_imu_wheel", True, True),
    ]:
        d = root / mode
        d.mkdir()
        keys = [
            "system",
            "estimator",
            "camera",
            "imu",
            "init",
            "wheel",
            "gps",
            "lidar",
            "vicon",
        ]
        write(d / "config.yaml", {"config_" + k: "config_" + k + ".yaml" for k in keys})
        est = load(template / "config_estimator.yaml")
        write(d / "config_estimator.yaml", est)
        init = {
            "window_time": 1.0,
            "imu_thresh": 0.5,
            "imu_wheel_thresh": 0.05,
            "imu_only_init": True,
            "imu_gravity_aligned": False,
            "use_gt": False,
            "use_gt_gnss": False,
            "use_gt_lidar": False,
            "cov_size": 0.001,
        }
        write(d / "config_init.yaml", {"init": init})
        write(
            d / "config_system.yaml",
            {
                "sys": {
                    "verbosity": 2,
                    "save_timing": True,
                    "path_timing": str(d / "timing.txt"),
                    "save_state": False,
                    "path_state": str(d / "state") + "/",
                    "save_trajectory": True,
                    "path_trajectory": str(d / "trajectory.txt"),
                    "save_prints": False,
                    "exp_id": 0,
                    "path_bag": str(bagout),
                    "bag_start": 0,
                    "bag_durr": -1,
                }
            },
        )
        # Manufacturer IMUInfo reports variances. Treat sqrt as assumed spectral densities,
        # disclose this approximation; do not claim a stationary Allan calibration.
        gi = cal[next(k for k in cal if "/t265/gyro/" in k and k.endswith("/imu_info"))]
        ai = cal[
            next(k for k in cal if "/t265/accel/" in k and k.endswith("/imu_info"))
        ]
        write(
            d / "config_imu.yaml",
            {
                "imu": {
                    "accel_noise": float(np.sqrt(np.mean(ai.noise_variances))),
                    "accel_bias": float(np.sqrt(np.mean(ai.bias_variances))),
                    "gyro_noise": float(np.sqrt(np.mean(gi.noise_variances))),
                    "gyro_bias": float(np.sqrt(np.mean(gi.bias_variances))),
                    "topic": "/imu_calibrated",
                }
            },
        )
        cam = load(template / "config_camera.yaml")
        cam["cam"].update(enabled=camen, max_slam=0)
        for i in range(2):
            ci = cal["/t265/fisheye" + str(i + 1) + "/camera_info"]
            K = ci.K
            cam["cam" + str(i)].update(
                T_imu_cam=(T_imu_base @ base_to("t265_fisheye" + str(i + 1))).tolist(),
                intrinsics=[float(K[0]), float(K[4]), float(K[2]), float(K[5])],
                distortion_coeffs=list(map(float, ci.D[:4])),
                distortion_model="equidistant",
                resolution=[ci.width, ci.height],
                topic=cameras[i],
            )
        write(d / "config_camera.yaml", cam)
        wh = load(sim / "config_wheel.yaml")
        wh["wheel"].update(
            enabled=wheelen,
            type="Wheel2DCen",
            topic=wheel,
            T_imu_wheel=T_imu_base.tolist(),
        )
        write(d / "config_wheel.yaml", wh)
        for k in ["gps", "lidar", "vicon"]:
            disabled = load(sim / ("config_" + k + ".yaml"))
            disabled[k]["enabled"] = False
            write(d / ("config_" + k + ".yaml"), disabled)
    audit = dict(
        original_bag=str(Path(a.bag).resolve()),
        input_sha256=sha(a.bag),
        derived_bag_sha256=sha(bagout),
        online_topics=online,
        excluded=["/gt", "/scan", "/tf_static", "all depth images", "all other topics"],
        GT_online=False,
        gt_parent="gt_map",
        gt_child="base_link",
        T_base_imu=T_base_imu.tolist(),
        imu_merge="gyro timestamps; bracketed linear interpolation of factory-affine-calibrated accelerometer; no extrapolation",
        gyro_intrinsic=G.tolist(),
        accel_intrinsic=A.tolist(),
        imu_count=len(merged),
        imu_first_last_ns=[merged[0][0], merged[-1][0]],
        noise_caveat="sqrt manufacturer IMUInfo variances treated as assumed spectral densities; not Allan-calibrated",
        wheel_noise_caveat="upstream fixed noise_w/v/p=0.01 assumed, no GT tuning",
        raw_license="OpenLORIS CC BY-ND 4.0; derived bag local only, not redistributed",
        mins_mapping="max_slam=0; MSCKF transient feature constraints only",
    )
    (root / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("bag")
    p.add_argument("output")
    p.add_argument("--source", required=True)
    main(p.parse_args())
