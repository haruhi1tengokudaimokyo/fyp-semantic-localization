"""Actual Gazebo image/reference collection and independent query recording.

Reference teleportation is offline map preparation only. Query motion is driven
through velocity commands/physical wheels. World poses are saved separately and
are never forwarded to the local or absolute localization workers.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import subprocess
import shutil
import time

import numpy as np
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def matrix(transform):
    p, q = transform.translation, transform.rotation
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    out[:3, 3] = [p.x, p.y, p.z]
    return out


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def rgb_array(message):
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(message.encoding)
    if channels is None:
        raise ValueError(f"unsupported raw camera encoding {message.encoding}")
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(message.height, message.step)
    im = raw[:, :message.width * channels].reshape(message.height, message.width, channels)[..., :3]
    return im[..., ::-1].copy() if message.encoding.startswith("bgr") else im.copy()


def match_reference_target(base, target):
    x, y, z, yaw = target
    observed_yaw = math.atan2(base[1, 0], base[0, 0])
    dyaw = math.atan2(math.sin(observed_yaw-yaw), math.cos(observed_yaw-yaw))
    upright=math.acos(float(np.clip(base[2,2],-1.,1.)))
    return (np.linalg.norm(base[:2, 3] - [x, y]) <= .04 and abs(dyaw) <= .03
            and abs(base[2,3]-z)<=.04 and upright<=math.radians(5.))


def synchronized_base(truth, timestamp_ns, target=None, max_delta_ns=15_000_000, settle_ns=150_000_000):
    """Match measured GT, requiring a settled history after mapping teleports.

    A target-matching sample immediately after a teleport cannot label an old
    renderer image. Every observed pose over the preceding settling interval
    must match the target, and that interval must actually be covered.
    """
    if not truth: return None
    time_ns, base = min(truth, key=lambda entry: abs(entry[0]-timestamp_ns))
    if abs(time_ns-timestamp_ns) > max_delta_ns: return None
    if target is not None:
        if not match_reference_target(base,target): return None
        history=sorted((t,b) for t,b in truth if timestamp_ns-settle_ns-25_000_000 <= t <= timestamp_ns)
        if len(history)<3 or history[0][0]>timestamp_ns-settle_ns: return None
        if any(not match_reference_target(b,target) for _,b in history): return None
        if any(t1-t0>30_000_000 for (t0,_),(t1,_) in zip(history,history[1:])): return None
    return base.copy()


def image_record(query_id, timestamp_ns, image_path, image_sha256, camera, map_id,
                 right_image_path, right_image_sha256, right_timestamp_ns):
    """Online-safe frame metadata: pose labels are added only by mapping code."""
    if right_timestamp_ns != timestamp_ns:
        raise ValueError('stereo timestamps must match exactly')
    if image_path == right_image_path:
        raise ValueError('left/right images must be distinct files')
    return dict(query_id=query_id,timestamp_ns=timestamp_ns,image_path=image_path,
                image_sha256=image_sha256,camera=camera,map_id=map_id,
                right_image_path=right_image_path,right_image_sha256=right_image_sha256,
                right_timestamp_ns=right_timestamp_ns)


def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo, Imu, JointState
    from tf2_msgs.msg import TFMessage
    from geometry_msgs.msg import Twist

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["reference", "query"])
    parser.add_argument("--world-manifest", type=Path, required=True)
    parser.add_argument("--world", default="airport_hall")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poses", type=Path, help="reference list of x/y/z/yaw base poses")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--image-hz", type=float, default=2)
    parser.add_argument("--commands", type=Path, help="query timed velocity schedule, evaluated in simulation time")
    parser.add_argument("--events", type=Path, help="predeclared simulator-only kidnapping events; never forwarded to localization")
    parser.add_argument("--resume-reference", action="store_true", help="resume only a failed, unsealed reference capture; revalidate every existing image/pose")
    parser.add_argument("--image-topic", default="/robot/camera/image")
    parser.add_argument("--info-topic", default="/robot/camera/camera_info")
    parser.add_argument("--right-image-topic", default="/robot/camera_right/image")
    parser.add_argument("--right-info-topic", default="/robot/camera_right/camera_info")
    parser.add_argument("--gt-topic", default="/robot/ground_truth")
    parser.add_argument("--cmd-topic", default="/robot/cmd_vel")
    args = parser.parse_args()
    if (args.output / args.mode).exists() and not args.resume_reference:
        raise FileExistsError("refusing to overwrite an existing capture split")
    if args.resume_reference and (args.mode != "reference" or (args.output/args.mode/"frames.jsonl").exists()):
        raise ValueError("only incomplete reference capture can resume")
    args.output.mkdir(parents=True, exist_ok=True)
    world = json.loads(args.world_manifest.read_text())
    previous = args.output / "map_manifest.json"
    if previous.exists():
        old_world = json.loads(previous.read_text())
        for key in ("map_id", "map_sha256"):
            if old_world.get(key) != world.get(key):
                raise ValueError("cannot mix capture splits from different maps")
    extrinsic = np.asarray(world["T_base_camera_optical"], float)
    rclpy.init()

    class Recorder(Node):
        def __init__(self):
            super().__init__("fyp_mapping_or_offline_recorder")
            self.image, self.info = None, None
            self.right_info = None
            self.left_images, self.right_images = {}, {}
            self.truth = deque(maxlen=1000)
            self.sensor_counts = dict(imu=0, joint_states=0)
            self.sensor_logs = {}
            if args.mode == "query":
                sensor_dir=args.output/"query/sensors"
                sensor_dir.mkdir(parents=True,exist_ok=True)
                for name in ("imu","joint_states"):
                    self.sensor_logs[name]=(sensor_dir/(name+".jsonl.partial")).open("w",buffering=1)
            self.create_subscription(Image, args.image_topic, self.on_image, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, args.info_topic, self.on_info, qos_profile_sensor_data)
            self.create_subscription(Image, args.right_image_topic, self.on_right_image, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, args.right_info_topic, self.on_right_info, qos_profile_sensor_data)
            self.create_subscription(TFMessage, args.gt_topic, self.on_truth, qos_profile_sensor_data)
            self.create_subscription(Imu, "/robot/imu", self.on_imu, qos_profile_sensor_data)
            self.create_subscription(JointState, "/robot/joint_states", self.on_joints, qos_profile_sensor_data)
            self.command = self.create_publisher(Twist, args.cmd_topic, 10)

        def on_imu(self,msg):
            self.sensor_counts["imu"]+=1
            if "imu" in self.sensor_logs:
                self.sensor_logs["imu"].write(json.dumps(dict(timestamp_ns=stamp_ns(msg.header.stamp),
                    frame_id=msg.header.frame_id,angular_velocity=[msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z],
                    linear_acceleration=[msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z],
                    angular_velocity_covariance=list(msg.angular_velocity_covariance),
                    orientation_excluded=True))+"\n")

        def on_joints(self,msg):
            self.sensor_counts["joint_states"]+=1
            if "joint_states" in self.sensor_logs:
                self.sensor_logs["joint_states"].write(json.dumps(dict(timestamp_ns=stamp_ns(msg.header.stamp),
                    names=list(msg.name),positions=list(msg.position),velocities=list(msg.velocity)))+"\n")

        def on_image(self, msg):
            self.image = msg
            self.left_images[stamp_ns(msg.header.stamp)] = msg
            while len(self.left_images)>16: del self.left_images[min(self.left_images)]

        def on_right_image(self, msg):
            self.right_images[stamp_ns(msg.header.stamp)] = msg
            while len(self.right_images)>16: del self.right_images[min(self.right_images)]

        def on_info(self, msg):
            self.info = msg

        def on_right_info(self, msg):
            self.right_info = msg

        def on_truth(self, msg):
            for tf in msg.transforms:
                if tf.child_frame_id in {"robot", "robot/base_link"}:
                    # PosePublisher model pose is world->model; child link
                    # relative transforms are never interpreted as world poses.
                    if tf.header.frame_id in {"", "world", args.world}:
                        self.truth.append((stamp_ns(tf.header.stamp), matrix(tf.transform)))

        def send(self, v=0., w=0.):
            command = Twist()
            command.linear.x, command.angular.z = float(v), float(w)
            self.command.publish(command)

    node = Recorder()

    def wait_ready(after=-1, target=None, timeout=30):
        limit = time.monotonic() + timeout
        while time.monotonic() < limit:
            rclpy.spin_once(node, timeout_sec=.02)
            if node.image is None or node.info is None or node.right_info is None or not node.truth:
                continue
            paired=sorted(set(node.left_images)&set(node.right_images))
            paired=[t for t in paired if t>after]
            if not paired:
                continue
            # Continuous query capture consumes the oldest unrecorded pair;
            # mapping only needs the latest settled view after teleportation.
            ts=paired[0] if args.mode == "query" else paired[-1]
            base=synchronized_base(node.truth,ts,target)
            if base is None: continue
            return ts, base, node.left_images[ts]
        raise TimeoutError("actual image/calibration/world-pose synchronization unavailable")

    records, truth_records = [], []
    if args.resume_reference:
        positions = json.loads(args.poses.read_text())
        partial = args.output/"reference/frames.jsonl.partial"
        records = [json.loads(line) for line in partial.read_text().splitlines() if line.strip()]
        for i, row in enumerate(records):
            if row["map_id"] != world["map_id"] or row["query_id"] != f"reference/{i:06d}":
                raise ValueError("resumed references have inconsistent identity")
            for key, checksum in [("image_path","image_sha256"),("right_image_path","right_image_sha256")]:
                if sha(args.output/row[key]) != row[checksum]:
                    raise ValueError("resumed reference image checksum changed")
            base = np.asarray(row["T_map_camera"]) @ np.linalg.inv(extrinsic)
            if not match_reference_target(base,positions[i]):
                raise ValueError("resumed reference does not match fixed mapping poses")
        atomic_json(args.output/"reference_resume_audit.json",dict(validated_existing_frames=len(records),
                    world_manifest_sha256=sha(args.world_manifest),map_sha256=world["map_sha256"],
                    policy="completed split immutable; existing RGB/stereo/checksum/pose revalidated; only missing mapping views captured"))

    def save(ts, base, message):
        minimum_free = (100 if str(args.output.resolve()).startswith("/data/") else 8) * 1024**3
        if shutil.disk_usage(args.output).free < minimum_free:
            raise RuntimeError("capture stopped at protected disk free-space reserve")
        camera = dict(width=node.info.width, height=node.info.height,
                      K=np.asarray(node.info.k, float).reshape(3, 3).tolist())
        idx = len(records)
        relative = Path(args.mode) / "rgb" / f"{idx:06d}.png"
        path = args.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        rgb = rgb_array(message)
        PILImage.fromarray(rgb).save(path, compress_level=1)
        query_id = f"{args.mode}/{idx:06d}"
        right=node.right_images[ts]
        if (node.right_info.width,node.right_info.height)!=(node.info.width,node.info.height) or not np.allclose(node.right_info.k,node.info.k,atol=1e-4):
            raise ValueError('rectified stereo calibration mismatch')
        if (right.width,right.height)!=(message.width,message.height):
            raise ValueError('stereo image dimensions mismatch')
        right_relative=Path(args.mode)/'rgb_right'/f'{idx:06d}.png'
        right_path=args.output/right_relative;right_path.parent.mkdir(parents=True,exist_ok=True)
        PILImage.fromarray(rgb_array(right)).save(right_path, compress_level=1)
        row=image_record(query_id,ts,str(relative),sha(path),camera,world['map_id'],
                         str(right_relative),sha(right_path),stamp_ns(right.header.stamp))
        T = base @ extrinsic
        if args.mode == "reference":
            row["T_map_camera"] = T.tolist()
            row["role"] = "offline_mapping_supervision_only"
        records.append(row)
        truth_records.append(dict(query_id=query_id, timestamp_ns=ts, T_map_base=base.tolist(), T_map_camera=T.tolist(),
                                  role="OFFLINE_EVALUATION_ONLY"))
        with (args.output / args.mode / "frames.jsonl.partial").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if len(records) % 10 == 0:
            print("CAPTURED", args.mode, len(records), flush=True)
        return camera

    try:
        ts, base, image = wait_ready()
        camera = None
        if args.mode == "reference":
            if args.poses is None:
                parser.error("reference needs predeclared mapping poses")
            positions = json.loads(args.poses.read_text())
            for target in positions[len(records):]:
                x, y, z, yaw = target
                node.send()
                request = f'name: "robot", position: {{x:{x}, y:{y}, z:{z}}}, orientation: {{z:{math.sin(yaw/2)}, w:{math.cos(yaw/2)}}}'
                for attempt in range(3):
                    result = subprocess.run(["gz", "service", "-s", f"/world/{args.world}/set_pose", "--reqtype", "gz.msgs.Pose",
                                             "--reptype", "gz.msgs.Boolean", "--timeout", "5000", "--req", request],
                                            check=True, capture_output=True, text=True, timeout=8)
                    if "true" in result.stdout:
                        break
                    if attempt == 2:
                        raise RuntimeError(f"Gazebo rejected mapping pose after 3 attempts: {result.stdout} {result.stderr}")
                # Let actual physics/renderer publish the moved camera. Reference
                # label is the measured simulator pose, not the requested pose.
                old = ts
                ts, base, image = wait_ready(after=old + 150_000_000, target=target)
                camera = save(ts, base, image)
        else:
            start, last_saved = ts, -1
            schedule = json.loads(args.commands.read_text()) if args.commands else [{"until_s": args.seconds, "v": .15, "w": 0.}]
            pending_events=json.loads(args.events.read_text()) if args.events else []
            event_log=[]
            deadline = time.monotonic() + args.seconds * 5 + 60
            while (ts - start) / 1e9 < args.seconds:
                if time.monotonic() > deadline:
                    raise TimeoutError("query simulation clock failed to advance")
                elapsed = (ts - start) / 1e9
                while pending_events and elapsed>=pending_events[0]["at_s"]:
                    event=pending_events.pop(0)
                    if event["type"]!="kidnap":raise ValueError("unsupported physical experiment event")
                    x,y,z,yaw=event["target_xyz_yaw"]
                    request=f'name:"robot", position:{{x:{x},y:{y},z:{z}}}, orientation:{{z:{math.sin(yaw/2)},w:{math.cos(yaw/2)}}}'
                    result=subprocess.run(["gz","service","-s",f"/world/{args.world}/set_pose","--reqtype","gz.msgs.Pose",
                        "--reptype","gz.msgs.Boolean","--timeout","5000","--req",request],capture_output=True,text=True,timeout=8,check=True)
                    if "true" not in result.stdout:raise RuntimeError("simulator kidnapping event failed")
                    event_log.append(dict(event,requested_at_timestamp_ns=ts,ground_truth_scope="offline event annotation, no localization input"))
                command = next((c for c in schedule if elapsed < c["until_s"]), {"v": 0, "w": 0})
                node.send(command["v"], command["w"])
                ts, base, image = wait_ready(after=ts, timeout=10)
                if last_saved < 0 or ts - last_saved >= 1e9 / args.image_hz:
                    camera = save(ts, base, image)
                    last_saved = ts
        if camera is None or len(records) < 3:
            raise RuntimeError("insufficient complete samples")
        (args.output / args.mode / "frames.jsonl.partial").replace(args.output / args.mode / "frames.jsonl")
        if args.mode == "query":
            offline = args.output / "offline_gt"
            offline.mkdir()
            (offline / "query_poses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in truth_records))
            atomic_json(offline/"events.json",event_log)
            for name, stream in node.sensor_logs.items():
                stream.close()
                partial=args.output/"query/sensors"/(name+".jsonl.partial")
                partial.replace(partial.with_suffix(""))
        if previous.exists() and not np.allclose(np.asarray(old_world["camera"]["K"]), camera["K"], atol=1e-4):
            raise ValueError("camera calibration changed between capture splits")
        atomic_json(args.output / "map_manifest.json", dict(world, camera=camera))
        atomic_json(args.output / f"{args.mode}_capture_receipt.json", dict(status="CAPTURED", mode=args.mode, frames=len(records),
                     sensor_counts=node.sensor_counts, simulator="Gazebo Harmonic", supervision="reference only; query GT isolated",
                     source_world_manifest_sha256=sha(args.world_manifest), query_physical_motion=args.mode == "query"))
        print(json.dumps({"status": "CAPTURED", "mode": args.mode, "frames": len(records), "output": str(args.output)}))
    except BaseException as exc:
        atomic_json(args.output / "failure.json", dict(status="FAILED_NOT_ACCEPTED", error=repr(exc), captured_frames=len(records)))
        raise
    finally:
        node.send()
        for stream in node.sensor_logs.values():
            if not stream.closed: stream.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
