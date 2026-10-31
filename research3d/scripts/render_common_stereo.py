"""Common, genuinely rendered stereo + disjoint ACE references, static HSSD prior.
All sensors here are explicitly simulated. No fabricated IMU or wheel data.
"""

import argparse, hashlib, json, math, os
from pathlib import Path

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
import numpy as np, habitat_sim, magnum as mn
from PIL import Image
from scipy.spatial.transform import Rotation
from habitat_sim.utils.common import quat_from_angle_axis


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(a.dataset)
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file = str(src / "hssd-hab.scene_dataset_config.json")
    cfg.scene_id = a.scene
    cfg.enable_physics = True
    sensors = []
    for name, kind, x in [
        ("left", habitat_sim.SensorType.COLOR, 0),
        ("right", habitat_sim.SensorType.COLOR, 0.25),
        ("depth", habitat_sim.SensorType.DEPTH, 0),
    ]:
        s = habitat_sim.CameraSensorSpec()
        s.uuid = name
        s.sensor_type = kind
        s.resolution = [480, 640]
        s.position = mn.Vector3(x, 1.2, 0)
        s.hfov = mn.Deg(90)
        sensors.append(s)
    ac = habitat_sim.agent.AgentConfiguration()
    ac.sensor_specifications = sensors
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ac]))
    sim.seed(a.seed)
    if not sim.pathfinder.is_loaded:
        ns = habitat_sim.NavMeshSettings()
        ns.set_defaults()
        ns.agent_radius = 0.15
        ns.agent_height = 1.3
        ns.include_static_objects = True
        assert sim.recompute_navmesh(sim.pathfinder, ns)
    agent = sim.initialize_agent(0)
    K = np.array([[320.0, 0, 319.5], [0, 320.0, 239.5], [0, 0, 1]])
    C = np.eye(4)
    C[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]

    def observe(p, yaw):
        state = habitat_sim.AgentState()
        state.position = p
        state.rotation = quat_from_angle_axis(float(yaw), np.array([0, 1, 0]))
        agent.set_state(state)
        obs = sim.get_sensor_observations()
        T = np.array(
            sim._sensors["left"]._sensor_object.node.absolute_transformation(), float
        ) @ np.diag([1, -1, -1, 1])
        return obs, T

    ace = out / "ace_scene"
    for split in ["train", "test"]:
        for sub in ["rgb", "poses", "calibration"]:
            (ace / split / sub).mkdir(parents=True, exist_ok=True)
    references = []
    cloud = []
    hashes = set()
    for i in range(a.references * 3):
        p = sim.pathfinder.get_random_navigable_point()
        obs, T = observe(p, (i % 8) * math.pi / 4)
        name = f"ref-{i:06d}"
        rgb = obs["left"][..., :3]
        if (
            rgb.std() < 5
            or np.mean(np.isfinite(obs["depth"]) & (obs["depth"] > 0)) < 0.2
            or hashlib.sha256(rgb.tobytes()).hexdigest() in hashes
        ):
            continue
        path = ace / "train/rgb" / (name + ".color.png")
        Image.fromarray(rgb).save(path)
        np.savetxt(ace / "train/poses" / (name + ".pose.txt"), T)
        np.savetxt(ace / "train/calibration" / (name + ".calibration.txt"), K)
        hashes.add(hashlib.sha256(rgb.tobytes()).hexdigest())
        d = obs["depth"][::8, ::8]
        y, x = np.mgrid[0:480:8, 0:640:8]
        v = np.isfinite(d) & (d > 0.1) & (d < 20)
        points = np.stack(
            [(x[v] - 319.5) * d[v] / 320, (y[v] - 239.5) * d[v] / 320, d[v]], 1
        )
        cloud.append(points @ T[:3, :3].T + T[:3, 3])
        references.append(dict(rgb=str(path), T_world_camera=T.tolist(), K=K.tolist()))
        if i % 50 == 0:
            print("REFERENCE", len(references), flush=True)
        if len(references) >= a.references:
            break
    assert len(references) >= min(a.references, 64)
    # Continuous navigable path with <=4cm translation and <=10deg yaw per frame.
    sim.seed(a.seed + 1)
    positions = [np.array(sim.pathfinder.get_random_navigable_point())]
    path_attempts = 0
    while len(positions) < a.frames + 1:
        path_attempts += 1
        assert (
            path_attempts < 10000
        ), "Cannot generate sufficient continuous navigable path"
        start = positions[-1]
        target = sim.pathfinder.get_random_navigable_point()
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = target
        if not sim.pathfinder.find_path(path) or path.geodesic_distance < 0.5:
            continue
        for p, q in zip(path.points[:-1], path.points[1:]):
            dist = np.linalg.norm(q - p)
            steps = max(1, int(np.ceil(dist / 0.04)))
            positions.extend(
                np.array(p) + (np.array(q) - p) * (i / steps)
                for i in range(1, steps + 1)
            )
    positions = positions[: a.frames + 1]
    stereo = out / "stereo"
    for d in ["image_left", "image_right", "imu"]:
        (stereo / d).mkdir(parents=True, exist_ok=True)
    poses = []
    posecv = []
    yaw = 0.0
    duplicate_references = 0
    for i, p in enumerate(positions[: a.frames]):
        direction = positions[i + 1] - p
        goal = math.atan2(-direction[0], -direction[2])
        delta = (goal - yaw + math.pi) % (2 * math.pi) - math.pi
        yaw = goal if i == 0 else yaw + np.clip(delta, -math.pi / 18, math.pi / 18)
        obs, T = observe(p, yaw)
        left = obs["left"][..., :3]
        right = obs["right"][..., :3]
        duplicate_references += int(
            hashlib.sha256(left.tobytes()).hexdigest() in hashes
        )
        lp = stereo / "image_left" / f"{i:06d}_left.png"
        Image.fromarray(left).save(lp)
        Image.fromarray(right).save(stereo / "image_right" / f"{i:06d}_right.png")
        name = f"frame-{i:06d}"
        link = ace / "test/rgb" / (name + ".color.png")
        if not link.exists():
            link.symlink_to(lp.resolve())
        np.savetxt(ace / "test/poses" / (name + ".pose.txt"), T)
        np.savetxt(ace / "test/calibration" / (name + ".calibration.txt"), K)
        ned = C @ T @ C.T
        poses.append(np.r_[ned[:3, 3], Rotation.from_matrix(ned[:3, :3]).as_quat()])
        posecv.append(T)
        if i % 100 == 0:
            print("STEREO", i, flush=True)
    assert duplicate_references == 0, "Query duplicates a train reference image"
    np.savetxt(stereo / "pose_left.txt", poses)
    np.save(stereo / "imu/cam_time.npy", np.arange(a.frames) * 0.1)
    cal = dict(
        width=640,
        height=480,
        K=K.tolist(),
        baseline=0.25,
        time_source="simulation clock, 0.1s/frame; no IMU or wheel measurements",
    )
    (stereo / "calibration.json").write_text(json.dumps(cal, indent=2))
    (ace / "camera.json").write_text(json.dumps(cal, indent=2))
    points = np.concatenate(cloud)
    _, idx = np.unique(
        np.floor(points / 0.03).astype(np.int32), axis=0, return_index=True
    )
    np.savez_compressed(
        out / "readonly_prior_map.npz", points_world=points[idx].astype(np.float32)
    )
    np.savez_compressed(
        out / "ground_truth_eval_only.npz", T_world_camera=np.stack(posecv)
    )
    (out / "references.json").write_text(json.dumps(references, indent=2))
    manifest = dict(
        simulated=True,
        scene_id=a.scene,
        references=len(references),
        continuous_stereo_frames=a.frames,
        reference_query_exact_duplicates=duplicate_references,
        K=K.tolist(),
        baseline_m=0.25,
        seed=a.seed,
        online_inputs=["rendered left RGB", "rendered right RGB"],
        IMU=False,
        wheel=False,
        GT_online=False,
        map_source="offline reference depth and actual reference camera poses, static original HSSD geometry",
        readonly_map_sha256=sha(out / "readonly_prior_map.npz"),
        coordinate_convention="OpenCV camera to Habitat world; TartanAir pose file is C @ T_world_camera @ C.T, NED",
        full_sequence=True,
        is_smoke=a.frames < 100,
    )
    (out / "acceptance.json").write_text(json.dumps(manifest, indent=2))
    sim.close()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("output")
    p.add_argument("--scene", default="105515211_173104185")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--references", type=int, default=384)
    p.add_argument("--seed", type=int, default=1305)
    main(p.parse_args())
