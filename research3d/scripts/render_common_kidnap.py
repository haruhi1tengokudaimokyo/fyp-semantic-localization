"""New true stereo measurements with an actual simulated camera teleport.
Offline simulator controls the camera; GT poses are written only for evaluation.
No fabricated image shift, IMU, wheel, or synthetic relative-motion input.
"""

import argparse, hashlib, json, math, os
from pathlib import Path

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
import habitat_sim, magnum as mn, numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from habitat_sim.utils.common import quat_from_angle_axis


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    prior = Path(a.prior)
    source = Path(a.hssd)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((prior / "acceptance.json").read_text())
    digest = sha(prior / "readonly_prior_map.npz")
    assert digest == meta["readonly_map_sha256"]
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file = str(source / "hssd-hab.scene_dataset_config.json")
    cfg.scene_id = meta["scene_id"]
    cfg.enable_physics = True
    sensors = []
    for name, x in [("left", 0), ("right", 0.25)]:
        s = habitat_sim.CameraSensorSpec()
        s.uuid = name
        s.sensor_type = habitat_sim.SensorType.COLOR
        s.resolution = [480, 640]
        s.position = mn.Vector3(x, 1.2, 0)
        s.hfov = mn.Deg(90)
        sensors.append(s)
    ac = habitat_sim.agent.AgentConfiguration()
    ac.sensor_specifications = sensors
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ac]))
    sim.seed(a.seed)
    if not sim.pathfinder.is_loaded:
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_radius = 0.15
        settings.agent_height = 1.3
        settings.include_static_objects = True
        assert sim.recompute_navmesh(sim.pathfinder, settings)
    agent = sim.initialize_agent(0)

    def segment(start, count):
        positions = [np.array(start)]
        for attempt in range(10000):
            if len(positions) >= count:
                return positions[:count]
            path = habitat_sim.ShortestPath()
            path.requested_start = positions[-1]
            path.requested_end = sim.pathfinder.get_random_navigable_point()
            if not sim.pathfinder.find_path(path) or path.geodesic_distance < 0.5:
                continue
            for p, q in zip(path.points[:-1], path.points[1:]):
                steps = max(1, int(np.ceil(np.linalg.norm(q - p) / 0.04)))
                positions.extend(
                    np.array(p) + (np.array(q) - p) * (i / steps)
                    for i in range(1, steps + 1)
                )
        raise RuntimeError("Insufficient navigable continuous path")

    before = segment(sim.pathfinder.get_random_navigable_point(), a.kidnap_frame)
    for attempt in range(10000):
        target = sim.pathfinder.get_random_navigable_point()
        if np.linalg.norm(target - before[-1]) >= 2:
            break
    else:
        raise RuntimeError("No >=2m physical teleport target in this layout")
    after = segment(target, a.frames - a.kidnap_frame)
    positions = before + after
    assert len(positions) == a.frames
    references = json.loads((prior / "references.json").read_text())
    hashes = {
        hashlib.sha256(
            np.array(Image.open(r["rgb"]).convert("RGB")).tobytes()
        ).hexdigest()
        for r in references
    }
    ace = out / "ace_scene"
    stereo = out / "stereo"
    for path in [
        stereo / "image_left",
        stereo / "image_right",
        stereo / "imu",
        ace / "test/rgb",
        ace / "test/poses",
        ace / "test/calibration",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    train = ace / "train"
    if not train.exists():
        train.symlink_to(
            (prior / "ace_scene/train").resolve(), target_is_directory=True
        )
    C = np.eye(4)
    C[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    truth = []
    ned = []
    yaw = 0.0
    K = np.array(meta["K"])
    for i, p in enumerate(positions):
        direction = positions[min(i + 1, len(positions) - 1)] - p
        if i == a.kidnap_frame - 1:
            direction = p - positions[i - 1]
        goal = math.atan2(-direction[0], -direction[2])
        delta = (goal - yaw + math.pi) % (2 * math.pi) - math.pi
        yaw = (
            goal
            if i in [0, a.kidnap_frame]
            else yaw + np.clip(delta, -math.pi / 18, math.pi / 18)
        )
        state = habitat_sim.AgentState()
        state.position = p
        state.rotation = quat_from_angle_axis(float(yaw), np.array([0, 1, 0]))
        agent.set_state(state)
        obs = sim.get_sensor_observations()
        T = np.array(
            sim._sensors["left"]._sensor_object.node.absolute_transformation(), float
        ) @ np.diag([1, -1, -1, 1])
        truth.append(T)
        N = C @ T @ C.T
        ned.append(np.r_[N[:3, 3], Rotation.from_matrix(N[:3, :3]).as_quat()])
        left = obs["left"][..., :3]
        assert hashlib.sha256(left.tobytes()).hexdigest() not in hashes
        lp = stereo / "image_left" / f"{i:06d}_left.png"
        Image.fromarray(left).save(lp)
        Image.fromarray(obs["right"][..., :3]).save(
            stereo / "image_right" / f"{i:06d}_right.png"
        )
        link = ace / "test/rgb" / f"frame-{i:06d}.color.png"
        if not link.exists():
            link.symlink_to(lp.resolve())
        np.savetxt(ace / "test/poses" / f"frame-{i:06d}.pose.txt", T)
        np.savetxt(ace / "test/calibration" / f"frame-{i:06d}.calibration.txt", K)
        if i % 100 == 0:
            print("RENDERED_KIDNAP_STEREO", i, flush=True)
    sim.close()
    np.savetxt(stereo / "pose_left.txt", ned)
    np.save(stereo / "imu/cam_time.npy", np.arange(a.frames) * 0.1)
    np.savez_compressed(
        out / "ground_truth_eval_only.npz", T_world_camera=np.stack(truth)
    )
    for name in ["readonly_prior_map.npz", "references.json"]:
        link = out / name
        if not link.exists():
            link.symlink_to((prior / name).resolve())
    calibration = dict(
        width=640,
        height=480,
        K=K.tolist(),
        baseline=0.25,
        time_source="simulation clock 0.1s/frame, not physical IMU",
    )
    (stereo / "calibration.json").write_text(json.dumps(calibration))
    (ace / "camera.json").write_text(json.dumps(calibration))
    assert sha(prior / "readonly_prior_map.npz") == digest
    result = dict(
        simulated=True,
        scene_id=meta["scene_id"],
        references=len(references),
        continuous_stereo_frames=a.frames,
        GT_online=False,
        IMU=False,
        wheel=False,
        readonly_map_sha256=digest,
        shared_prior_and_ace_training_reference_root=str(prior),
        K=K.tolist(),
        baseline_m=0.25,
        kidnap_frame=a.kidnap_frame,
        measured_rendered_camera_jump_m=float(
            np.linalg.norm(
                truth[a.kidnap_frame][:3, 3] - truth[a.kidnap_frame - 1][:3, 3]
            )
        ),
        actual_camera_teleport=True,
        image_postprocessing_fault=False,
        simulated_timestamp_interval_s=0.1,
        seed=a.seed,
        full_sequence=True,
        is_smoke=False,
        reference_query_exact_duplicates=0,
    )
    assert result["measured_rendered_camera_jump_m"] >= 2
    (out / "acceptance.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("prior")
    p.add_argument("hssd")
    p.add_argument("output")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--kidnap-frame", type=int, default=300)
    p.add_argument("--seed", type=int, default=1315)
    main(p.parse_args())
