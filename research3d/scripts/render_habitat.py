"""Render fixed Habitat scenes; record actual sensor transforms and semantic IDs.
No physics steps or online map updates. Per scene deterministically sample navmesh.
"""

import argparse, json, math, os
from pathlib import Path

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
import numpy as np
from PIL import Image
import habitat_sim
import magnum as mn
from habitat_sim.utils.common import quat_from_angle_axis


def run(root, config, scene, out, views):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file = str(Path(root) / config)
    cfg.scene_id = scene
    cfg.enable_physics = True
    cfg.gpu_device_id = 0
    sensors = []
    for name, kind in [
        ("rgb", habitat_sim.SensorType.COLOR),
        ("depth", habitat_sim.SensorType.DEPTH),
        ("semantic", habitat_sim.SensorType.SEMANTIC),
    ]:
        s = habitat_sim.CameraSensorSpec()
        s.uuid = name
        s.sensor_type = kind
        s.resolution = [240, 320]
        s.position = mn.Vector3(0, 1.2, 0)
        s.hfov = mn.Deg(90)
        sensors.append(s)
    ac = habitat_sim.agent.AgentConfiguration()
    ac.sensor_specifications = sensors
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ac]))
    sim.seed(20261031)
    if not sim.pathfinder.is_loaded:
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()
        settings.agent_radius = 0.15
        settings.agent_height = 1.3
        settings.include_static_objects = True
        if not sim.recompute_navmesh(sim.pathfinder, settings):
            raise RuntimeError("Navmesh construction failed")
    agent = sim.initialize_agent(0)
    rows = []
    cloud = []
    col = []
    all_ids = {}
    camera_points = []
    roundtrip = []
    for i in range(views):
        state = habitat_sim.AgentState()
        state.position = sim.pathfinder.get_random_navigable_point()
        state.rotation = quat_from_angle_axis(
            (i % 8) * math.pi / 4, np.array([0, 1, 0])
        )
        agent.set_state(state)
        obs = sim.get_sensor_observations()
        rgb = obs["rgb"][..., :3]
        depth = obs["depth"]
        sem = obs["semantic"]
        # Use actual RGB sensor transform, NOT the body pose or a guessed height.
        gl = np.array(
            sim._sensors["rgb"]._sensor_object.node.absolute_transformation(),
            dtype=float,
        )
        T = gl @ np.diag([1, -1, -1, 1])
        K = np.array([[160, 0, 159.5], [0, 160, 119.5], [0, 0, 1]], float)
        Image.fromarray(rgb).save(out / f"{i:04d}.rgb.png")
        np.save(out / f"{i:04d}.depth.npy", depth)
        np.save(out / f"{i:04d}.semantic.npy", sem)
        np.savetxt(out / f"{i:04d}.T_world_camera.txt", T)
        ids, counts = np.unique(sem, return_counts=True)
        for k, v in zip(ids, counts):
            all_ids[str(k)] = all_ids.get(str(k), 0) + int(v)
        y, x = np.mgrid[0:240:4, 0:320:4]
        z = depth[::4, ::4]
        valid = (z > 0) & np.isfinite(z) & (z < 20)
        eye = np.stack(
            [
                (x[valid] - 159.5) * z[valid] / 160,
                (y[valid] - 119.5) * z[valid] / 160,
                z[valid],
            ],
            -1,
        )
        world = eye @ T[:3, :3].T + T[:3, 3]
        inv = (world - T[:3, 3]) @ T[:3, :3]
        roundtrip.append(float(np.max(np.abs(inv - eye))))
        cloud.append(world)
        col.append(rgb[::4, ::4][valid])
        camera_points.append(T[:3, 3])
        rows.append(
            dict(
                frame=i,
                T_world_camera=T.tolist(),
                K=K.tolist(),
                valid_depth_fraction=float(np.mean((depth > 0) & np.isfinite(depth))),
                rgb_std=float(rgb.std()),
                semantic_ids=ids.tolist(),
            )
        )
    np.savez_compressed(
        out / "rendered_cloud.npz",
        points_world=np.concatenate(cloud),
        colors_rgb=np.concatenate(col),
        camera_positions=np.array(camera_points),
    )
    ss = sim.semantic_scene
    categories = []
    if ss:
        for obj in ss.objects:
            if obj and obj.category:
                categories.append(
                    dict(
                        id=obj.id,
                        category=obj.category.name(),
                        index=obj.category.index(),
                    )
                )
    report = dict(
        scene=scene,
        habitat_version=habitat_sim.__version__,
        views=rows,
        semantic_pixel_counts=all_ids,
        semantic_scene_objects=categories,
        max_roundtrip_m=max(roundtrip),
        physics_enabled=True,
        physics_steps=0,
        depth="axial camera Z, meters",
        pose="actual RGB sensor OpenGL transform @ diag(1,-1,-1,1)",
        coordinate_check="algebraic roundtrip; independent mesh depth comparison recorded separately",
    )
    (out / "acceptance.json").write_text(json.dumps(report, indent=2))
    sim.close()
    print(scene, "rendered", views, "ids", all_ids, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("config")
    p.add_argument("scene")
    p.add_argument("output")
    p.add_argument("--views", type=int, default=24)
    a = p.parse_args()
    run(a.root, a.config, a.scene, a.output, a.views)
