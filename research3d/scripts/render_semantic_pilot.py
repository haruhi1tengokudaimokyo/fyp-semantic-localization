"""Partial-label HSSD segmentation pilot. No invented architecture labels.
Two official training scenes provide spatial train/validation positions; the
third official validation scene is held out entirely for final testing.
Unknown and unselected classes are IGNORE, not background.
"""

import argparse, hashlib, json, math, os
from pathlib import Path

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
import numpy as np
from PIL import Image
import habitat_sim, magnum as mn
from habitat_sim.utils.common import quat_from_angle_axis

GROUPS = {
    "seating": ["couch", "seat", "bench", "stool"],
    "work_surface": ["desk", "table", "counter", "kitchen_counter"],
    "bed": ["bed"],
    "storage": ["cabinet", "chest", "rack", "shelf"],
    "plumbing": ["shower", "bathtub", "toilet", "washbasin"],
    "display": ["tv", "computer", "laptop"],
    "door": ["door"],
}
SCENES = ["107734110_175999914", "107734119_175999932", "105515211_173104185"]


def main(a):
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    data = Path(a.dataset)
    lex = json.loads((data / "semantics/hssd-hab_semantic_lexicon.json").read_text())
    ids = {x["name"]: x["id"] for x in lex["classes"]}
    lut = np.full(65536, 255, np.uint8)
    for i, names in enumerate(GROUPS.values()):
        for n in names:
            lut[ids[n]] = i
    plan = json.loads(Path(a.building_plan).read_text()) if a.building_plan else None
    scenes = [r["scene"] for r in plan["scenes"]] if plan else SCENES
    scene_plan = {r["scene"]: r for r in plan["scenes"]} if plan else {}
    rows = []
    hist = {}
    hashes = set()
    rejected = []
    for sn, scene in enumerate(scenes):
        checkpoint = root / "scene_checkpoints" / (scene + ".json")
        if checkpoint.exists():
            cached = json.loads(checkpoint.read_text())
            assert cached["positions"] == a.positions and cached["plan_sha256"] == (
                hashlib.sha256(Path(a.building_plan).read_bytes()).hexdigest()
                if plan
                else None
            )
            assert all(
                (root / r["rgb"]).exists()
                and (root / r["depth"]).exists()
                and (root / r["label"]).exists()
                for r in cached["rows"]
            )
            rows.extend(cached["rows"])
            hashes.update(r["rgb_sha256"] for r in cached["rows"])
            hist.update({k: np.array(v) for k, v in cached["hist"].items()})
            rejected.extend(cached["rejected"])
            continue
        scene_start = len(rows)
        rejected_start = len(rejected)
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_dataset_config_file = str(data / "hssd-hab.scene_dataset_config.json")
        cfg.scene_id = scene
        cfg.enable_physics = True
        cfg.gpu_device_id = 0
        sensors = []
        for name, kind in [
            ("rgb", habitat_sim.SensorType.COLOR),
            ("semantic", habitat_sim.SensorType.SEMANTIC),
            ("depth", habitat_sim.SensorType.DEPTH),
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
        sim.seed(1305 + sn)
        if not sim.pathfinder.is_loaded:
            ns = habitat_sim.NavMeshSettings()
            ns.set_defaults()
            ns.agent_radius = 0.15
            ns.agent_height = 1.3
            ns.include_static_objects = True
            assert sim.recompute_navmesh(sim.pathfinder, ns)
        agent = sim.initialize_agent(0)
        points = []
        # Reject positions less than 0.5m apart. Entire positions stay in one split.
        for attempt in range(10000):
            p = sim.pathfinder.get_random_navigable_point()
            if all(np.linalg.norm(p - q) > 0.5 for q in points):
                points.append(np.array(p))
            if len(points) == a.positions:
                break
        assert len(points) >= min(30, a.positions), (scene, len(points))
        for pi, p in enumerate(points):
            split = (
                scene_plan[scene]["split"]
                if plan
                else ("test" if sn == 2 else ("val" if pi >= 40 else "train"))
            )
            d = root / split / scene
            d.mkdir(parents=True, exist_ok=True)
            for angle in range(4):
                name = f"{pi:03d}_{angle}"
                state = habitat_sim.AgentState()
                state.position = p
                state.rotation = quat_from_angle_axis(
                    angle * math.pi / 2, np.array([0, 1, 0])
                )
                agent.set_state(state)
                obs = sim.get_sensor_observations()
                rgb = obs["rgb"][..., :3]
                sem = obs["semantic"]
                label = lut[np.clip(sem, 0, 65535)]
                depth = obs["depth"]
                # All labels are derived from actual native class IDs. Invalid depth ignored.
                label[(~np.isfinite(depth)) | (depth <= 0)] = 255
                sha = hashlib.sha256(rgb.tobytes()).hexdigest()
                valid_depth = float(np.mean(np.isfinite(depth) & (depth > 0)))
                if sha in hashes or rgb.std() < 5 or valid_depth < 0.5:
                    rejected.append(
                        dict(
                            scene=scene,
                            position_id=pi,
                            yaw=angle,
                            duplicate=sha in hashes,
                            rgb_std=float(rgb.std()),
                            valid_depth_fraction=valid_depth,
                        )
                    )
                    continue
                hashes.add(sha)
                Image.fromarray(rgb).save(d / (name + ".rgb.png"))
                Image.fromarray(label).save(d / (name + ".label.png"))
                # Store depth/pose for later read-only 3D semantic correspondence experiments.
                np.save(d / (name + ".depth.npy"), depth)
                gl = np.array(
                    sim._sensors["rgb"]._sensor_object.node.absolute_transformation(),
                    float,
                )
                T = gl @ np.diag([1, -1, -1, 1])
                key = split + "/" + scene
                hist.setdefault(key, np.zeros(8, np.int64))
                hist[key] += np.bincount(
                    np.where(label == 255, 7, label).ravel(), minlength=8
                )
                rows.append(
                    dict(
                        split=split,
                        scene=scene,
                        building_group=(
                            scene_plan[scene]["building_group"]
                            if plan
                            else scene.split("_")[0]
                        ),
                        position_id=pi,
                        position=p.tolist(),
                        rgb=str((d / (name + ".rgb.png")).relative_to(root)),
                        label=str((d / (name + ".label.png")).relative_to(root)),
                        depth=str((d / (name + ".depth.npy")).relative_to(root)),
                        T_world_camera=T.tolist(),
                        K=[[160, 0, 159.5], [0, 160, 119.5], [0, 0, 1]],
                        rgb_sha256=sha,
                    )
                )
        sim.close()
        checkpoint.parent.mkdir(exist_ok=True)
        payload = dict(
            rows=rows[scene_start:],
            hist={k: v.tolist() for k, v in hist.items() if k.endswith("/" + scene)},
            rejected=rejected[rejected_start:],
            positions=a.positions,
            plan_sha256=(
                hashlib.sha256(Path(a.building_plan).read_bytes()).hexdigest()
                if plan
                else None
            ),
        )
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload))
        temporary.replace(checkpoint)
        print(scene, "finished", len(rows), flush=True)
    protocol = dict(
        rejected_views=rejected,
        quality_rule="reject RGB std<5, valid depth<50%, exact duplicate RGB; fixed before any model evaluation",
        classes=list(GROUPS),
        class_sources=GROUPS,
        ignore_index=255,
        unlabelled="ID0 and unselected native classes ignored, NEVER converted into wall/background",
        split=(
            "frozen whole synthetic-house groups, disjoint train/val/test"
            if plan
            else "first two official train scene IDs: spatial train/val; third held out"
        ),
        position_min_distance_m=0.5,
        seed=1305,
        independent_layout_groups=plan["independent_layout_groups"] if plan else None,
        building_plan=plan,
        limitations=(
            plan["limitations"]
            if plan
            else [
                "three scene IDs only; independent physical building identities unverified",
                "partial-label furniture/object semantic pilot, no wall/window/column labels",
                "known-pixel conditional metrics cannot establish complete-scene segmentation",
                "not yet evidence for localization improvements",
            ]
        ),
        pixel_histogram={k: v.tolist() for k, v in hist.items()},
        frames=rows,
    )
    if plan:
        assert (
            len([r for r in rows if r["split"] == "train"]) >= 3000
        ), "Formal training set below preregistered minimum"
        for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
            assert not (
                {r["building_group"] for r in rows if r["split"] == left}
                & {r["building_group"] for r in rows if r["split"] == right}
            )
    (root / "manifest.json").write_text(json.dumps(protocol, indent=2))
    print("DONE", len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("output")
    p.add_argument("--building-plan")
    p.add_argument("--positions", type=int, default=50)
    main(p.parse_args())
