"""Re-render depth/native labels at existing offline reference poses only.
RGB queries and eval-only GT stay the original continuous 600-frame sequence.
The immutable original map is unchanged; added reference features are a derived
representation of the same original static HSSD geometry and known reference poses.
"""

import argparse, hashlib, json, os
from pathlib import Path

os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
import habitat_sim, magnum as mn, numpy as np, quaternion
from PIL import Image
from render_semantic_pilot import GROUPS


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(a):
    common = Path(a.common)
    data = Path(a.hssd)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((common / "acceptance.json").read_text())
    digest = sha(common / "readonly_prior_map.npz")
    assert digest == meta["readonly_map_sha256"]
    references = json.loads((common / "references.json").read_text())
    assert len(references) >= 256
    lex = json.loads((data / "semantics/hssd-hab_semantic_lexicon.json").read_text())
    ids = {x["name"]: x["id"] for x in lex["classes"]}
    lut = np.full(65536, 255, np.uint8)
    for i, names in enumerate(GROUPS.values()):
        for name in names:
            lut[ids[name]] = i
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_dataset_config_file = str(data / "hssd-hab.scene_dataset_config.json")
    cfg.scene_id = meta["scene_id"]
    cfg.enable_physics = True
    sensors = []
    for name, kind in [
        ("rgb", habitat_sim.SensorType.COLOR),
        ("depth", habitat_sim.SensorType.DEPTH),
        ("semantic", habitat_sim.SensorType.SEMANTIC),
    ]:
        s = habitat_sim.CameraSensorSpec()
        s.uuid = name
        s.sensor_type = kind
        s.resolution = [480, 640]
        s.position = mn.Vector3(0, 1.2, 0)
        s.hfov = mn.Deg(90)
        sensors.append(s)
    ac = habitat_sim.agent.AgentConfiguration()
    ac.sensor_specifications = sensors
    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ac]))
    agent = sim.initialize_agent(0)
    rows = []
    checks = []
    for i, r in enumerate(references):
        T = np.array(r["T_world_camera"])
        R = T[:3, :3] @ np.diag([1, -1, -1])
        state = habitat_sim.AgentState()
        state.position = T[:3, 3] - R @ np.array([0, 1.2, 0])
        state.rotation = quaternion.from_rotation_matrix(R)
        agent.set_state(state)
        obs = sim.get_sensor_observations()
        actual = np.array(
            sim._sensors["rgb"]._sensor_object.node.absolute_transformation()
        ) @ np.diag([1, -1, -1, 1])
        assert np.allclose(actual, T, atol=2e-5)
        original = np.array(Image.open(r["rgb"]).convert("RGB"))
        error = float(
            np.abs(original.astype(float) - obs["rgb"][..., :3].astype(float)).mean()
        )
        assert error < 1.0, ("Reference RGB/geometry misalignment", i, error)
        depth = obs["depth"]
        label = lut[np.clip(obs["semantic"], 0, 65535)]
        label[(~np.isfinite(depth)) | (depth <= 0)] = 255
        directory = out / "references"
        directory.mkdir(exist_ok=True)
        dp = directory / f"{i:06d}.depth.npy"
        lp = directory / f"{i:06d}.label.png"
        np.save(dp, depth)
        Image.fromarray(label).save(lp)
        rows.append(
            dict(
                split="test",
                scene=meta["scene_id"],
                position_id=i,
                rgb=str(Path(r["rgb"]).resolve()),
                depth=str(dp.resolve()),
                label=str(lp.resolve()),
                K=r["K"],
                T_world_camera=r["T_world_camera"],
            )
        )
        checks.append(error)
        if i % 50 == 0:
            print("OFFLINE_REFERENCE", i, "RGB_MAE", error, flush=True)
    sim.close()
    # Dataset packaging boundary only: GT goes to offline map-builder/evaluator,
    # whose online queries.json removes every pose, depth and semantic label.
    truth = np.load(common / "ground_truth_eval_only.npz")["T_world_camera"]
    queries = sorted((common / "stereo/image_left").glob("*.png"))
    assert len(queries) == len(truth) == 600
    for i, (rgb, T) in enumerate(zip(queries, truth)):
        rows.append(
            dict(
                split="test",
                scene=meta["scene_id"],
                position_id=len(references) + i,
                rgb=str(rgb.resolve()),
                K=meta["K"],
                T_world_camera=T.tolist(),
            )
        )
    split = json.loads(Path(a.building_plan).read_text())
    building = meta["scene_id"].split("_")[0]
    membership = next(
        r["split"] for r in split["scenes"] if r["house_prefix"] == building
    )
    result = dict(
        classes=list(GROUPS),
        frames=rows,
        reference_positions=len(references),
        query_frames=600,
        parent_map_sha256=digest,
        simulated=True,
        GT_online=False,
        reference_rgb_mean_absolute_error_max=max(checks),
        semantic_model_common_building_split=membership,
        semantic_generalization_claim=(
            "Same-building system integration, not unseen-building semantic generalization"
            if membership == "train"
            else "Held-out-building semantic model"
        ),
        map_unchanged=sha(common / "readonly_prior_map.npz") == digest,
    )
    (out / "manifest.json").write_text(json.dumps(result, indent=2))
    (out / "acceptance.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2)
    )
    assert result["map_unchanged"]
    print("COMMON_CONSTRAINT_DATASET", len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("common")
    p.add_argument("hssd")
    p.add_argument("output")
    p.add_argument("--building-plan", required=True)
    main(p.parse_args())
