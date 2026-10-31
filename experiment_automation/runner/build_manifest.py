"""Discover verified artifacts and build an executable A--J experiment DAG."""

import json, sys
from pathlib import Path
from core import ROOT, REPO, atomic

B = Path("/data/fyp_3d")
MAC = str(B / "envs/macvo/bin/python")
ACE = str(B / "envs/ace/bin/python")
SENSOR = str(B / "envs/sensors/bin/python")
HAB = str(B / "envs/habitat/bin/python")
SCRIPT = REPO / "research3d/scripts"
tasks = []


def add(
    tid,
    group,
    ds,
    cmd,
    validation,
    deps=(),
    gpu=False,
    kind="cpu",
    priority=10,
    inputs=(),
    formal=True,
    coverage=None,
    blocked=None,
    seed=1305,
    disk=1,
    ram=1000,
):
    s = dict(
        task_id=tid,
        experiment_id=group,
        dataset_id=ds,
        command=cmd,
        validation=validation,
        depends_on=list(deps),
        inputs=[str(x) for x in inputs],
        formal=formal,
        seed=seed,
        max_attempts=3 if kind == "download" or tid == "D.expanded.render" else 2,
        retry_delay_s=30,
        priority=priority,
        kind=kind,
        resources=dict(
            gpu=gpu,
            cpu_slots=0 if kind == "download" else (2 if gpu else 1),
            ram_mib=ram,
            disk_gib=disk,
            min_gpu_free_mib=4500,
        ),
        coverage=coverage or {},
    )
    if blocked:
        s["blocked_reason"] = blocked
    tasks.append(s)
    return s


J = lambda p, **kw: dict(path=str(p), kind="json", **kw)
# Existing evidence is revalidated, never rerun as a formal large-scale result.
for tid, group, ds, p, criteria, coverage in [
    (
        "reuse.ace.heads.official",
        "A",
        "7scenes_heads",
        B / "runs/ace_adapter_official/summary.json",
        {
            "equals": {"n": 1000},
            "finite": ["median_translation_m", "median_rotation_deg"],
        },
        {"full_sequence": True, "images": 1000, "scene_ids": ["heads"]},
    ),
    (
        "reuse.ace.heads.local",
        "A",
        "7scenes_heads",
        B / "runs/ace_adapter_local/summary.json",
        {"equals": {"n": 1000}, "finite": ["median_translation_m"]},
        {"full_sequence": True, "images": 1000, "scene_ids": ["heads"]},
    ),
    (
        "reuse.macvo.paper200",
        "B",
        "macvo_demo",
        B / "runs/macvo/paper_cov_200/summary.json",
        {"equals": {"finished_frames": 200}, "finite": ["raw.ate_se3_aligned_rmse_m"]},
        {"images": 200},
    ),
    (
        "reuse.semantic.pilot",
        "D",
        "hssd_pilot",
        B / "runs/semantic/segformer_b0_40epochs/summary.json",
        {"equals": {"status": "completed"}, "finite": ["test.mean_iou"]},
        {
            "images": 585,
            "epochs": 40,
            "scene_ids": ["105515211", "107734110", "107734119"],
        },
    ),
    (
        "reuse.clip.pilot",
        "F",
        "hssd_pilot",
        B / "runs/semantic/clip_hloc_relocalization/summary.json",
        {"equals": {"geometry.queries": 76}, "finite": ["geometry.recall_25cm_10deg"]},
        {"images": 76},
    ),
    (
        "reuse.recovery.pilot",
        "I",
        "7scenes_heads",
        B / "runs/recovery/temporal_gate/summary.json",
        {"finite": ["clean.temporal_gate.recall_5cm_5deg"]},
        {"images": 1000},
    ),
]:
    s = add(
        tid,
        group,
        ds,
        [],
        [J(p, **criteria)],
        inputs=[p],
        formal=tid.startswith("reuse.ace"),
        coverage=coverage,
    )
    s["reuse_existing"] = True
# All downloaded stereo frames are formal, unlike the 200-frame pilot.
root = B / "datasets/macvo_demo/P001_select"
n = len(list((root / "image_left").glob("*.png")))
assert n == 1274
for config, priority in [("paper", 100), ("fast", 70)]:
    out = B / "runs/formal/macvo" / ("demo_" + config + "_1274")
    cmd = [
        MAC,
        str(SCRIPT / "run_macvo.py"),
        str(root),
        str(out),
        "--source",
        str(B / "sources/MAC-VO"),
        "--weights",
        str(B / "models/MACVO_FrontendCov.pth"),
        "--frames",
        str(n),
        "--config",
        config,
        "--encoder",
        "fp32",
        "--decoder",
        "fp32",
        "--frontend-cache",
        str(B / "automation/checkpoints/macvo_dense"),
    ]
    if config == "paper":
        cmd += ["--motion-weights", str(B / "models/MACVO_posenet.pkl")]
    add(
        "macvo.demo." + config,
        "B",
        "macvo_demo",
        cmd,
        [
            J(
                out / "summary.json",
                equals={"finished_frames": n},
                finite=["raw.ate_se3_aligned_rmse_m", "raw.tracking_availability"],
            ),
            dict(
                path=str(out / "trajectory.npz"),
                kind="npz",
                lengths={"T_world_camera_raw": n, "lost_before_interpolation": n},
                finite=["T_world_camera_raw"],
            ),
        ],
        gpu=True,
        priority=priority,
        inputs=[root / "pose_left.txt", B / "models/MACVO_FrontendCov.pth"],
        disk=20,
        ram=2400,
        coverage={"full_sequence": True, "images": n, "scene_ids": ["official_demo"]},
    )
    out_a = out / "audit"
    add(
        "macvo.demo." + config + ".residuals",
        "B",
        "macvo_demo",
        [MAC, str(SCRIPT / "audit_macvo_residuals.py"), str(out)],
        [J(out / "residual_audit.json", finite=["optimized_pixel_residual_median_px"])],
        deps=["macvo.demo." + config],
        inputs=[out / "observations.npz"],
        priority=40,
    )
out = B / "runs/formal/macvo/demo_stereo_1274"
add(
    "macvo.demo.geometric",
    "B",
    "macvo_demo",
    [
        MAC,
        str(SCRIPT / "run_stereo_geometric.py"),
        str(root),
        str(out),
        "--frames",
        str(n),
    ],
    [J(out / "summary.json", equals={"n": n}, finite=["ate_se3_aligned_rmse_m"])],
    inputs=[root / "pose_left.txt"],
    priority=95,
    coverage={"full_sequence": True, "images": n},
    ram=600,
)
out = B / "runs/formal/macvo/demo_weight_replay_1274"
add(
    "macvo.demo.weight_replay",
    "B",
    "macvo_demo",
    [
        MAC,
        str(SCRIPT / "replay_macvo_weights.py"),
        str(B / "runs/formal/macvo/demo_fast_1274"),
        str(out),
        "--source",
        str(B / "sources/MAC-VO"),
    ],
    [
        J(
            out / "summary.json",
            equals={
                "audit.frames": n,
                "audit.identity_applied_inside_actual_solver": True,
            },
            finite=["identity.metrics.ate_se3_aligned_rmse_m"],
        )
    ],
    deps=["macvo.demo.fast"],
    gpu=True,
    priority=60,
    ram=1500,
)
# Complete official ACE splits for every remaining 7-Scenes room, scene-specific heads only.
add(
    "ace.models.all7",
    "A",
    "7scenes",
    [MAC, str(ROOT / "runner/jobs.py"), "ace-models"],
    [
        J(
            B / "models/ace_official/acceptance.json",
            equals={"complete_7scenes_heads": True},
        )
    ],
    kind="download",
    priority=85,
    ram=400,
    disk=2,
)
for scene in ["chess", "fire", "office", "pumpkin", "redkitchen", "stairs"]:
    archive = B / "downloads" / (scene + ".zip")
    ds = B / "datasets/7scenes" / ("7scenes_" + scene)
    dl = "ace.data." + scene
    prep = dl + ".prepare"
    url = (
        "https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/"
        + scene
        + ".zip"
    )
    add(
        dl,
        "A",
        "7scenes_" + scene,
        [MAC, str(SCRIPT / "download_range.py"), url, str(archive), "--workers", "6"],
        [J(str(archive) + ".json", finite=["bytes"], minimum={"bytes": 1000000})],
        kind="download",
        priority=65,
        ram=300,
        disk=8,
    )
    add(
        prep,
        "A",
        "7scenes_" + scene,
        [
            ACE,
            str(SCRIPT / "prepare_heads.py"),
            str(archive),
            str(B / "datasets/7scenes"),
            "--ace",
            str(B / "sources/ace"),
            "--scene",
            scene,
        ],
        [
            J(
                ds / "acceptance.json",
                equals={"scene": scene, "outer_zip_crc_verified": True},
                minimum={"counts.train.frames": 1000, "counts.test.frames": 1000},
            )
        ],
        deps=[dl],
        inputs=[archive],
        priority=65,
        ram=1200,
        disk=12,
    )
    out = B / "runs/formal/ace" / scene / "official"
    s = add(
        "ace." + scene + ".official",
        "A",
        "7scenes_" + scene,
        [
            ACE,
            str(SCRIPT / "evaluate_ace.py"),
            str(ds),
            str(B / "models/ace_official" / ("7scenes_" + scene + ".pt")),
            str(out),
            "--ace",
            str(B / "sources/ace"),
        ],
        [
            J(
                out / "summary.json",
                finite=["median_translation_m", "median_rotation_deg"],
                minimum={"n": 1000},
            )
        ],
        deps=[prep, "ace.models.all7"],
        gpu=True,
        priority=50,
        inputs=[
            ds / "acceptance.json",
            B / "models/ace_official" / ("7scenes_" + scene + ".pt"),
        ],
        coverage={"scene_ids": [scene], "full_sequence": True},
        ram=2500,
    )
    s["model_dataset_id"] = "7scenes_" + scene
    for seed in [2089, 2090, 2091]:
        model = B / "runs/formal/ace" / scene / ("seed" + str(seed))
        train = "ace." + scene + ".train." + str(seed)
        s = add(
            train,
            "A",
            "7scenes_" + scene,
            [
                ACE,
                str(SCRIPT / "train_ace_local.py"),
                str(ds),
                str(model),
                "--ace",
                str(B / "sources/ace"),
                "--seed",
                str(seed),
            ],
            [
                J(
                    model / "training_resources.json",
                    equals={"epochs": 16, "base_seed": seed},
                ),
                dict(path=str(model / "heads_local.pt"), min_bytes=1000000),
            ],
            deps=[prep],
            gpu=True,
            priority=35,
            inputs=[ds / "acceptance.json"],
            seed=seed,
            ram=5500,
            coverage={"scene_ids": [scene], "epochs": 16},
        )
        s["model_dataset_id"] = "7scenes_" + scene
        out = model / "evaluation"
        s = add(
            train + ".eval",
            "A",
            "7scenes_" + scene,
            [
                ACE,
                str(SCRIPT / "evaluate_ace.py"),
                str(ds),
                str(model / "heads_local.pt"),
                str(out),
                "--ace",
                str(B / "sources/ace"),
            ],
            [
                J(
                    out / "summary.json",
                    finite=["median_translation_m", "median_rotation_deg"],
                    minimum={"n": 1000},
                )
            ],
            deps=[train],
            gpu=True,
            priority=36,
            inputs=[model / "heads_local.pt"],
            seed=seed,
            ram=2500,
            coverage={"scene_ids": [scene], "full_sequence": True},
        )
        s["model_dataset_id"] = "7scenes_" + scene
# Actual download + strict integrity audit for expanded 3D assets.
add(
    "hssd.expand24",
    "D",
    "hssd",
    [
        MAC,
        str(SCRIPT / "fetch_assets.py"),
        str(B / "datasets/hssd_expanded"),
        "--dataset",
        "hssd",
        "--scenes",
        "24",
        "--unique-buildings",
    ],
    [J(B / "datasets/hssd_expanded/acquisition.json", minimum={"files.0.bytes": 1})],
    kind="download",
    priority=75,
    disk=12,
    ram=800,
)
add(
    "hssd.expand24.integrity",
    "D",
    "hssd",
    [MAC, str(SCRIPT / "audit_assets.py"), str(B / "datasets/hssd_expanded")],
    [
        J(
            B / "datasets/hssd_expanded/integrity.json",
            equals={"bad": []},
            minimum={"files": 100},
        )
    ],
    deps=["hssd.expand24"],
    priority=75,
    ram=500,
)
# Remaining OpenLORIS complete bags, data preparation can run independently of GPU.
for number in [2, 4, 5, 6, 7]:
    name = "office1-" + str(number)
    bag = B / "datasets/openloris" / (name + ".bag")
    tid = "openloris.data." + name
    add(
        tid,
        "C",
        name,
        [
            ACE,
            str(SCRIPT / "download_tar_member.py"),
            str(B / "runs/openloris_office_tar_members.json"),
            name + ".bag",
            str(bag),
        ],
        [
            J(
                bag.with_suffix(".acquisition.json"),
                finite=["bytes"],
                minimum={"bytes": 1000000},
            )
        ],
        kind="download",
        priority=55,
        ram=400,
        disk=8,
    )
    out = B / "runs/formal/openloris" / name
    add(
        tid + ".prepare",
        "C",
        name,
        [
            SENSOR,
            str(SCRIPT / "prepare_openloris_mins.py"),
            str(bag),
            str(out),
            "--source",
            str(B / "sources/MINS"),
        ],
        [
            J(
                out / "audit.json",
                equals={"GT_online": False},
                minimum={"imu_count": 1000},
            )
        ],
        deps=[tid],
        inputs=[bag],
        priority=55,
        ram=3000,
        disk=12,
    )
# Repair formally observed decomposed-object closure gap; preserve initial file-hash audit.
add(
    "hssd.expand24.closure_repair",
    "D",
    "hssd",
    [
        MAC,
        str(SCRIPT / "fetch_assets.py"),
        str(B / "datasets/hssd_expanded"),
        "--dataset",
        "hssd",
        "--scenes",
        "24",
        "--unique-buildings",
    ],
    [J(B / "datasets/hssd_expanded/acquisition.json", minimum={"files.0.bytes": 1})],
    deps=["hssd.expand24"],
    kind="download",
    priority=90,
    disk=2,
    ram=800,
)
add(
    "hssd.expand24.closure_audit",
    "D",
    "hssd",
    [MAC, str(SCRIPT / "audit_hssd_closure.py"), str(B / "datasets/hssd_expanded")],
    [
        J(
            B / "datasets/hssd_expanded/closure_integrity.json",
            equals={"complete": True, "missing": []},
            minimum={"scenes": 24},
        )
    ],
    deps=["hssd.expand24.closure_repair"],
    priority=90,
)
# Expanded semantic data: conservative whole-house groups, no architecture label invention.
SEM = str(B / "envs/semantic310/bin/python")
expanded = B / "datasets/hssd_semantic_formal"
plan = B / "datasets/hssd_expanded/building_split.json"
add(
    "D.building_split.audit",
    "D",
    "hssd",
    [
        MAC,
        str(SCRIPT / "prepare_building_split.py"),
        str(B / "datasets/hssd_expanded"),
        str(plan),
    ],
    [
        J(
            plan,
            minimum={"independent_layout_groups": 12},
            equals={"real_physical_building_identity_claim": False},
        )
    ],
    deps=["hssd.expand24.integrity"],
    priority=80,
    ram=500,
)
add(
    "D.expanded.render",
    "D",
    "hssd",
    [
        HAB,
        str(SCRIPT / "render_semantic_pilot.py"),
        str(B / "datasets/hssd_expanded"),
        str(expanded),
        "--building-plan",
        str(plan),
        "--positions",
        "150",
    ],
    [J(expanded / "manifest.json", minimum={"independent_layout_groups": 12})],
    deps=["D.building_split.audit", "hssd.expand24.closure_audit"],
    gpu=True,
    priority=65,
    ram=2500,
    disk=18,
    inputs=[plan],
)
for strategy in ["full", "frozen", "scratch"]:
    for seed in [1305, 1306, 1307]:
        model = B / "runs/formal/semantic" / (strategy + "_seed" + str(seed))
        add(
            "D.train." + strategy + "." + str(seed),
            "D",
            "hssd",
            [
                SEM,
                str(SCRIPT / "train_semantic_pilot.py"),
                str(expanded),
                str(model),
                "--weights",
                str(B / "models/mit-b0"),
                "--epochs",
                "40",
                "--seed",
                str(seed),
                "--strategy",
                strategy,
            ],
            [
                J(
                    model / "summary.json",
                    equals={
                        "status": "completed",
                        "epochs": 40,
                        "seed": seed,
                        "strategy": strategy,
                    },
                    minimum={"independent_layout_groups": 12},
                    finite=["test.mean_iou"],
                ),
                dict(path=str(model / "best.pt"), min_bytes=1000000),
            ],
            deps=["D.expanded.render"],
            gpu=True,
            priority=45 if strategy == "full" and seed == 1305 else 22,
            seed=seed,
            ram=2500,
            inputs=[expanded / "manifest.json"],
            coverage={"epochs": 40},
        )
localization = B / "runs/formal/relocalization_hssd"
add(
    "E.multiscene.geometry",
    "E",
    "hssd",
    [
        SEM,
        str(SCRIPT / "formal_relocalization_suite.py"),
        str(expanded),
        str(localization),
        "--geometry-only",
    ],
    [
        J(
            localization / "geometry_summary.json",
            equals={"status": "completed", "GT_online": False},
            minimum={"held_out_scenes": 2, "unique_queries": 100},
        )
    ],
    deps=["D.expanded.render"],
    gpu=True,
    priority=44,
    ram=2200,
    disk=2,
    inputs=[expanded / "manifest.json"],
)
add(
    "F.formal.clip.ablation",
    "F",
    "hssd",
    [
        SEM,
        str(SCRIPT / "formal_relocalization_suite.py"),
        str(expanded),
        str(localization),
        "--checkpoint",
        str(B / "runs/formal/semantic/full_seed1305/best.pt"),
    ],
    [
        J(
            localization / "summary.json",
            equals={"status": "completed", "GT_online": False},
            minimum={"held_out_scenes": 2, "unique_queries": 100},
        )
    ],
    deps=["E.multiscene.geometry", "D.train.full.1305"],
    gpu=True,
    priority=44,
    ram=2400,
    disk=2,
    inputs=[
        expanded / "manifest.json",
        B / "runs/formal/semantic/full_seed1305/best.pt",
    ],
)
# Common-scene integration is an executable branch, independent of semantic training.
common = B / "datasets/common_hssd105"
ace_model = B / "runs/formal/common/ace_seed2089"
ace_eval = ace_model / "evaluation"
mac_run = B / "runs/formal/common/macvo_paper600"
joint = B / "runs/formal/common/continuous_joint"
add(
    "G.common.stereo.render",
    "G",
    "habitat_common",
    [
        HAB,
        str(SCRIPT / "render_common_stereo.py"),
        str(B / "datasets/hssd"),
        str(common),
        "--frames",
        "600",
        "--references",
        "384",
    ],
    [
        J(
            common / "acceptance.json",
            equals={
                "simulated": True,
                "GT_online": False,
                "continuous_stereo_frames": 600,
                "is_smoke": False,
            },
            minimum={"references": 256},
        ),
        dict(path=str(common / "readonly_prior_map.npz"), min_bytes=10000),
    ],
    gpu=True,
    priority=99,
    disk=4,
    ram=1800,
    coverage={
        "images": 600,
        "full_sequence": True,
        "scene_ids": ["105515211_173104185"],
    },
)
s = add(
    "G.common.ace.train",
    "G",
    "habitat_common",
    [
        ACE,
        str(SCRIPT / "train_ace_local.py"),
        str(common / "ace_scene"),
        str(ace_model),
        "--ace",
        str(B / "sources/ace"),
        "--seed",
        "2089",
    ],
    [
        J(
            ace_model / "training_resources.json",
            equals={"epochs": 16, "base_seed": 2089},
        ),
        dict(path=str(ace_model / "heads_local.pt"), min_bytes=1000000),
    ],
    deps=["G.common.stereo.render"],
    gpu=True,
    priority=99,
    seed=2089,
    ram=5500,
    inputs=[common / "acceptance.json"],
)
s["model_dataset_id"] = "habitat_common"
s = add(
    "G.common.ace.eval",
    "G",
    "habitat_common",
    [
        ACE,
        str(SCRIPT / "evaluate_ace.py"),
        str(common / "ace_scene"),
        str(ace_model / "heads_local.pt"),
        str(ace_eval),
        "--ace",
        str(B / "sources/ace"),
    ],
    [
        J(
            ace_eval / "summary.json",
            equals={"n": 600},
            finite=["median_translation_m", "median_rotation_deg"],
        )
    ],
    deps=["G.common.ace.train"],
    gpu=True,
    priority=99,
    ram=2500,
    inputs=[ace_model / "heads_local.pt"],
)
s["model_dataset_id"] = "habitat_common"
add(
    "G.common.macvo",
    "G",
    "habitat_common",
    [
        MAC,
        str(SCRIPT / "run_macvo.py"),
        str(common / "stereo"),
        str(mac_run),
        "--source",
        str(B / "sources/MAC-VO"),
        "--weights",
        str(B / "models/MACVO_FrontendCov.pth"),
        "--motion-weights",
        str(B / "models/MACVO_posenet.pkl"),
        "--frames",
        "600",
        "--config",
        "paper",
        "--encoder",
        "fp32",
        "--decoder",
        "fp32",
        "--frontend-cache",
        str(B / "automation/checkpoints/macvo_dense"),
    ],
    [
        J(
            mac_run / "summary.json",
            equals={"finished_frames": 600},
            finite=["raw.ate_se3_aligned_rmse_m"],
        )
    ],
    deps=["G.common.stereo.render"],
    gpu=True,
    priority=98,
    disk=10,
    ram=2500,
    inputs=[common / "stereo/calibration.json"],
)
add(
    "G.ace_macvo.integration",
    "G",
    "habitat_common",
    [
        MAC,
        str(SCRIPT / "integrate_common_system.py"),
        str(common),
        str(ace_eval),
        str(mac_run),
        str(joint),
    ],
    [
        J(
            joint / "summary.json",
            equals={
                "audit.frames": 600,
                "audit.GT_online": False,
                "audit.map_unchanged": True,
                "audit.continuous_state_propagation": True,
            },
            finite=["MACVO_ACE.recall_5cm_5deg", "MACVO_ACE_gate.valid_pose_fraction"],
        )
    ],
    deps=["G.common.ace.eval", "G.common.macvo"],
    priority=99,
    inputs=[
        common / "readonly_prior_map.npz",
        ace_eval / "predictions.npz",
        mac_run / "trajectory.npz",
    ],
)
# Stereo+IMU requires no wheel provenance assumption. All official office bags.
for number in range(1, 8):
    name = f"office1-{number}"
    out = B / "runs/formal/openloris" / name
    prep = "openloris.data." + name + ".prepare"
    if number in [1, 3]:
        add(
            prep,
            "C",
            name,
            [
                SENSOR,
                str(SCRIPT / "prepare_openloris_mins.py"),
                str(B / "datasets/openloris" / (name + ".bag")),
                str(out),
                "--source",
                str(B / "sources/MINS"),
            ],
            [
                J(
                    out / "audit.json",
                    equals={"GT_online": False},
                    minimum={"imu_count": 1000},
                )
            ],
            priority=55,
            ram=3000,
            disk=8,
        )
    run = "mins." + name + ".stereo_imu"
    add(
        run,
        "C",
        name,
        [
            SENSOR,
            str(SCRIPT / "run_mins_suite.py"),
            str(out),
            "--binary",
            str(B / "build/mins/mins_bag"),
            "--modes",
            "stereo_imu",
        ],
        [
            J(
                out / "execution.json",
                equals={"stereo_imu.returncode": 0, "stereo_imu.use_gt": False},
            )
        ],
        deps=[prep],
        priority=58,
        ram=2400,
        coverage={"full_sequence": True},
    )
    add(
        run + ".eval",
        "C",
        name,
        [SENSOR, str(SCRIPT / "evaluate_mins.py"), str(out), "--modes", "stereo_imu"],
        [J(out / "comparison.json", equals={"stereo_imu.protocol.GT_online": False})],
        deps=[run],
        priority=58,
        ram=500,
    )
# Controlled recovery faults are reported separately from physical kidnapping.
recovery = B / "runs/formal/common/controlled_recovery_v2"
add(
    "I.common.controlled_recovery.v2",
    "I",
    "habitat_common",
    [
        MAC,
        str(SCRIPT / "evaluate_common_recovery.py"),
        str(common),
        str(ace_eval),
        str(mac_run),
        str(recovery),
    ],
    [
        J(
            recovery / "summary.json",
            equals={
                "scope.frames": 600,
                "scope.GT_online": False,
                "scope.controlled_faults_only": True,
                "scope.physical_kidnapping": False,
            },
        )
    ],
    deps=["G.ace_macvo.integration"],
    priority=80,
    inputs=[ace_eval / "predictions.npz", mac_run / "trajectory.npz"],
)
# Actual walking/sitting people, distinct from simulated dynamics and packet loss.
for sequence in ["sitting_xyz", "sitting_static", "walking_xyz", "walking_static"]:
    ds = B / "datasets/tum_rgbd" / sequence
    tid = "J.tum." + sequence
    add(
        tid + ".data",
        "J",
        "tum_" + sequence,
        [
            MAC,
            str(SCRIPT / "acquire_tum_rgbd.py"),
            sequence,
            str(B / "downloads/tum" / (sequence + ".tgz")),
            str(ds),
        ],
        [
            J(
                ds / "acceptance.json",
                equals={
                    "all_png_verified": True,
                    "full_sequence": True,
                    "real_people": True,
                },
            )
        ],
        kind="download",
        priority=50,
        disk=2,
        ram=400,
    )
    add(
        tid + ".rgbd_vo",
        "J",
        "tum_" + sequence,
        [
            MAC,
            str(SCRIPT / "run_tum_rgbd_vo.py"),
            str(ds),
            str(B / "runs/formal/tum" / sequence),
        ],
        [
            J(
                B / "runs/formal/tum" / sequence / "summary.json",
                equals={
                    "status": "completed",
                    "GT_online": False,
                    "prior_map_available": False,
                },
                minimum={"metrics.raw_rgb_frames": 100},
                finite=[
                    "metrics.ate_se3_aligned_rmse_m",
                    "metrics.tracking_availability_all_rgb_frames",
                ],
            )
        ],
        deps=[tid + ".data"],
        priority=50,
        ram=900,
        inputs=[ds / "acceptance.json"],
        coverage={"full_sequence": True},
    )
# Requirements that are not yet executable stay visibly BLOCKED, never fake COMPLETED.
for tid, group, ds, reason in [
    (
        "C.wheel.provenance.audit",
        "C",
        "openloris",
        "Need source-level /odom publisher/driver provenance audit and encoder-vs-upstream-fusion determination; current preliminary /odom results cannot establish this",
    ),
    (
        "B.multitrajectory.acquire",
        "B",
        "tartanair",
        "Additional official compatible full trajectories must be selected and acquired with calibration/pose validation; single demo is insufficient",
    ),
    (
        "F.omcl.official",
        "F",
        "omcl",
        "Dense LSeg/OpenScene feature map + OMCL dependencies/checkpoints not yet validated; pooled CLIP baseline is not OMCL",
    ),
    (
        "H.full.system.ablation",
        "H",
        "habitat_common",
        "Depends on validated joint system and same-scene geometry/semantic/VLM constraints; no forced unreliable semantic updates",
    ),
    (
        "I.formal.recovery",
        "I",
        "habitat_common",
        "Requires continuous common system and physical/simulated motion-controlled kidnapping protocol; current injected-fault replay is preliminary",
    ),
    (
        "J.dynamic.multidataset",
        "J",
        "dynamic_benchmark",
        "TUM/VIODE/InCrowd accessibility, actual dynamic-person labels and lawful sensor/map availability must be audited separately; packet outage is not people",
    ),
]:
    add(tid, group, ds, [], [], formal=True, blocked=reason)
atomic(ROOT / "manifests/tasks.json", tasks)
print("tasks", len(tasks))
if __name__ == "__main__":
    pass
