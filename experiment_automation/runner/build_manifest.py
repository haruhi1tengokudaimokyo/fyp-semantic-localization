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
        max_attempts=2,
        retry_delay_s=30,
        priority=priority,
        kind=kind,
        resources=dict(
            gpu=gpu,
            cpu_slots=2 if gpu else 1,
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
        "D.building_split.audit",
        "D",
        "hssd",
        "Expanded scene closure and geometry/layout duplicate audit required before independent-building train/val/test split",
    ),
    (
        "D.formal.multiseed",
        "D",
        "hssd",
        "Requires audited multi-building manifest and resumable full training implementation; pilot weights are not formal completion",
    ),
    (
        "E.multiscene.geometry",
        "E",
        "hssd",
        "Requires expanded immutable reference/query maps; 76-query pilot not formal completion",
    ),
    (
        "F.formal.clip.ablation",
        "F",
        "hssd",
        "Requires common expanded query maps and Top-K protocol; CLIP pilot is available but not multi-building benchmark",
    ),
    (
        "F.omcl.official",
        "F",
        "omcl",
        "Dense LSeg/OpenScene feature map + OMCL dependencies/checkpoints not yet validated; pooled CLIP baseline is not OMCL",
    ),
    (
        "G.common.stereo.render",
        "G",
        "habitat_common",
        "Implementation in progress: must generate continuous true rendered stereo, immutable map, disjoint ACE references and test GT",
    ),
    (
        "G.ace_macvo.integration",
        "G",
        "habitat_common",
        "Requires common-scene ACE adaptation and real MACVO motion; separate-dataset trajectories cannot be fused",
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
