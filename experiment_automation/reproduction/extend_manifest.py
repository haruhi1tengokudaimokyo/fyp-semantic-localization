"""Four-paper continuation of the existing DAG; no second scheduler."""

from pathlib import Path


def extend(tasks, repo, base):
    root = repo / "experiment_automation/reproduction"
    mac = str(base / "envs/macvo/bin/python")
    ace = str(base / "envs/ace/bin/python")
    levels = {
        "Official Original",
        "Official Weights Evaluation",
        "Hardware Adapted",
        "Original Method on Alternative Data",
        "Independent Baseline",
    }
    for s in tasks:
        s.setdefault(
            "paper_id",
            (
                "ACE"
                if s["experiment_id"] == "A"
                else ("MACVO" if s["experiment_id"] == "B" and s["command"] else None)
            ),
        )
        if s["experiment_id"] == "A":
            level = (
                "Official Weights Evaluation"
                if "official" in s["task_id"]
                else "Hardware Adapted"
            )
        elif s.get("paper_id") == "MACVO":
            level = "Hardware Adapted"
        else:
            level = "Independent Baseline"
        s.setdefault("reproduction_level", level)
        s.setdefault(
            "online_input_condition", "GT-free online; GT offline evaluation only"
        )
        if s["task_id"] in ["D.train.scratch.1306", "D.train.scratch.1307"]:
            s["blocked_reason"] = (
                "Held by 2026-11-01 four-paper scope: preserve existing evidence; prioritize original-paper reproduction over additional independent semantic training."
            )
    for paper, python in [
        ("MACVO", mac),
        ("ACE", ace),
        ("SG2Loc", str(base / "envs/sg2loc/bin/python")),
        ("OMCL", str(base / "envs/omcl/bin/python")),
    ]:
        output = base / "runs/reproduction" / paper / "source_audit.json"
        tasks.append(
            dict(
                task_id="R." + paper + ".source_audit",
                experiment_id="R_" + paper,
                dataset_id="source_audit",
                paper_id=paper,
                reproduction_level="Independent Baseline",
                online_input_condition="Source audit only; localization not executed",
                command=[python, str(root / "audit_sources.py"), paper, str(output)],
                validation=[
                    dict(
                        path=str(output),
                        kind="json",
                        equals={
                            "status": "audited",
                            "paper_id": paper,
                            "localization_executed": False,
                        },
                    )
                ],
                depends_on=[],
                inputs=[],
                formal=False,
                seed=1305,
                max_attempts=2,
                retry_delay_s=30,
                priority=110,
                kind="cpu",
                resources=dict(
                    gpu=False,
                    cpu_slots=1,
                    ram_mib=700,
                    disk_gib=0,
                    min_gpu_free_mib=4500,
                ),
                coverage={},
            )
        )

    def task(
        tid,
        paper,
        cmd,
        checks,
        deps=(),
        gpu=False,
        ram=700,
        disk=0,
        kind="cpu",
        level="Hardware Adapted",
        inputs=(),
        blocked=None,
    ):
        s = dict(
            task_id=tid,
            experiment_id="R_" + paper,
            dataset_id=tid,
            paper_id=paper,
            reproduction_level=level,
            online_input_condition="GT-free online; GT offline evaluation only",
            command=cmd,
            validation=checks,
            depends_on=list(deps),
            inputs=[str(x) for x in inputs],
            formal=True,
            seed=2089,
            max_attempts=3 if kind == "download" else 2,
            retry_delay_s=120,
            priority=105,
            kind=kind,
            resources=dict(
                gpu=gpu,
                cpu_slots=0 if kind == "download" else (2 if gpu else 1),
                ram_mib=ram,
                disk_gib=disk,
                min_gpu_free_mib=4500,
            ),
            coverage={},
        )
        if blocked:
            s["blocked_reason"] = blocked
        tasks.append(s)
        return s

    def j(p, **equals):
        return dict(path=str(p), kind="json", equals=equals)

    audit = base / "runs/reproduction/ACE/data_audit_7scenes.json"
    task(
        "R.ACE.data_audit",
        "ACE",
        [
            ace,
            str(root / "audit_ace_data.py"),
            str(base / "datasets/7scenes"),
            str(audit),
        ],
        [j(audit, status="audited", original_calibrated_pose_verified=True)],
        level="Official Original",
    )
    for scene in [
        "chess",
        "fire",
        "heads",
        "office",
        "pumpkin",
        "redkitchen",
        "stairs",
    ]:
        ds = base / "datasets/7scenes" / ("7scenes_" + scene)
        out = base / "runs/reproduction/ACE/7scenes" / scene / "buffer8M_mmap"
        tid = "R.ACE.7scenes." + scene + ".train8M"
        task(
            tid,
            "ACE",
            [
                ace,
                str(repo / "research3d/scripts/train_ace_local.py"),
                str(ds),
                str(out),
                "--ace",
                str(base / "sources/ace"),
                "--buffer-features",
                "8000000",
                "--buffer-storage",
                "mmap_cpu",
                "--buffer-cache-dir",
                str(repo.parent / "ace_feature_cache"),
            ],
            [
                j(
                    out / "training_resources.json",
                    buffer_features=8000000,
                    buffer_storage="mmap_cpu",
                    epochs=16,
                    batch_size=5120,
                ),
                dict(path=str(out / "heads_local.pt"), min_bytes=1000000),
            ],
            deps=["R.ACE.data_audit"],
            gpu=True,
            ram=3300,
            disk=0,
            inputs=[audit, ds / "acceptance.json"],
        )
        evaluation = out / "official_evaluation"
        task(
            tid + ".eval",
            "ACE",
            [
                ace,
                str(root / "official_ace_eval.py"),
                str(base / "sources/ace"),
                str(ds),
                str(out / "heads_local.pt"),
                str(evaluation),
            ],
            [
                j(
                    evaluation / "summary.json",
                    status="completed",
                    upstream_script_unchanged=True,
                )
            ],
            deps=[tid],
            gpu=True,
            ram=1700,
            inputs=[out / "heads_local.pt"],
        )
    for group in ["apt1", "apt2", "office1", "office2"]:
        archive = base / "downloads/12scenes" / (group + ".zip")
        task(
            "R.ACE.12scenes." + group + ".download",
            "ACE",
            [
                mac,
                str(root / "acquire_public.py"),
                "https://graphics.stanford.edu/projects/reloc/data/" + group + ".zip",
                str(archive),
            ],
            [j(archive.with_suffix(".zip.json"), zip_crc_verified=True)],
            kind="download",
            disk=1,
            level="Official Original",
        )
    for paper in ["SG2Loc", "OMCL"]:
        task(
            "R." + paper + ".official_benchmark",
            paper,
            [],
            [],
            deps=["R." + paper + ".source_audit"],
            level="Official Original",
            blocked=(
                "Official 3RScan/ScanNet authorization and associated preprocessing/features/checkpoint required; source audit preserves exact motion/gravity input condition."
                if paper == "SG2Loc"
                else "Matterport3D authorization, LSeg checkpoint and Kaolin compiled kernels not yet validated. Paper-condition pose-derived odometry and GT-free global runs are separate protocols."
            ),
        )
        task(
            "R." + paper + ".replica_core",
            paper,
            [],
            [],
            deps=["R." + paper + ".source_audit"],
            level="Original Method on Alternative Data",
            blocked="Full original Replica mesh/semantic assets not yet acquired or storage-budget verified; original feature/map adapter must pass before running core method. ReplicaCAD and HSSD are not Replica.",
        )
    for paper in ["SG2Loc", "OMCL"]:
        out = base / "runs/reproduction" / paper / "environment.json"
        task(
            "R." + paper + ".environment",
            paper,
            [
                str(base / "envs" / paper.lower() / "bin/python"),
                str(root / "install_environment.py"),
                paper,
                str(out),
            ],
            [j(out, status="environment_prepared", localization_executed=False)],
            deps=["R." + paper + ".source_audit"],
            kind="download",
            disk=1,
            ram=900,
        )
    replica = base / "runs/reproduction/data/replica_inventory.json"
    task(
        "R.data.replica.inventory",
        "SG2Loc",
        [mac, str(root / "replica_inventory.py"), str(replica)],
        [j(replica, status="inventoried", full_dataset_acquired=False)],
        kind="download",
        level="Independent Baseline",
    )
    for paper, python in [
        ("ACE", ace),
        ("SG2Loc", str(base / "envs/sg2loc/bin/python")),
        ("OMCL", str(base / "envs/omcl/bin/python")),
    ]:
        output = base / "runs/reproduction" / paper / "runtime_source_audit_v2.json"
        task(
            "R." + paper + ".runtime_verify",
            paper,
            [python, str(root / "audit_sources.py"), paper, str(output)],
            [j(output, status="audited", localization_executed=False)],
            deps=(
                ["R." + paper + ".source_audit"]
                if paper == "ACE"
                else ["R." + paper + ".environment"]
            ),
            level="Independent Baseline",
        )
    for s in tasks:
        if s["task_id"] in ["R.SG2Loc.official_benchmark", "R.OMCL.official_benchmark"]:
            s["online_input_condition"] = (
                "Paper condition: GT-derived relative motion and gravity; not GT-free FYP"
                if s["paper_id"] == "SG2Loc"
                else "Paper condition: noisy GT-derived odometry and known-start tracking; not GT-free global localization"
            )
    for group in ["apt1", "apt2", "office1", "office2"]:
        archive = base / "downloads/12scenes" / (group + ".zip")
        out = base / "datasets/12scenes"
        task(
            "R.ACE.12scenes." + group + ".prepare",
            "ACE",
            [ace, str(root / "prepare_12scenes.py"), str(archive), group, str(out)],
            [
                j(
                    out / (group + "_acceptance.json"),
                    status="prepared",
                    pose_version="original",
                    complete_RGB_pose_archive_selection=True,
                    complete_official_scene_group=True,
                )
            ],
            deps=["R.ACE.12scenes." + group + ".download"],
            level="Official Original",
            disk=1,
            ram=900,
            inputs=[archive],
        )
    lseg = base / "models/lseg/demo_e200.ckpt"
    task(
        "R.OMCL.lseg_weights",
        "OMCL",
        [str(base / "envs/omcl/bin/python"), str(root / "acquire_lseg.py"), str(lseg)],
        [
            j(
                lseg.with_suffix(".json"),
                status="acquired",
                localization_executed=False,
            ),
            dict(
                path=str(lseg),
                min_bytes=100000000,
                fresh=False,
                sha256_from_json=dict(
                    path=str(lseg.with_suffix(".json")), key="sha256"
                ),
            ),
        ],
        deps=["R.OMCL.environment"],
        kind="download",
        ram=2500,
        disk=2,
        level="Official Weights Evaluation",
    )
    for paper in ["SG2Loc", "OMCL"]:
        output = base / "runs/reproduction" / paper / "kernel_probe.json"
        s = task(
            "R." + paper + ".kernel_probe",
            paper,
            [
                str(base / "envs" / paper.lower() / "bin/python"),
                str(root / "kernel_probe.py"),
                paper,
                str(output),
            ],
            [j(output, status="kernel_verified", localization_executed=False)],
            deps=["R." + paper + ".environment"],
            gpu=True,
            ram=1200,
        )
        s["priority"] = 115
        s["formal"] = False
    for s in tasks:
        if s["task_id"].startswith("R.") and any(
            k in s["task_id"]
            for k in [
                "source_audit",
                "runtime_verify",
                ".environment",
                ".inventory",
                ".kernel_probe",
                ".lseg_weights",
            ]
        ):
            s["formal"] = False
    for paper in ["SG2Loc", "OMCL"]:
        output = base / "runs/reproduction" / paper / "model_dependencies.json"
        s = task(
            "R." + paper + ".model_dependencies",
            paper,
            [
                str(base / "envs" / paper.lower() / "bin/python"),
                str(root / "install_model_dependencies.py"),
                paper,
                str(output),
            ],
            [j(output, status="dependencies_installed", localization_executed=False)],
            deps=["R." + paper + ".environment"],
            kind="download",
            disk=1,
            ram=1000,
        )
        s["formal"] = False
        kernel = next(
            s for s in tasks if s["task_id"] == "R." + paper + ".kernel_probe"
        )
        kernel["depends_on"] = ["R." + paper + ".model_dependencies"]
    output = base / "runs/reproduction/SG2Loc/nvrtc_runtime.json"
    s = task(
        "R.SG2Loc.nvrtc_runtime",
        "SG2Loc",
        [
            str(base / "envs/sg2loc/bin/python"),
            str(root / "install_nvrtc.py"),
            str(output),
        ],
        [j(output, status="compiler_runtime_loaded", localization_executed=False)],
        deps=["R.SG2Loc.model_dependencies"],
        kind="download",
        disk=0.2,
        ram=700,
    )
    s["formal"] = False
    kernel = next(s for s in tasks if s["task_id"] == "R.SG2Loc.kernel_probe")
    kernel["depends_on"] = ["R.SG2Loc.nvrtc_runtime"]
    # cuda.pathfinder resolves PyPI NVIDIA libraries in this isolated venv.
    kernel["env"] = {
        "LD_LIBRARY_PATH": ":".join(
            str(base / "envs/sg2loc/lib/python3.10/site-packages/nvidia" / directory)
            for directory in ["cuda_nvrtc/lib", "cuda_nvcc/nvvm/lib64", "nvjitlink/lib"]
        )
    }
    output = base / "runs/reproduction/OMCL/lseg_batch1_probe.json"
    s = task(
        "R.OMCL.lseg_batch1_probe",
        "OMCL",
        [
            str(base / "envs/omcl/bin/python"),
            str(root / "probe_lseg.py"),
            str(lseg),
            str(output),
        ],
        [
            j(
                output,
                status="resource_probe_passed",
                localization_executed=False,
                original_LSeg_checkpoint_strict_loaded=True,
            )
        ],
        deps=["R.OMCL.lseg_weights", "R.OMCL.model_dependencies"],
        gpu=True,
        ram=4500,
        level="Official Weights Evaluation",
        inputs=[lseg],
    )
    s["formal"] = False
    s["priority"] = 114
    output = base / "runs/reproduction/data/auxiliary_metadata_audit.json"
    s = task(
        "R.data.auxiliary.audit",
        "SG2Loc",
        [mac, str(root / "auxiliary_data_audit.py"), str(output)],
        [j(output, status="metadata_audited")],
        level="Independent Baseline",
    )
    s["formal"] = False
    output = base / "datasets/hypersim_supervision"
    s = task(
        "R.data.hypersim.ai001001",
        "SG2Loc",
        [mac, str(root / "acquire_hypersim_scene.py"), "ai_001_001", str(output)],
        [
            j(
                output / "ai_001_001/acceptance.json",
                status="accepted_supervision_data",
                acquired_frames=98,
                released_frames=98,
                all_selected_members_crc_verified=True,
                localization_executed=False,
            )
        ],
        deps=["R.data.auxiliary.audit"],
        kind="download",
        disk=1,
        ram=1200,
        level="Independent Baseline",
    )
    s["formal"] = False
    import csv, json, hashlib

    source = base / "sources/hypersim/evermotion_dataset/analysis/metadata_images.csv"
    expected = sum(
        r["scene_name"] == "ai_001_001" and r["included_in_public_release"] == "True"
        for r in csv.DictReader(source.open())
    )
    first = next(s for s in tasks if s["task_id"] == "R.data.hypersim.ai001001")
    first["validation"][0]["equals"].update(
        acquired_frames=expected, released_frames=expected
    )
    plan = base / "runs/reproduction/data/auxiliary_metadata_audit.json"
    if plan.is_file():
        document = json.loads(plan.read_text())
        subset = document["hypersim"]
        original = (
            base
            / "sources/hypersim/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv"
        )
        assert (
            subset["metadata_sha256"]
            == hashlib.sha256(original.read_bytes()).hexdigest()
        ), "Frozen Hypersim plan is stale"
        deps = []
        dataset = base / "datasets/hypersim_formal_subset"
        for selected in subset["frozen_plan"]:
            scene = selected["scene"]
            tid = "R.data.hypersim.subset." + scene
            s = task(
                tid,
                "SG2Loc",
                [
                    mac,
                    str(root / "acquire_hypersim_scene.py"),
                    scene,
                    str(dataset),
                    "--plan",
                    str(plan),
                ],
                [
                    j(
                        dataset / scene / "acceptance.json",
                        status="accepted_supervision_data",
                        acquired_frames=len(selected["planned_frames"]),
                        all_selected_members_crc_verified=True,
                        localization_executed=False,
                        partition=selected["split"],
                    )
                ],
                deps=["R.data.auxiliary.audit"],
                kind="download",
                disk=0.25,
                ram=1200,
                level="Independent Baseline",
                inputs=[plan],
            )
            s["formal"] = False
            s["priority"] = 90
            deps.append(tid)
        s = task(
            "R.data.hypersim.subset_qualification",
            "SG2Loc",
            [
                mac,
                str(root / "qualify_hypersim_subset.py"),
                str(plan),
                str(dataset),
                str(dataset / "qualified"),
            ],
            [
                j(
                    dataset / "qualified/acceptance.json",
                    status="qualified_supervision_subset",
                    acquired_frames=subset["planned_frames"],
                    acquired_scenes=subset["planned_scenes"],
                    exact_decoded_RGB_cross_split_overlap=0,
                    localization_executed=False,
                )
            ],
            deps=deps,
            inputs=[plan],
            level="Independent Baseline",
        )
        s["formal"] = False
    # Calibration v2 is a new scientific acceptance, never a rewrite of v1.
    import copy

    v2deps = []
    for old in list(tasks):
        if not old["task_id"].startswith("R.data.hypersim.subset."):
            continue
        revised = copy.deepcopy(old)
        scene = old["task_id"].split(".")[-1]
        revised["task_id"] = "R.data.hypersim.subset.v2." + scene
        revised["command"] = [
            c.replace("acquire_hypersim_scene.py", "acquire_hypersim_scene_v2.py")
            for c in old["command"]
        ]
        for check in revised["validation"]:
            check["path"] = str(Path(check["path"]).parent / "v2" / "acceptance.json")
            check["equals"]["calibration_revision"] = "hypersim_camera_v2"
        revised["config"] = {
            "calibration_revision": "hypersim_camera_v2",
            "geometry_gate": "analytic per-pixel IEEE half-ULP bound; zero violations",
            "reason": "General physical camera RQ factorization; float64 before unit scaling; float16 storage precision",
        }
        revised["inputs"] += [
            repo / "research3d/knownmap/dataset_geometry.py",
            root / "acquire_hypersim_scene_v2.py",
        ]
        revised["inputs"] = [str(p) for p in revised["inputs"]]
        revised["priority"] = 91
        tasks.append(revised)
        v2deps.append(revised["task_id"])
        old["blocked_reason"] = (
            "Superseded by explicit camera/float16 calibration v2; preserve original attempts and accepted artifacts. No v1 data enters new training."
        )
    old = next(
        (s for s in tasks if s["task_id"] == "R.data.hypersim.subset_qualification"),
        None,
    )
    if old:
        revised = copy.deepcopy(old)
        revised["task_id"] += ".v2"
        revised["depends_on"] = v2deps
        revised["command"][-1] = str(
            base / "datasets/hypersim_formal_subset/qualified_v2"
        )
        revised["command"] += ["--revision", "v2"]
        revised["validation"][0]["path"] = str(
            base / "datasets/hypersim_formal_subset/qualified_v2/acceptance.json"
        )
        revised["validation"][0]["equals"]["calibration_revision"] = "v2"
        tasks.append(revised)
        old["blocked_reason"] = (
            "v1 calibration quarantined; requires full v2 per-scene qualification, no v1/v2 mixing."
        )
    for s in tasks:
        if s["task_id"].startswith("R.ACE.7scenes."):
            s["dataset_id"] = "7scenes_" + s["task_id"].split(".")[3]
            s["model_dataset_id"] = s["dataset_id"]
    output = base / "runs/reproduction/MACVO/official_metrics_existing_demo.json"
    task(
        "R.MACVO.official_metrics_existing",
        "MACVO",
        [mac, str(root / "macvo_official_metrics.py"), str(output)],
        [
            j(
                output,
                status="completed",
                upstream_evaluation_functions_used=True,
                GPU_recomputed=False,
            )
        ],
        inputs=[
            base / "runs/formal/macvo" / d / "trajectory.npz"
            for d in ["demo_paper_1274", "demo_fast_1274", "demo_stereo_1274"]
        ],
    )
    v3deps = []
    for old in list(tasks):
        if not old["task_id"].startswith("R.data.hypersim.subset.v2."):
            continue
        revised = copy.deepcopy(old)
        scene = old["task_id"].split(".")[-1]
        revised["task_id"] = "R.data.hypersim.subset.v3." + scene
        revised["command"] = [
            c.replace("acquire_hypersim_scene_v2.py", "acquire_hypersim_scene_v3.py")
            for c in old["command"]
        ]
        for check in revised["validation"]:
            check["path"] = check["path"].replace("/v2/", "/v3/")
            check["equals"]["calibration_revision"] = "hypersim_camera_v3"
            check["equals"]["per_frame_calibration"] = True
        revised["config"] = {
            "calibration_revision": "hypersim_camera_v3",
            "geometry_gate": "Each camera first planned frame, every valid pixel within analytic storage bound; zero violations",
            "basis": "Per-frame RQ of full publisher world-ray matrix; no SO3 assumption on source axes, no GT fitting",
        }
        revised["inputs"] = [
            str(p).replace(
                "acquire_hypersim_scene_v2.py", "acquire_hypersim_scene_v3.py"
            )
            for p in revised["inputs"]
        ]
        # Fully CRC-receipted raw caches need only the extra per-camera position
        # probes, not another scene download. Keep the 30GiB reserve unchanged.
        cached = (
            base / "datasets/hypersim_formal_subset" / scene / "v2/raw_receipt.json"
        )
        if cached.is_file():
            revised["resources"]["disk_gib"] = 0.02
        else:
            revised["resources"]["disk_gib"] = 0.25
        revised["priority"] = 94
        revised["nonretryable_exit_codes"] = [65]
        revised["reuse_existing"] = True
        tasks.append(revised)
        v3deps.append(revised["task_id"])
        old["blocked_reason"] = (
            "Superseded by v3 per-frame full world-ray calibration; original v2 receipts/failed attempts preserved. No v2-qualified data enters new training."
        )
    if v3deps:
        output = base / "datasets/hypersim_formal_subset/qualified_v3"
        plan = base / "runs/reproduction/data/auxiliary_metadata_audit.json"
        s = task(
            "R.data.hypersim.cleaned_qualification.v3",
            "SG2Loc",
            [
                mac,
                str(root / "qualify_hypersim_cleaned.py"),
                str(plan),
                str(base / "datasets/hypersim_formal_subset"),
                str(output),
            ],
            [
                dict(
                    path=str(output / "acceptance.json"),
                    kind="json",
                    equals=dict(
                        status="qualified_cleaned_supervision_subset",
                        calibration_revision="v3",
                        exact_decoded_RGB_cross_split_overlap=0,
                        localization_executed=False,
                    ),
                    minimum=dict(acquired_scenes=40, acquired_frames=8000),
                )
            ],
            deps=["R.data.auxiliary.audit"],
            ram=300,
            inputs=[plan],
        )
        s.update(
            formal=False,
            priority=95,
            max_wall_seconds=172800,
            task_role="supporting_data_or_runtime",
        )
        for old in tasks:
            if old["task_id"] == "R.data.hypersim.subset_qualification.v2":
                old["blocked_reason"] = (
                    "All new supervision requires v3 per-frame world-ray calibration, quality exclusions explicitly frozen before training."
                )
    for s in tasks:
        if s["task_id"].startswith("R.ACE.7scenes.") and s["task_id"].endswith(".eval"):
            s["priority"] = 112
    output = base / "runs/reproduction/MACVO/historical_metric_runtime.json"
    s = task(
        "R.MACVO.historical_metric_runtime",
        "MACVO",
        [mac, str(root / "install_metric_runtime.py"), str(output)],
        [j(output, status="historical_metrics_runtime_installed", version="v1.29.0")],
        kind="download",
        disk=0.05,
    )
    s["formal"] = False
    output = base / "runs/reproduction/MACVO/official_metrics_existing_demo_v2.json"
    task(
        "R.MACVO.official_metrics_existing.v2",
        "MACVO",
        [mac, str(root / "macvo_official_metrics.py"), str(output), "--historical-evo"],
        [
            j(
                output,
                status="completed",
                upstream_evaluation_functions_used=True,
                GPU_recomputed=False,
                evo_version="v1.29.0",
            )
        ],
        deps=["R.MACVO.historical_metric_runtime"],
        inputs=[
            base / "runs/formal/macvo" / d / "trajectory.npz"
            for d in ["demo_paper_1274", "demo_fast_1274", "demo_stereo_1274"]
        ],
    )
    for s in tasks:
        if s["task_id"] == "R.MACVO.official_metrics_existing":
            s["blocked_reason"] = (
                "evo1.38 rejects upstream duplicate alignment flags; versioned isolated evo1.29 evaluation supersedes this failed attempt."
            )
    output = base / "runs/reproduction/protocol_comparison.json"
    s = task(
        "R.protocol.crosspaper.audit",
        "ACE",
        [mac, str(root / "audit_paper_protocols.py"), str(output)],
        [j(output, status="audited", localization_executed=False)],
        inputs=[
            base / "downloads" / (name + "_paper.txt")
            for name in ["macvo", "ace", "sg2loc", "omcl"]
        ],
    )
    s.update(
        formal=False,
        experiment_id="R_PROTOCOL",
        paper_id=None,
        task_role="supporting_data_or_runtime",
    )
    output = (
        repo.parents[1]
        / "outputs/FYP_Automation/hypersim_full_visual_duplicate_audit.json"
    )
    s = task(
        "R.data.hypersim.visual_duplicate_audit.v3",
        "SG2Loc",
        [
            mac,
            str(root / "audit_visual_duplicates.py"),
            str(base / "datasets/hypersim_formal_subset"),
            str(output),
            "--require-full",
        ],
        [
            j(
                output,
                status="diagnostic_completed",
                scope="full_qualified_subset",
                cross_split_comparison_performed=True,
                selection_changed=False,
                localization_executed=False,
            )
        ],
        deps=["R.data.hypersim.cleaned_qualification.v3"],
        ram=700,
        disk=0.02,
    )
    s.update(formal=False, task_role="supporting_data_or_runtime")
    output = base / "runs/reproduction/ACE/cached_model_members_12scenes.json"
    s = task(
        "R.ACE.12scenes.cached_model_inventory",
        "ACE",
        [mac, str(root / "ace_12_models.py"), "inventory", str(output)],
        [
            j(
                output,
                status="inventoried",
                original_DSLAM_12_heads_complete=True,
                localization_executed=False,
            )
        ],
        inputs=[base / "downloads/ace_models.tar.gz"],
        ram=700,
    )
    s.update(formal=False, task_role="supporting_data_or_runtime")
    modelroot = base / "models/ace_official/12Scenes"
    s = task(
        "R.ACE.12scenes.models.acquire",
        "ACE",
        [mac, str(root / "ace_12_models.py"), "acquire", str(modelroot)],
        [
            j(
                modelroot / "acceptance.json",
                status="acquired",
                complete_original_12scenes_heads=True,
                member_zip_crc_verified=True,
                localization_executed=False,
            )
        ],
        deps=["R.ACE.12scenes.apt1.prepare"],
        kind="download",
        disk=0.12,
        ram=700,
        level="Official Weights Evaluation",
    )
    s["formal"] = False
    train_template = next(
        s for s in tasks if s["task_id"] == "R.ACE.7scenes.chess.train8M"
    )
    eval_template = next(
        s for s in tasks if s["task_id"] == "R.ACE.7scenes.chess.train8M.eval"
    )
    old_ds = str(base / "datasets/7scenes/7scenes_chess")
    old_out = str(base / "runs/reproduction/ACE/7scenes/chess/buffer8M_mmap")
    for group, scenes in {
        "apt1": ["kitchen", "living"],
        "apt2": ["bed", "kitchen", "living", "luke"],
        "office1": ["gates362", "gates381", "lounge", "manolis"],
        "office2": ["5a", "5b"],
    }.items():
        for scene in scenes:
            name = "12scenes_" + group + "_" + scene
            ds = base / "datasets/12scenes" / name
            out = base / "runs/reproduction/ACE/12scenes" / name / "buffer8M_mmap"
            tid = "R.ACE.12scenes." + group + "." + scene + ".train8M"
            train = copy.deepcopy(train_template)
            train.update(
                task_id=tid,
                dataset_id=name,
                model_dataset_id=name,
                depends_on=["R.ACE.12scenes." + group + ".prepare"],
                inputs=[str(ds / "acceptance.json")],
            )
            train["command"] = [
                c.replace(old_ds, str(ds)).replace(old_out, str(out))
                for c in train["command"]
            ]
            for check in train["validation"]:
                check["path"] = check["path"].replace(old_out, str(out))
            tasks.append(train)
            evaluation = copy.deepcopy(eval_template)
            evaluation.update(
                task_id=tid + ".eval",
                dataset_id=name,
                model_dataset_id=name,
                depends_on=[tid],
                inputs=[str(out / "heads_local.pt")],
            )
            evaluation["command"] = [
                c.replace(old_ds, str(ds)).replace(old_out, str(out))
                for c in evaluation["command"]
            ]
            for check in evaluation["validation"]:
                check["path"] = check["path"].replace(old_out, str(out))
            tasks.append(evaluation)
            official = copy.deepcopy(evaluation)
            official.update(
                task_id="R.ACE.12scenes." + group + "." + scene + ".official.eval",
                depends_on=[
                    "R.ACE.12scenes.models.acquire",
                    "R.ACE.12scenes." + group + ".prepare",
                ],
                inputs=[str(modelroot / (name + ".pt"))],
                reproduction_level="Official Weights Evaluation",
            )
            official["command"] = [
                c.replace(
                    str(out / "heads_local.pt"), str(modelroot / (name + ".pt"))
                ).replace(
                    str(out / "official_evaluation"),
                    str(out.parent / "pretrained_official_evaluation"),
                )
                for c in official["command"]
            ]
            for check in official["validation"]:
                check["path"] = check["path"].replace(
                    str(out / "official_evaluation"),
                    str(out.parent / "pretrained_official_evaluation"),
                )
            tasks.append(official)
    # Preserve failed upstream-import attempts. New version fixes only compiled
    # DSAC discovery; data, heads and upstream evaluation script stay unchanged.
    for old in list(tasks):
        if not old["task_id"].startswith("R.ACE.") or not old["task_id"].endswith(
            ".eval"
        ):
            continue
        revised = copy.deepcopy(old)
        revised["task_id"] += ".v2"
        oldout = old["command"][-1]
        newout = oldout + "_v2"
        revised["command"][-1] = newout
        for check in revised["validation"]:
            check["path"] = check["path"].replace(oldout, newout)
        revised["config"] = {
            "runtime_fix": "Expose actual compiled dsacstar extension to unmodified upstream script through subprocess PYTHONPATH; preflight requires forward_rgb"
        }
        revised["priority"] = 113
        tasks.append(revised)
        old["blocked_reason"] = (
            "Superseded by explicit eval.v2 compiled-DSAC path repair; earlier failures/partial outputs retained, no retraining."
        )
    for s in tasks:
        if s["task_id"] == "macvo.demo.geometric":
            s["reproduction_level"] = "Independent Baseline"
    return tasks
