"""Official MAC-VO modules with GT-free observations, mapping disabled, honest failures.
Main algorithm modules are upstream. Instrumentation exports correspondences and covariance.
"""

import argparse, hashlib, json, os, random, resource, sys, time, traceback, subprocess
from pathlib import Path
from macvo_instrumentation import install_from_environment,StageProfiler
import numpy as np


def main(a):
    import torch, pypose as pp

    source = Path(a.source).resolve()
    root = Path(a.dataset).resolve()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))
    torch.set_num_threads(4)
    os.environ['FYP_MACVO_AUDIT']=json.dumps(dict(source=str(source),output=str(out),mode=a.backend_covariance,capture_graphs=a.capture_graphs))
    install_from_environment()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    from Utility.Config import load_config, asNamespace
    from DataLoader import SequenceBase, smart_transform
    from Odometry.MACVO import MACVO
    from motion.metrics import evaluate
    from Module.Network.FlowFormer.configs import submission
    from Module.Network.FlowFormerCov.flownet import FlowFormerCov

    original_config = submission.get_cfg

    def checkpoint_only_config():
        c = original_config()
        c.latentcostformer.pretrain = False
        return c

    submission.get_cfg = checkpoint_only_config

    def strict_checkpoint(self, weights):
        state = {k.removeprefix("module."): v for k, v in weights.items()}
        # Current upstream deletes unused Twins classification norms and GMA positional
        # embeddings. Its new delta buffer is a deterministic 9x9 sampling grid.
        obsolete = {
            "memory_encoder.feat_encoder.svt.norm.weight",
            "memory_encoder.feat_encoder.svt.norm.bias",
            "context_encoder.svt.norm.weight",
            "context_encoder.svt.norm.bias",
            "memory_decoder.att.pos_emb.rel_ind",
            "memory_decoder.att.pos_emb.rel_height.weight",
            "memory_decoder.att.pos_emb.rel_width.weight",
        }
        expected = self.state_dict()
        extra = set(state) - set(expected)
        missing = set(expected) - set(state)
        assert extra <= obsolete and missing <= {"memory_decoder.delta"}, (
            extra,
            missing,
        )
        assert not (
            missing & dict(self.named_parameters()).keys()
        ), "Missing learned parameter"
        for k in extra:
            del state[k]
        for k in missing:
            state[k] = expected[k]
        self.load_state_dict(state, strict=True)
        (out / "checkpoint_audit.json").write_text(
            json.dumps(
                dict(
                    removed_unused_keys=sorted(extra),
                    constructed_deterministic_buffers=sorted(missing),
                    missing_learned_parameters=[],
                    loaded_keys=len(state),
                ),
                indent=2,
            )
        )

    FlowFormerCov.load_ddp_state_dict = strict_checkpoint
    config_name = "Paper_Reproduce.yaml" if a.config == "paper" else "MACVO_Fast.yaml"
    cfg, cfg_dict = load_config(source / "Config/Experiment/MACVO" / config_name)
    if a.config == "paper":
        if not a.motion_weights:
            raise ValueError("Paper config requires official --motion-weights")
        cfg_dict["Odometry"]["motion"]["args"]["weight"] = str(
            Path(a.motion_weights).resolve()
        )
    cfg_dict["Odometry"]["name"] = (
        "MACVO_" + a.config + "_" + a.decoder + "_cov_weights"
    )
    cfg_dict["Odometry"]["args"]["mapping"] = False
    cfg_dict["Odometry"]["frontend"]["args"].update(
        weight=str(Path(a.weights).resolve()), dec_dtype=a.decoder, enc_dtype=a.encoder
    )
    # Retain the official parallel optimizer; no direct GT-based motion prior.
    datacfg = {
        "type": "TartanAir_NoIMU",
        "name": "full_original_TartanAir_trajectory",
        "args": dict(
            root=str(root), gtDepth=False, gtFlow=False, gtPose=False, compressed=True
        ),
    }
    if a.dataset_kind == 'euroc':
        datacfg = dict(type='EuRoC_NoIMU',name=root.parent.name,args=dict(root=str(root),gt_pose=False))
    metadata = dict(
        source_commit=subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        config=cfg_dict,
        data=datacfg,
        frame_limit=a.frames,
        seed=a.seed,
        online_inputs="left/right RGB only, fixed intrinsics/baseline",
        gt_online=False,
        dense_mapping=False,
        decoder=a.decoder,
        optimizer_weighting=(
            "official 3D matching covariance"
            if a.config == "paper"
            else "official pixel/disparity covariance"
        ),
        experiment_config=config_name,
        encoder=a.encoder,
        timestamps_available=(root / "imu/cam_time.npy").exists() or a.dataset_kind == "euroc",
        physical_timestamps=(a.dataset_kind == "euroc"),
        timestamp_caveat="TartanAir/Habitat synthetic sequences: supplied timestamps are simulation time; upstream assumes 10Hz if absent. No physical IMU/wheel or real sensor-time fusion",
        checkpoint_loading="strict state_dict; no redundant ImageNet download",
        backend_covariance=a.backend_covariance,
        optimizer_initialization='Actual live official TartanMotionNet; no static initialization replay',
        profiling='Nonblocking per-stage CUDA events; no per-frame global synchronize; full pipeline wall through final optimization drain',
        precision_variant='Separate hardware-adapted experiment, not replacing FP32 Paper baseline',
        instrumentation_sha256=hashlib.sha256(Path(__file__).with_name('macvo_instrumentation.py').read_bytes()).hexdigest(),
        covariance_scope="3D point observation covariance and pixel/disparity residual covariance; NOT a calibrated pose covariance",
    )
    (out / "protocol.json").write_text(json.dumps(metadata, indent=2))
    base_sequence = SequenceBase.instantiate(datacfg["type"], datacfg["args"])
    if (root / "calibration.json").exists():
        calibration = json.loads((root / "calibration.json").read_text())
        assert (calibration["width"], calibration["height"]) == (640, 480)
        base_sequence.lcam_K = torch.tensor(
            calibration["K"], dtype=torch.float32
        ).unsqueeze(0)
        base_sequence.baseline = float(calibration["baseline"])
        metadata["explicit_camera_calibration"] = calibration
        (out / "protocol.json").write_text(json.dumps(metadata, indent=2))
    assert a.frames == 0, "Formal full-sequence wrapper does not accept a frame cap"
    metadata["full_source_frames"] = len(base_sequence)
    metadata["frame_limit"] = None
    (out / "protocol.json").write_text(json.dumps(metadata, indent=2))
    seq = smart_transform(base_sequence, cfg.Preprocess)
    model = None
    rows = []
    start = time.perf_counter()
    captures = {}
    try:
        model = MACVO.from_config(asNamespace({"Odometry": cfg_dict["Odometry"]}))
        cache_counts = None
        profiler=StageProfiler()
        for obj,name,label,gpu in [
            (model.Frontend,'estimate_pair','frontend_pair',True),
            (model.Frontend,'estimate_depth','frontend_depth',True),
            (model.MotionEstimator,'predict','motion_prior',True),
            (model.KeypointSelector,'select_point','keypoint_selection',True),
            (model.Optimizer,'write_map','optimizer_wait_and_writeback',False),
            (model.Optimizer,'get_graph_data','build_solver_input',False)]:
            profiler.wrap(obj,name,label,gpu)
        if a.frontend_cache:
            from motion.frontend_cache import install_cache

            namespace = dict(
                frontend=cfg_dict["Odometry"]["frontend"],
                source_commit=metadata["source_commit"],
                weight_sha256=hashlib.sha256(Path(a.weights).read_bytes()).hexdigest(),
                compat_patch_sha256=hashlib.sha256(
                    (source / "Module/Network/FlowFormerCov/covhead.py").read_bytes()
                ).hexdigest(),
            )
            cache_counts = install_cache(model.Frontend, a.frontend_cache, namespace)

        diagnostics = []

        def tensor_stats(x):
            if x is None:
                return None
            x = x.detach().float()
            v = x[torch.isfinite(x)]
            return dict(
                shape=list(x.shape),
                finite_fraction=float(torch.isfinite(x).float().mean()),
                minimum=float(v.min()) if v.numel() else None,
                maximum=float(v.max()) if v.numel() else None,
                median=float(v.median()) if v.numel() else None,
            )

        original_filter = model.OutlierFilter.filter

        def audit_filter(values, device):
            mask = original_filter(values, device)
            if len(diagnostics) < 10:
                diagnostics.append(
                    dict(
                        input_count=len(values),
                        accepted=int(mask.sum()),
                        fields={k: tensor_stats(v) for k, v in values.data.items()},
                    )
                )
                (out / "observation_audit.json").write_text(
                    json.dumps(diagnostics, indent=2)
                )
            return mask

        model.OutlierFilter.filter = audit_filter
        refiner = model.MapRefiner.elaborate_map

        def capture_raw(frames):
            captures["raw_pose_ned"] = (
                pp.SE3(frames.data["pose"].tensor).matrix().numpy().copy()
            )
            captures["raw_lost"] = frames.data["need_interp"].tensor.numpy().copy()
            return refiner(frames)

        model.MapRefiner.elaborate_map = capture_raw
        torch.cuda.reset_peak_memory_stats()
        pipeline_start=time.perf_counter()
        for i, frame in enumerate(seq):
            assert (
                frame.gt_pose is None
                and frame.stereo.gt_depth is None
                and frame.stereo.gt_flow is None
            )
            t = time.perf_counter()
            model.run(frame)
            rows.append(
                dict(
                    frame=i,
                    runtime_ms=(time.perf_counter() - t) * 1000,
                    matches_total=len(model.graph.match),
                )
            )
            if i % 10 == 0:
                print(
                    "FRAME",
                    i,
                    "ms",
                    round(rows[-1]["runtime_ms"], 2),
                    "matches",
                    rows[-1]["matches_total"],
                    flush=True,
                )
        model.terminate()
        torch.cuda.synchronize()
        pipeline_seconds=time.perf_counter()-pipeline_start
        stages=profiler.finish()
        solver_rows=[]
        for log in out.glob('solver_audit.*.jsonl'):
            solver_rows.extend(json.loads(line) for line in log.read_text().splitlines())
        unique={row['frame']:row for row in solver_rows}
        assert unique, 'Actual optimizer process did not execute audited weighting'
        assert all(row['mode']==a.backend_covariance and not row['GT_read'] for row in unique.values())
        (out/'stage_profile.json').write_text(json.dumps(dict(stages=stages,
            inference_pipeline_seconds=pipeline_seconds,inference_frames=len(rows),
            inference_pipeline_mean_ms=1000*pipeline_seconds/len(rows),
            initialization_and_final_optimizer_drain_included=True,
            per_frame_global_sync=False,solver_jobs=len(unique),
            solver_mean_ms=1000*sum(r['wall_seconds'] for r in unique.values())/len(unique),
            solver_pids=sorted({r['pid'] for r in unique.values()}),
            cpu_gpu_spans_overlap_not_additive=True),indent=2))
        pose = pp.SE3(model.graph.frames.data["pose"].tensor).matrix().numpy()
        raw = captures["raw_pose_ned"]
        lost = captures["raw_lost"]
        assert len(pose) == len(seq) == len(rows)
        # Ground-truth is loaded only AFTER all online estimates and postprocessing finish.
        gt_valid = np.ones(len(pose), dtype=bool)
        if a.dataset_kind == 'euroc':
            from DataLoader.Dataset.EuRoC import load_EurocGTPose
            times = base_sequence.cam_timestamps.numpy()
            offline_gt, gt_valid = load_EurocGTPose(root / 'state_groundtruth_estimate0/data.csv', times)
            gt = np.full_like(pose, np.nan)
            gt[gt_valid] = offline_gt.matrix().numpy()
            C = np.eye(4)  # EuRoC upstream graph pose is the body, not optical camera.
        else:
            gt = pp.SE3(torch.from_numpy(np.loadtxt(root / 'pose_left.txt')[:len(pose)])).matrix().numpy()
            C = np.eye(4)
            C[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        cv = C.T @ pose @ C
        raw_cv = C.T @ raw @ C
        gt_cv = C.T @ gt @ C
        np.savez_compressed(
            out / "trajectory.npz",
            T_world_camera=cv,
            T_world_camera_raw=raw_cv,
            T_world_camera_gt=gt_cv,
            lost_before_interpolation=lost,
            frame_idx=np.arange(len(pose)),
            gt_valid=gt_valid,
        )
        np.savez_compressed(out / "observations.npz", **model.graph.serialize())
        np.savez_compressed(
            out / "relative_motion.npz",
            T_previous_current=np.linalg.inv(raw_cv[:-1]) @ raw_cv[1:],
            frame_previous=np.arange(len(pose) - 1),
            frame_current=np.arange(1, len(pose)),
        )
        (out / "per_frame.json").write_text(json.dumps(rows, indent=2))
        metrics = dict(
            raw=evaluate(raw[gt_valid], gt[gt_valid], lost[gt_valid]),
            interpolated=evaluate(pose[gt_valid], gt[gt_valid], lost[gt_valid]),
            runtime_mean_ms=float(np.mean([r["runtime_ms"] for r in rows])),
            runtime_median_ms=float(np.median([r["runtime_ms"] for r in rows])),
            runtime_p95_ms=float(np.percentile([r["runtime_ms"] for r in rows], 95)),
            wall_seconds=time.perf_counter() - start,
            torch_peak_allocated_MiB=torch.cuda.max_memory_allocated() / 1024**2,
            torch_peak_reserved_MiB=torch.cuda.max_memory_reserved() / 1024**2,
            maxrss_parent_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            frontend_cache=cache_counts,
            initialization_included=True,
            finished_frames=len(rows),
        )
        (out / "summary.json").write_text(json.dumps(metrics, indent=2))
        print("RESULT", metrics, flush=True)
    except Exception as exc:
        failure = dict(
            error_type=type(exc).__name__,
            message=str(exc),
            completed_frames=len(rows),
            wall_seconds=time.perf_counter() - start,
            traceback=traceback.format_exc(),
        )
        (out / "failure.json").write_text(json.dumps(failure, indent=2))
        print(traceback.format_exc(), flush=True)
        if model is not None:
            try:
                model.Optimizer.terminate()
            except Exception:
                pass
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("output")
    p.add_argument("--source", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--decoder", choices=["bf16", "fp16", "fp32"], default="fp32")
    p.add_argument("--encoder", choices=["bf16", "fp16", "fp32"], default="fp32")
    p.add_argument("--config", choices=["fast", "paper"], default="fast")
    p.add_argument("--motion-weights")
    p.add_argument("--frontend-cache")
    p.add_argument('--backend-covariance',choices=['original','identity'],default='original')
    p.add_argument('--capture-graphs',action='store_true')
    p.add_argument("--seed", type=int, default=1305)
    p.add_argument('--dataset-kind', choices=['tartan','euroc'], default='tartan')
    main(p.parse_args())
