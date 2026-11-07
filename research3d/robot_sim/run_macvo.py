"""Official MAC-VO Paper configuration on complete synchronized Gazebo stereo.

Only the dataset adapter/checkpoint-runtime compatibility differs from upstream.
The official FlowFormerCov, TartanMotionNet and covariance-weighted ICP run in
FP32 on the local 6GB GPU. GT is a separate *post-inference* evaluate command.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image

from research3d.robot_sim.prepare_ace import (
    atomic_json, camera_checked, check_query_online, read_jsonl, safe_image, sha256,
)


def load_stereo_manifest(dataset):
    root = Path(dataset).resolve(strict=True)
    manifest = json.loads((root / "map_manifest.json").read_text())
    rows = read_jsonl(root / "query/frames.jsonl")
    camera = camera_checked(manifest["camera"])
    baseline = manifest.get("stereo_baseline_m", manifest.get("calibration", {}).get("stereo_baseline_m"))
    if baseline is None:
        baseline = manifest.get("stereo", {}).get("baseline_m")
    baseline = float(baseline) if baseline is not None else 0.0
    if not 0.01 <= baseline <= 1:
        raise ValueError("explicit calibrated stereo baseline required")
    stereo = manifest.get("calibration", manifest.get("stereo", {}))
    expected_extrinsic = np.eye(4)
    expected_extrinsic[0, 3] = baseline
    relative = np.asarray(stereo.get("T_left_right"), dtype=float)
    if stereo.get("rectified") is not True or relative.shape != (4, 4) or not np.allclose(relative, expected_extrinsic, atol=1e-6):
        raise ValueError("rectified right-camera pose in left optical frame must match positive baseline extrinsic")
    if "T_base_camera_optical" in manifest and "T_base_right_camera_optical" in stereo:
        derived = np.linalg.inv(np.asarray(manifest["T_base_camera_optical"], float)) @ np.asarray(stereo["T_base_right_camera_optical"], float)
        if not np.allclose(derived, relative, atol=1e-6):
            raise ValueError("base/camera and stereo extrinsics disagree")
    if len(rows) < 3:
        raise ValueError("need a complete trajectory of at least three stereo frames")
    if manifest.get("world_path") and sha256(root / manifest["world_path"]) != manifest["map_sha256"]:
        raise ValueError("world geometry checksum changed")
    ids, paths, stamps = set(), set(), []
    for row in rows:
        check_query_online(row)
        if row.get("map_id", manifest["map_id"]) != manifest["map_id"]:
            raise ValueError("query/map identity mismatch")
        if row["query_id"] in ids:
            raise ValueError("duplicate query id")
        ids.add(row["query_id"])
        stamp = row["timestamp_ns"]
        if not isinstance(stamp, int) or (stamps and stamp <= stamps[-1]):
            raise ValueError("nonmonotonic stereo timestamps")
        if row["right_timestamp_ns"] != stamp:
            raise ValueError("left/right timestamps are not exactly synchronized")
        if camera_checked(row.get("camera", camera)) != camera:
            raise ValueError("calibration changed during sequence")
        stamps.append(stamp)
        for field, checksum in (("image_path", "image_sha256"), ("right_image_path", "right_image_sha256")):
            p = safe_image(root, row[field])
            if p in paths:
                raise ValueError("stereo image path reused")
            paths.add(p)
            if sha256(p) != row[checksum]:
                raise ValueError("stereo image checksum mismatch")
            with Image.open(p) as im:
                im.load()
                if im.size != (camera["width"], camera["height"]):
                    raise ValueError("stereo image size/calibration mismatch")
    return root, manifest, rows, camera, baseline


def make_sequence(root, rows, camera, baseline):
    """Build exactly the upstream RGB tensor and StereoFrame contract; no GT."""
    import torch
    import pypose as pp
    from DataLoader import SequenceBase
    from DataLoader.Interface import StereoFrame, StereoData

    class GazeboStereoSequence(SequenceBase):
        def __init__(self):
            super().__init__(len(rows))

        @classmethod
        def is_valid_config(cls, config):
            raise ValueError("explicit frozen Gazebo manifest required")

        def __getitem__(self, local_index):
            row = rows[self.get_index(local_index)]
            images = []
            for field in ("image_path", "right_image_path"):
                with Image.open(safe_image(root, row[field])) as im:
                    image = np.asarray(im.convert("RGB")).copy()
                images.append(torch.from_numpy(image).float().permute(2, 0, 1)[None] / 255.0)
            stamp = row["timestamp_ns"]
            return StereoFrame(
                idx=[local_index], time_ns=[stamp], gt_pose=None,
                stereo=StereoData(T_BS=pp.identity_SE3(1),
                                  K=torch.tensor(camera["K"], dtype=torch.float32)[None],
                                  baseline=torch.tensor([baseline], dtype=torch.float32),
                                  time_ns=[stamp], width=camera["width"], height=camera["height"],
                                  imageL=images[0], imageR=images[1], gt_depth=None, gt_flow=None, flow_mask=None),
            )

    return GazeboStereoSequence()


def install_checkpoint_compat(out):
    """Same audited released-weight compatibility as the preserved benchmarks."""
    from Module.Network.FlowFormer.configs import submission
    from Module.Network.FlowFormerCov.flownet import FlowFormerCov
    original = submission.get_cfg

    def checkpoint_only():
        c = original()
        c.latentcostformer.pretrain = False
        return c

    submission.get_cfg = checkpoint_only

    def load_strict(self, weights):
        state = {k.removeprefix("module."): v for k, v in weights.items()}
        obsolete = {
            "memory_encoder.feat_encoder.svt.norm.weight", "memory_encoder.feat_encoder.svt.norm.bias",
            "context_encoder.svt.norm.weight", "context_encoder.svt.norm.bias",
            "memory_decoder.att.pos_emb.rel_ind", "memory_decoder.att.pos_emb.rel_height.weight",
            "memory_decoder.att.pos_emb.rel_width.weight",
        }
        expected = self.state_dict()
        extra, missing = set(state) - set(expected), set(expected) - set(state)
        if not extra <= obsolete or not missing <= {"memory_decoder.delta"}:
            raise ValueError("unsupported released-weight difference")
        if missing & dict(self.named_parameters()).keys():
            raise ValueError("missing learned parameters")
        for key in extra:
            del state[key]
        for key in missing:
            state[key] = expected[key]
        self.load_state_dict(state, strict=True)
        atomic_json(out / "checkpoint_audit.json", dict(removed_unused_keys=sorted(extra),
                    constructed_deterministic_buffers=sorted(missing), missing_learned_parameters=[], loaded_keys=len(state)))

    FlowFormerCov.load_ddp_state_dict = load_strict


def ned_to_optical(poses):
    C = np.eye(4)
    C[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    return C.T @ poses @ C


def infer(args):
    root, manifest, rows, camera, baseline = load_stereo_manifest(args.dataset)
    out, source = Path(args.output).resolve(), Path(args.source).resolve(strict=True)
    if out.exists():
        raise FileExistsError("refusing to overwrite existing MAC-VO run")
    out.mkdir(parents=True)
    sys.path.insert(0, str(source))
    import torch
    import pypose as pp
    from Utility.Config import load_config, asNamespace
    from Odometry.MACVO import MACVO
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    install_checkpoint_compat(out)
    _, cfg = load_config(source / "Config/Experiment/MACVO/Paper_Reproduce.yaml")
    cfg["Odometry"]["name"] = "Gazebo_Paper_FP32_full_sequence"
    cfg["Odometry"]["args"]["mapping"] = False
    cfg["Odometry"]["frontend"]["args"].update(weight=str(Path(args.weights).resolve()), enc_dtype="fp32", dec_dtype="fp32")
    cfg["Odometry"]["motion"]["args"]["weight"] = str(Path(args.motion_weights).resolve())
    protocol = dict(schema="robot_macvo_run_v1", map_id=manifest["map_id"], map_sha256=manifest["map_sha256"],
                    dataset_root=str(root), map_manifest_sha256=sha256(root / "map_manifest.json"),
                    online_manifest_sha256=sha256(root / "query/frames.jsonl"),
                    source_commit=subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
                    source_modules_sha256={name: sha256(source / name) for name in (
                        "Odometry/MACVO.py", "Module/MotionModel.py",
                        "Module/Optimization/TwoFramePGO/Optimizer.py", "Module/Optimization/TwoFramePGO/Graphs.py",
                        "Module/Network/FlowFormerCov/covhead.py")},
                    source_worktree_status=subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).splitlines(),
                    source_change_note="pre-existing covhead.py hardware compatibility retained; this adapter does not edit upstream source",
                    checkpoint_sha256=sha256(args.weights), motion_checkpoint_sha256=sha256(args.motion_weights),
                    config={"Odometry": cfg["Odometry"]}, seed=args.seed, frames=len(rows), full_captured_sequence=True,
                    preprocessing="native 640x480 RGB; no resize, frame clipping, GT loader or upstream dataset template instantiated",
                    timestamp_source="synchronized Gazebo sensor simulation clock", GT_online=False,
                    online_inputs="rectified stereo RGB + calibration + timestamps only", camera=camera, baseline_m=baseline,
                    pose_convention="T_odom_camera optical, initial arbitrary odometric gauge; no world/map anchoring",
                    output_policy="raw pre-interpolation trajectory and explicit lost mask; no test GT read",
                    paper_scope="new Gazebo environment adaptation; not original paper benchmark",
                    dense_mapping=False, covariance_scope="point covariance; not calibrated pose covariance")
    atomic_json(out / "protocol.json", protocol)
    seq = make_sequence(root, rows, camera, baseline)
    model, timings, capture = None, [], {}
    started = time.perf_counter()
    try:
        model = MACVO.from_config(asNamespace({"Odometry": cfg["Odometry"]}))
        initialization_ms = (time.perf_counter() - started) * 1000
        original_refiner = model.MapRefiner.elaborate_map

        def preserve_raw(frames):
            capture["raw"] = pp.SE3(frames.data["pose"].tensor).matrix().numpy().copy()
            capture["lost"] = frames.data["need_interp"].tensor.numpy().copy()
            return original_refiner(frames)

        model.MapRefiner.elaborate_map = preserve_raw
        torch.cuda.reset_peak_memory_stats()
        for index, frame in enumerate(seq):
            if frame.gt_pose is not None or frame.stereo.gt_depth is not None or frame.stereo.gt_flow is not None:
                raise AssertionError("supervision leaked into MAC-VO")
            torch.cuda.synchronize()
            t = time.perf_counter()
            model.run(frame)
            torch.cuda.synchronize()
            timings.append(dict(query_id=rows[index]["query_id"], timestamp_ns=rows[index]["timestamp_ns"],
                                inference_ms=(time.perf_counter() - t) * 1000,
                                wall_since_start_ms=(time.perf_counter() - started) * 1000,
                                matches_total=len(model.graph.match)))
            if index % 10 == 0:
                atomic_json(out / "progress.json", dict(status="RUNNING", frames=index + 1, total=len(rows)))
                print(f"MACVO {index + 1}/{len(rows)} {timings[-1]['inference_ms']:.1f}ms", flush=True)
        model.terminate()
        raw = ned_to_optical(capture["raw"])
        lost = capture["lost"].reshape(-1).astype(bool)
        if len(raw) != len(rows) or len(lost) != len(rows):
            raise ValueError("MAC-VO output count mismatch")
        if not np.isfinite(raw).all():
            raise ValueError("nonfinite raw pose; do not interpolate/drop failures")
        if sha256(root / "query/frames.jsonl") != protocol["online_manifest_sha256"]:
            raise ValueError("online capture changed during inference")
        temp = out / "trajectory.partial.npz"
        np.savez_compressed(temp, T_odom_camera=raw, lost=lost,
                            timestamp_ns=np.array([r["timestamp_ns"] for r in rows], dtype=np.int64),
                            query_ids=np.array([r["query_id"] for r in rows]))
        os.replace(temp, out / "trajectory.npz")
        with (out / "predictions.jsonl.partial").open("w") as f:
            for i, row in enumerate(rows):
                f.write(json.dumps(dict(query_id=row["query_id"], timestamp_ns=row["timestamp_ns"],
                                       source="official_MACVO_Paper_FP32", status="lost" if lost[i] else "estimated",
                                       T_odom_camera=raw[i].tolist(), covariance_se3=None,
                                       inference_latency_ms=timings[i]["inference_ms"],
                                       map_id=manifest["map_id"], map_sha256=manifest["map_sha256"]), allow_nan=False) + "\n")
        os.replace(out / "predictions.jsonl.partial", out / "predictions.jsonl")
        atomic_json(out / "per_frame.json", timings)
        summary = dict(status="INFERENCE_COMPLETE_NOT_EVALUATED", frames=len(rows), full_captured_sequence=True,
                       lost_frames=int(lost[1:].sum()), GT_online=False, raw_before_interpolation=True,
                       wall_seconds=time.perf_counter() - started,
                       model_initialization_ms=initialization_ms,
                       runtime_mean_ms=float(np.mean([t["inference_ms"] for t in timings])),
                       runtime_p95_ms=float(np.percentile([t["inference_ms"] for t in timings], 95)),
                       torch_peak_allocated_MiB=torch.cuda.max_memory_allocated() / 1024**2,
                       trajectory_sha256=sha256(out / "trajectory.npz"))
        atomic_json(out / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        return summary
    except Exception as exc:
        atomic_json(out / "failure.json", dict(status="FAILED", error_type=type(exc).__name__, error=str(exc),
                    completed_frames=len(timings), traceback=traceback.format_exc()))
        if model is not None:
            try:
                model.Optimizer.terminate()
            except Exception:
                pass
        raise


def evaluate_run(run, ground_truth):
    """Separate CPU process; offline truth cannot affect saved predictions."""
    from research3d.motion.metrics import evaluate
    from research3d.knownmap.contracts import se3
    run = Path(run)
    protocol = json.loads((run / "protocol.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    if sha256(run / "trajectory.npz") != summary["trajectory_sha256"]:
        raise ValueError("trajectory changed before independent evaluation")
    with np.load(run / "trajectory.npz", allow_pickle=False) as z:
        est, lost, ids, times = z["T_odom_camera"], z["lost"], z["query_ids"], z["timestamp_ns"]
    truths = read_jsonl(ground_truth)
    gt = {r["query_id"]: r for r in truths}
    if len(gt) != len(truths) or set(gt) != set(ids.tolist()):
        raise ValueError("GT/query identity mismatch (no dropping unmatched frames)")
    matrices = []
    for qid, stamp in zip(ids, times):
        r = gt[str(qid)]
        if int(r["timestamp_ns"]) != int(stamp) or r.get("map_id", protocol["map_id"]) != protocol["map_id"]:
            raise ValueError("GT timestamp/map mismatch")
        matrices.append(se3(r["T_map_camera"]))
    metrics = evaluate(est, np.asarray(matrices), lost)
    intervals = np.diff(times.astype(np.float64)) / 1e9
    lost_duration = float(intervals[lost[1:].astype(bool)].sum())
    current_run = longest_run = 0.0
    for dt, bad in zip(intervals, lost[1:]):
        current_run = current_run + float(dt) if bad else 0.0
        longest_run = max(longest_run, current_run)
    metrics.update(status="EVALUATED", source="independent CPU SE3 metrics on raw MAC-VO output",
                   GT_online=False, full_captured_sequence=True, trajectory_sha256=summary["trajectory_sha256"],
                   offline_gt_sha256=sha256(ground_truth), duration_sim_seconds=float(times[-1] - times[0]) / 1e9,
                   timestamp_note="real Gazebo simulation timestamps; no physical hardware time claim",
                   input_sampling=dict(median_interval_s=float(np.median(intervals)),
                                       max_interval_s=float(intervals.max()),
                                       recorded_intervals_over_1_5_median=int((intervals > 1.5 * np.median(intervals)).sum()),
                                       observed_rate_hz=len(intervals) / float(intervals.sum()),
                                       scope="capture gaps are retained; these are separate from algorithm lost frames"),
                   failure_duration_seconds=lost_duration, max_failure_run_seconds=longest_run,
                   time_weighted_tracking_availability=1.0 - lost_duration / float(intervals.sum()),
                   world_absolute_localization=False, alignment_warning="GT rigid alignment only for VO evaluation, never online localization")
    atomic_json(run / "metrics.json", metrics)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--dataset", required=True)
    i = sub.add_parser("infer")
    i.add_argument("--dataset", required=True)
    i.add_argument("--output", required=True)
    i.add_argument("--source", default="/data/fyp_3d/sources/MAC-VO")
    i.add_argument("--weights", default="/data/fyp_3d/models/MACVO_FrontendCov.pth")
    i.add_argument("--motion-weights", default="/data/fyp_3d/models/MACVO_posenet.pkl")
    i.add_argument("--seed", type=int, default=1305)
    e = sub.add_parser("evaluate")
    e.add_argument("--run", required=True)
    e.add_argument("--ground-truth", required=True)
    args = p.parse_args()
    if args.action == "infer":
        infer(args)
    elif args.action == "evaluate":
        print(json.dumps(evaluate_run(args.run, args.ground_truth), indent=2))
    else:
        _, manifest, rows, camera, baseline = load_stereo_manifest(args.dataset)
        print(json.dumps(dict(status="STEREO_INPUT_VERIFIED", frames=len(rows), camera=camera,
                              baseline_m=baseline, map_id=manifest["map_id"]), indent=2))


if __name__ == "__main__":
    main()
