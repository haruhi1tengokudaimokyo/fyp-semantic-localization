"""Persistent official ACE + DSAC* RPC for a single, explicitly bound map.

stdin/stdout carry JSON lines; native/Python model logs go to stderr. Online
inputs are RGB, calibration, timestamp and map identity only. A Python audit
guard additionally rejects dataset-file opens except the active RGB. This is
testable leak prevention, not an OS security sandbox or a pose covariance model.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

from research3d.knownmap.contracts import Camera, SensorObservation
from research3d.robot_sim.prepare_ace import camera_checked, sha256


class DatasetReadGuard:
    def __init__(self, roots):
        self.roots = tuple(Path(p).resolve() for p in roots)
        self.allowed = None

    def __call__(self, event, args):
        if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(root) for root in self.roots):
            if path != self.allowed:
                raise PermissionError("online ACE cannot open dataset files other than the active RGB")
            flags = args[2] if len(args) > 2 else 0
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                raise PermissionError("online ACE image access is read-only")


class Worker:
    def __init__(self, binding, online_root, predictor, install_guard=True):
        self.binding = dict(binding)
        self.online_root = Path(online_root).resolve(strict=True)
        self.predictor = predictor
        self.last_stamp = -1
        self.camera = camera_checked(binding["camera"])
        roots = [self.online_root]
        if binding.get("dataset_root"):
            roots.append(binding["dataset_root"])
        self.guard = DatasetReadGuard(roots)
        if install_guard:
            sys.addaudithook(self.guard)

    def localize(self, request):
        start = time.perf_counter()
        allowed = {"op", "request_id", "query_id", "map_id", "map_sha256", "timestamp_ns", "image_path", "camera", "metadata"}
        if set(request) - allowed:
            raise ValueError("unexpected input fields; poses/GT/labels are not online observations")
        if request.get("op") != "localize" or request["map_id"] != self.binding["map_id"]:
            raise ValueError("operation or map identity mismatch")
        if request.get("map_sha256", self.binding["map_sha256"]) != self.binding["map_sha256"]:
            raise ValueError("map geometry checksum mismatch")
        meta = request.get("metadata", {})
        if not isinstance(meta, dict) or set(meta) - {"sequence_id", "sensor_frame"}:
            raise ValueError("unsupported metadata (no GT, event labels or hidden state)")
        stamp = request["timestamp_ns"]
        if not isinstance(stamp, int) or isinstance(stamp, bool) or stamp < 0 or stamp <= self.last_stamp:
            raise ValueError("timestamps must strictly increase")
        cam = camera_checked(request["camera"])
        if cam != self.camera:
            raise ValueError("query calibration differs from trained-map camera")
        path = Path(request["image_path"]).resolve(strict=True)
        if not path.is_relative_to(self.online_root) or path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("query must be an RGB image inside online-root")
        if path.stat().st_size > 16 * 1024**2:
            raise ValueError("query image exceeds pilot size limit")
        self.guard.allowed = path
        try:
            digest = sha256(path)
            observation = SensorObservation(str(request["query_id"]), Camera(**cam), path, stamp)
            state, correspondences = self.predictor.predict(observation)
            self.last_stamp = stamp
            diag = state.diagnostics
            good = diag.status == "estimated" and diag.inliers >= 4 and bool(state.hypotheses)
            T = state.hypotheses[0].T_world_camera.tolist() if good else None
            repro = float(diag.median_reprojection_px)
            errors = [] if correspondences is None else [float(x) for x in correspondences.reprojection_error_px]
            geometric_inliers = [x for x in errors if math.isfinite(x) and x < 10.0]
            return dict(
                status="ok" if good else "failed", source="official_ace_dsacstar",
                request_id=request.get("request_id"), query_id=request["query_id"], map_id=request["map_id"],
                timestamp_ns=stamp, stamp_sec=stamp / 1e9, T_map_camera_optical=T, T_map_camera=T,
                inliers=int(diag.inliers), reprojection_error_px=repro if math.isfinite(repro) else None,
                reprojection_statistic="all_valid_scene_coordinates_median",
                reprojection_error_scope="median over all finite positive-depth predictions, not only DSAC inliers",
                reprojection_under_10px_median_px=statistics.median(geometric_inliers) if geometric_inliers else None,
                reprojection_under_10px_fraction=len(geometric_inliers) / len(errors) if errors else None,
                reprojection_truncation_threshold_px=10.0,
                latency_ms=(time.perf_counter() - start) * 1000, algorithm_latency_ms=float(diag.runtime_ms),
                image_sha256=digest, head_sha256=self.binding["head_sha256"], map_sha256=self.binding["map_sha256"],
                covariance_se3=None, accepted=False,
                failure_reason=None if good else "official DSAC did not produce a valid estimate",
            )
        finally:
            self.guard.allowed = None


def load_binding(path, source):
    binding = json.loads(Path(path).read_text())
    if binding.get("schema") != "robot_ace_model_v1":
        raise ValueError("missing scene-specific model binding")
    if binding.get("status") not in {"TRAINED_NOT_LOCALIZATION_VALIDATED", "LOCALIZATION_VALIDATED"}:
        raise ValueError("binding is not a completed training artifact")
    if sha256(binding["head_path"]) != binding["head_sha256"]:
        raise ValueError("ACE head hash mismatch")
    if sha256(Path(source) / "ace_encoder_pretrained.pt") != binding["encoder_sha256"]:
        raise ValueError("ACE encoder hash mismatch")
    if sha256(Path(source) / "ace_trainer.py") != binding["trainer_sha256"]:
        raise ValueError("ACE source differs from trained-head provenance")
    return binding


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binding", required=True)
    p.add_argument("--online-root", required=True)
    p.add_argument("--source", required=True)
    args = p.parse_args()
    # Also route C++/DSAC stdout to stderr; preserve a private descriptor for RPC.
    rpc = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def emit(value):
        rpc.write(json.dumps(value, allow_nan=False) + "\n")
        rpc.flush()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            from research3d.knownmap.ace import ACE
            binding = load_binding(args.binding, args.source)
            predictor = ACE(args.source, binding["head_path"])
            worker = Worker(binding, args.online_root, predictor)
        emit(dict(event="ready", protocol="robot_ace_rpc_v1", map_id=binding["map_id"],
                  map_sha256=binding["map_sha256"], head_sha256=binding["head_sha256"]))
    except Exception as e:
        emit(dict(event="fatal", error_type=type(e).__name__, error=str(e)))
        return 2
    for line in sys.stdin:
        request = {}
        try:
            if len(line) > 65536:
                raise ValueError("RPC request exceeds 64KiB")
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            if request.get("op") == "ping":
                emit(dict(event="pong"))
                continue
            if request.get("op") == "shutdown":
                emit(dict(event="bye"))
                return 0
            with contextlib.redirect_stdout(sys.stderr):
                response = worker.localize(request)
            emit(response)
        except Exception as e:
            emit(dict(status="error", request_id=request.get("request_id") if isinstance(request, dict) else None,
                      error_type=type(e).__name__, failure_reason=str(e)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
