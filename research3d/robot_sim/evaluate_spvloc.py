"""Offline fixed-map SPVLoc scoring; query truth is never imported online.

Ranks are taken exactly as emitted. Top-3 oracle recall is a candidate-pool
diagnostic, never an achieved pose-selection success rate. Missing and invalid
predictions remain failures in every recall denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from research3d.fusion.contracts import discrepancy, read_jsonl, sha256_file, write_json, write_jsonl


THRESHOLDS = {"5cm5deg": (.05, 5.), "25cm10deg": (.25, 10.), "1m30deg": (1., 30.)}


def unique(rows, description):
    result = {}
    for row in rows:
        key = row["query_id"]
        if key in result:
            raise ValueError(f"duplicate {description} query_id: {key}")
        result[key] = row
    return result


def error_or_reason(value, truth):
    try:
        return list(discrepancy(value, truth)), None
    except (ValueError, TypeError) as exc:
        return None, f"invalid_pose: {exc}"


def summarize(errors, denominator):
    values = np.asarray([x for x in errors if x is not None], dtype=float).reshape(-1, 2)
    result = dict(evaluated=denominator, valid=len(values), missing_or_invalid=denominator-len(values),
                  coverage=len(values)/denominator, translation_rmse_m=None, translation_median_m=None,
                  translation_p95_m=None, rotation_median_deg=None, rotation_p95_deg=None)
    if len(values):
        result.update(translation_rmse_m=float(np.sqrt(np.mean(values[:, 0]**2))),
                      translation_median_m=float(np.median(values[:, 0])),
                      translation_p95_m=float(np.quantile(values[:, 0], .95)),
                      rotation_median_deg=float(np.median(values[:, 1])),
                      rotation_p95_deg=float(np.quantile(values[:, 1], .95)))
    for key, (translation, rotation) in THRESHOLDS.items():
        count = int(sum(e[0] <= translation and e[1] <= rotation for e in values))
        result[f"success_count_{key}"] = count
        result[f"success_{key}"] = count / denominator
    return result


def score_rows(frames, truths, predictions, map_id, map_sha256):
    frames, truths, predictions = unique(frames, "input"), unique(truths, "truth"), unique(predictions, "prediction")
    if not frames or set(frames) != set(truths):
        raise ValueError("nonempty query input and offline truth IDs must match exactly")
    if set(predictions) - set(frames):
        raise ValueError("predictions contain a query outside the declared sequence")
    rows = []
    latencies = []
    for key, frame in frames.items():
        gt = truths[key]
        if frame["map_id"] != map_id or frame["timestamp_ns"] != gt["timestamp_ns"]:
            raise ValueError("input/truth map or timestamp association mismatch")
        prediction = predictions.get(key)
        row = dict(query_id=key, timestamp_ns=frame["timestamp_ns"],
                   top1_error=None, refined_top1_error=None, candidate_errors=[],
                   top1_failure=None, refined_top1_failure=None)
        if prediction is None:
            row["top1_failure"] = row["refined_top1_failure"] = "missing_prediction"
        else:
            if (prediction["map_id"] != map_id or prediction["map_sha256"] != map_sha256
                    or prediction["timestamp_ns"] != frame["timestamp_ns"]
                    or prediction["image_sha256"] != frame["image_sha256"]):
                raise ValueError(f"prediction identity/calibration binding mismatch: {key}")
            if prediction.get("online_gt_used") is not False:
                raise ValueError("prediction lacks explicit GT-free input declaration")
            latency = prediction.get("latency_s")
            if latency is not None and np.isfinite(latency) and latency >= 0:
                latencies.append(latency)
            if prediction["status"] != "ok":
                row["top1_failure"] = row["refined_top1_failure"] = "inference_" + prediction["status"]
            else:
                pool = prediction["candidates"]
                if len(pool) > 3 or [c["rank"] for c in pool] != list(range(1, len(pool)+1)):
                    raise ValueError("expected emitted ordered Top-3 ranks; do not re-rank using truth")
                for candidate in pool:
                    error, reason = error_or_reason(candidate["T_map_camera_optical"], gt["T_map_camera"])
                    row["candidate_errors"].append(dict(rank=candidate["rank"], error=error, failure=reason,
                                                       viewport_score=candidate["viewport_score"]))
                if pool:
                    row["top1_error"] = row["candidate_errors"][0]["error"]
                    row["top1_failure"] = row["candidate_errors"][0]["failure"]
                else:
                    row["top1_failure"] = "no_candidate"
                refined = prediction.get("refined_top1_T_map_camera_optical")
                if refined is None:
                    row["refined_top1_failure"] = "missing_refined_prediction"
                else:
                    row["refined_top1_error"], row["refined_top1_failure"] = error_or_reason(refined, gt["T_map_camera"])
        row["top3_candidate_recall"] = {
            name: any(c["error"] is not None and c["error"][0] <= t and c["error"][1] <= r
                      for c in row["candidate_errors"])
            for name, (t, r) in THRESHOLDS.items()}
        for kind in ("top1", "refined_top1"):
            error = row[kind+"_error"]
            row[kind+"_success"] = {name: error is not None and error[0] <= t and error[1] <= r
                                    for name, (t, r) in THRESHOLDS.items()}
        rows.append(row)
    total = len(rows)
    report = dict(status="EVALUATED_FIXED_MAP_GAZEBO_DEVELOPMENT_PILOT", evaluated=total,
                  map_id=map_id, map_sha256=map_sha256, alignment="none; original fixed map camera optical frame",
                  ground_truth_usage="offline only; exact query_id and timestamp_ns association",
                  test_tuning=False, top1=summarize([r["top1_error"] for r in rows], total),
                  refined_top1=summarize([r["refined_top1_error"] for r in rows], total),
                  top3_oracle_candidate_recall={name: sum(r["top3_candidate_recall"][name] for r in rows)/total
                                               for name in THRESHOLDS},
                  top3_scope="Any one unrefined candidate satisfying BOTH thresholds. Requires oracle selection; not actual localization success.",
                  failure_counts={name: dict(Counter(r[name+"_failure"] for r in rows if r[name+"_failure"] is not None))
                                  for name in ("top1", "refined_top1")},
                  failure_counts_scope="Inference/data/pose-validity failures only; geometrically wrong finite predictions are counted by success thresholds and per-frame success flags.",
                  latency_s=None if not latencies else dict(p50=float(np.median(latencies)), p95=float(np.quantile(latencies, .95))),
                  generalization_scope="scene03000 was a checkpoint-selection validation building; cross-renderer development only, not unseen-building evidence")
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.dataset/"map_manifest.json").read_text())
    frame_path, gt_path = args.dataset/"query/frames.jsonl", args.dataset/"offline_gt/query_poses.jsonl"
    report, rows = score_rows(read_jsonl(frame_path), read_jsonl(gt_path), read_jsonl(args.predictions),
                              manifest["map_id"], manifest["map_sha256"])
    report["input_sha256"] = {"frames": sha256_file(frame_path), "offline_gt": sha256_file(gt_path),
                              "predictions": sha256_file(args.predictions)}
    write_jsonl(args.output/"per_frame_errors.jsonl", rows)
    write_json(args.output/"metrics.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
