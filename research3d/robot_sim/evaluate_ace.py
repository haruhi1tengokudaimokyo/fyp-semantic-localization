"""CPU-only independent fixed-map ACE evaluation; never imported online."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from research3d.knownmap.contracts import se3
from research3d.robot_sim.prepare_ace import atomic_json, read_jsonl, sha256


THRESHOLDS = {"5cm5deg": (.05, 5.), "25cm10deg": (.25, 10.), "1m30deg": (1., 30.)}


def evaluate(dataset, run):
    dataset, run = Path(dataset), Path(run)
    manifest = json.loads((dataset / "map_manifest.json").read_text())
    truth_rows = read_jsonl(dataset / "offline_gt/query_poses.jsonl")
    query_rows = read_jsonl(dataset / "query/frames.jsonl")
    predicted = read_jsonl(run / "predictions.jsonl")
    truth = {r["query_id"]: r for r in truth_rows}
    queries = {r["query_id"]: r for r in query_rows}
    by_id = {r["query_id"]: r for r in predicted}
    if len(by_id) != len(predicted) or len(truth) != len(truth_rows) or len(queries) != len(query_rows):
        raise ValueError("duplicate query identity")
    if set(by_id) != set(truth) or set(queries) != set(truth):
        raise ValueError("full-query coverage required; no unmatched frames dropped")
    errors, latencies, rows = [], [], []
    for p in predicted:
        qid = p["query_id"]
        g, q = truth[qid], queries[qid]
        if p["timestamp_ns"] != g["timestamp_ns"] or q["timestamp_ns"] != g["timestamp_ns"]:
            raise ValueError("exact timestamp mismatch")
        if p["map_id"] != manifest["map_id"] or p["map_sha256"] != manifest["map_sha256"]:
            raise ValueError("map binding mismatch")
        if p["image_sha256"] != q["image_sha256"]:
            raise ValueError("query image binding mismatch")
        valid = p["status"] == "ok" and p.get("T_map_camera") is not None
        te = re = None
        if valid:
            estimate, gt = se3(p["T_map_camera"]), se3(g["T_map_camera"])
            te = float(np.linalg.norm(estimate[:3, 3] - gt[:3, 3]))
            re = float(np.degrees(Rotation.from_matrix(gt[:3, :3].T @ estimate[:3, :3]).magnitude()))
            errors.append([te, re])
        passed = {name: bool(valid and te <= t and re <= r) for name, (t, r) in THRESHOLDS.items()}
        rows.append(dict(query_id=qid, timestamp_ns=p["timestamp_ns"], model_status=p["status"],
                         valid_pose=valid, translation_error_m=te, rotation_error_deg=re,
                         inliers=p["inliers"], success=passed, accepted_by_system=p.get("accepted", False)))
        latencies.append(float(p["latency_ms"]))
    e = np.asarray(errors, float)
    result = dict(status="EVALUATED_GAZEBO_DEVELOPMENT_MAP", method="official ACE + DSAC*",
                  map_id=manifest["map_id"], map_sha256=manifest["map_sha256"], frames=len(predicted),
                  pose_outputs=len(errors), pose_output_coverage=len(errors) / len(predicted),
                  error_scope="all mathematically valid raw poses; no confidence filtering, GT fitting, interpolation or failure deletion",
                  alignment="none: original fixed map coordinates", GT_online=False,
                  thresholds={k: dict(translation_m=t, rotation_deg=r) for k, (t, r) in THRESHOLDS.items()},
                  success_count={k: sum(r["success"][k] for r in rows) for k in THRESHOLDS},
                  success_rate={k: sum(r["success"][k] for r in rows) / len(rows) for k in THRESHOLDS},
                  failure_count={k: sum(not r["success"][k] for r in rows) for k in THRESHOLDS},
                  translation_rmse_m=float(np.sqrt(np.mean(e[:, 0] ** 2))) if len(e) else None,
                  translation_median_m=float(np.median(e[:, 0])) if len(e) else None,
                  translation_p95_m=float(np.quantile(e[:, 0], .95)) if len(e) else None,
                  rotation_median_deg=float(np.median(e[:, 1])) if len(e) else None,
                  rotation_p95_deg=float(np.quantile(e[:, 1], .95)) if len(e) else None,
                  algorithm_latency_ms=dict(mean=float(np.mean(latencies)), median=float(np.median(latencies)),
                                            p95=float(np.quantile(latencies, .95))),
                  min_inliers=min(r["inliers"] for r in rows),
                  source_predictions_sha256=sha256(run / "predictions.jsonl"),
                  offline_gt_sha256=sha256(dataset / "offline_gt/query_poses.jsonl"),
                  test_tuning=False, system_gate_applied=False,
                  scope="new synthetic rendering on a development building; not ACE 7-Scenes paper reproduction or real-robot validation")
    atomic_json(run / "metrics.json", result)
    atomic_json(run / "per_query_errors.json", rows)
    atomic_json(run / "failures_1m30deg.json", [r for r in rows if not r["success"]["1m30deg"]])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(evaluate(args.dataset, args.run), indent=2))


if __name__ == "__main__":
    main()
