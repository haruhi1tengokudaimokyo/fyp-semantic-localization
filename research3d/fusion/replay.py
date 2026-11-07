"""Causal frozen-frontend replay, not an online speed benchmark or a pose graph.

One accepted absolute constraint per output avoids treating shared-RGB branches
as independent measurements. Separate recovery hypotheses are never averaged.
Every late result is associated to its own image and propagated through odometry.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .contracts import (IDENTITY, discrepancy, identity, pose, read_jsonl,
                        validate_motion, validate_observation, write_json, write_jsonl,
                        sha256_file)


def _close(a, b, gates):
    d, r = discrepancy(a, b)
    return d <= gates[0] and r <= gates[1]


def run_replay(motions, observations, config):
    """Return estimates/audit; no GT argument, no automatic global alignment.

    Outputs become available when the motion frontend completes each image.
    Cached future predictions are never used before their completion timestamp.
    Absolute results after the last motion output are reported as unconsumed.
    """
    if not motions:
        raise ValueError("empty motion stream")
    for row in motions:
        validate_motion(row)
    ident = identity(motions[0])
    by_id = {row["query_id"]: row for row in motions}
    if len(by_id) != len(motions):
        raise ValueError("duplicate motion query IDs")
    previous_capture, previous_finish = -np.inf, -np.inf
    chains = {}
    continuity = 0
    previous = None
    for row in motions:
        if identity(row) != ident:
            raise ValueError("one replay must use one source/building/map/camera")
        if row["capture_time_s"] <= previous_capture or row["completed_time_s"] < previous_finish:
            raise ValueError("motion stream must have increasing capture and monotonic completion")
        previous_capture, previous_finish = row["capture_time_s"], row["completed_time_s"]
        if previous is None or not previous["valid"] or not row["valid"] or row["segment_id"] != previous["segment_id"]:
            continuity += 1
        chains[row["query_id"]] = continuity
        previous = row
    seen = set()
    for row in observations:
        validate_observation(row)
        key = (row["source"], row["query_id"])
        if key in seen:
            raise ValueError("duplicate frontend result")
        seen.add(key)
        ref = by_id.get(row["query_id"])
        if ref is None or identity(row) != ident:
            raise ValueError("observation has no matching same-map image")
        if abs(ref["capture_time_s"] - row["capture_time_s"]) > 1e-6:
            raise ValueError("observation and image time disagree")
        for field in ("image_sha256", "calibration_sha256"):
            if row["provenance"][field] != ref["provenance"][field]:
                raise ValueError(f"observation uses different {field}")
    cfg = copy.deepcopy(config)
    enabled = cfg["absolute_sources"]
    if any(s not in {"ace", "spvloc"} for s in enabled):
        raise ValueError("unknown source")
    # Initialization from query GT is deliberately not supported by this entry.
    if cfg.get("initialization", "global_unknown") != "global_unknown":
        raise ValueError("use a separate declared known-start evaluation, not GT in this replay")
    queue = sorted((x for x in observations if x["source"] in enabled),
                   key=lambda x: (x["completed_time_s"], x["capture_time_s"], x["source"]))
    pending = list(queue)
    transform = None  # map <- odom
    modes = []
    segment = None
    conflicts = 0
    last_global_time = None
    outputs, decisions = [], []
    for motion in motions:
        t = motion["capture_time_s"]
        finish = motion["completed_time_s"]
        valid = motion["valid"]
        if motion["segment_id"] != segment or not valid:
            transform, modes, conflicts = None, [], 0
            last_global_time = None
        segment = motion["segment_id"]
        odom = pose(motion["T_odom_camera"]) if valid else None
        before = None if transform is None or odom is None else transform @ odom
        accepted = []
        chosen = None
        rejected = []
        batch = []
        available = [ob for ob in pending if ob["completed_time_s"] <= finish and ob["capture_time_s"] <= t]
        pending = [ob for ob in pending if not (ob["completed_time_s"] <= finish and ob["capture_time_s"] <= t)]
        for ob in available:
            ref = by_id[ob["query_id"]]
            reason = None
            age = t - ob["capture_time_s"]
            if not valid or not ref["valid"] or chains[ref["query_id"]] != chains[motion["query_id"]]:
                reason = "broken_motion_chain"
            elif age > cfg["max_observation_age_s"]:
                reason = "stale_observation"
            elif not ob["candidates"]:
                reason = "no_candidate"
            if reason:
                rejected.append(dict(source=ob["source"], query_id=ob["query_id"], reason=reason))
                continue
            for candidate in ob["candidates"][:cfg["top_k"]]:
                quality = candidate["quality"]
                source = ob["source"]
                floor = cfg["quality"][source]
                passed = (quality["pnp_inliers"] >= floor["min_pnp_inliers"] if source == "ace"
                          else quality["viewport_score"] >= floor["min_viewport_score"])
                if not passed:
                    continue
                z = pose(candidate["T_map_camera"])
                origin = z @ np.linalg.inv(pose(ref["T_odom_camera"]))
                batch.append(dict(origin=origin, pose_now=origin @ odom, observation=ob,
                                  rank=candidate["rank"], source=source))
        # Highest priority is deterministic, not comparison of incompatible scores.
        batch.sort(key=lambda b: (cfg["source_priority"].index(b["source"]), b["rank"],
                                  -b["observation"]["capture_time_s"]))
        if transform is not None:
            consistent = [b for b in batch if _close(before, b["pose_now"], cfg["tracking_gate"][b["source"]])]
            if consistent:
                chosen = consistent[0]
                conflicts = 0
            elif batch:
                conflicts += 1
            if cfg["independent_recovery"] and conflicts >= cfg["conflicts_before_lost"]:
                transform = None
                modes = []
        if transform is None and valid and batch:
            modes = [m for m in modes if t - m["last_time"] <= cfg["hypothesis_ttl_s"]]
            # Process older images first. Source correlation never creates two hits.
            for b in sorted(batch, key=lambda b: (b["observation"]["capture_time_s"], b["rank"])):
                matching = [m for m in modes if _close(m["origin"], b["origin"], cfg["recovery_gate"][b["source"]])]
                if matching:
                    m = min(matching, key=lambda m: discrepancy(m["origin"], b["origin"])[0])
                else:
                    m = dict(origin=b["origin"], queries={}, last_time=b["observation"]["capture_time_s"], item=b)
                    modes.append(m)
                qid = b["observation"]["query_id"]
                duplicate_image = qid in m["queries"]
                m["queries"][qid] = b["observation"]["capture_time_s"]
                m["last_time"] = max(m["last_time"], b["observation"]["capture_time_s"])
                # Keep a real candidate, never the mean of different room modes.
                if not duplicate_image or cfg["source_priority"].index(b["source"]) < cfg["source_priority"].index(m["item"]["source"]):
                    m["origin"], m["item"] = b["origin"], b
            modes = [m for m in modes if t - m["last_time"] <= cfg["hypothesis_ttl_s"]]
            for m in modes:
                m["queries"] = {qid: captured for qid, captured in m["queries"].items()
                                if t - captured <= cfg["hypothesis_ttl_s"]}
            mature = [m for m in modes if len(m["queries"]) >= cfg["confirmation_frames"]]
            # Ambiguous coherent alternatives cause abstention, not arbitrary tie-breaking.
            if len(mature) == 1:
                chosen = mature[0]["item"]
                modes = []
        if chosen is not None:
            transform = chosen["origin"].copy()
            last_global_time = chosen["observation"]["capture_time_s"]
            conflicts = 0
            accepted = [dict(source=chosen["source"], query_id=chosen["observation"]["query_id"], rank=chosen["rank"])]
        if transform is not None and last_global_time is not None and t - last_global_time > cfg["max_dead_reckoning_s"]:
            transform = None
        estimate = None if transform is None or odom is None else transform @ odom
        status = ("TRACKING" if accepted else "DEAD_RECKONING") if estimate is not None else ("RECOVERING" if modes else "LOST")
        out = {k: motion[k] for k in IDENTITY}
        out.update(schema_version=1, query_id=motion["query_id"], timestamp_s=t,
                   pose_time_s=t, output_time_s=finish, valid=estimate is not None,
                   T_map_camera=None if estimate is None else estimate.tolist(),
                   T_before_correction=None if before is None else before.tolist(),
                   accepted_observations=accepted, status=status,
                   local_odometry_valid=valid, T_odom_camera=motion["T_odom_camera"],
                   provenance={"mode": "causal_cached_frontend_replay", "online_gt_used": False})
        outputs.append(out)
        decisions.append(dict(query_id=motion["query_id"], status=status, accepted=accepted,
                              rejected=rejected, recovery_modes=len(modes), conflicts=conflicts))
    return outputs, dict(decisions=decisions, enabled_absolute_sources=enabled,
                         absolute_results_not_completed_by_last_output=len(pending),
                         future_outputs_rewritten=False, estimate_alignment_to_gt=False,
                         backend="candidate_gating_and_map_odom_correction_not_pose_graph",
                         real_time_claim=False, scores_are_probabilities=False,
                         correlated_sources_combined_as_independent=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/fusion/experiment.json"))
    p.add_argument("--variant", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("refuse to overwrite a replay run")
    from .protocol import validate_config
    protocol = validate_config(json.loads(args.config.read_text()))
    cfg = {**protocol["replay_defaults"], **protocol["variants"][args.variant]}
    outputs, audit = run_replay(read_jsonl(args.motion), read_jsonl(args.observations), cfg)
    audit.update(variant=args.variant, config=cfg, inputs={str(f): sha256_file(f) for f in (args.motion, args.observations, args.config)})
    write_jsonl(args.output / "outputs.jsonl", outputs)
    write_json(args.output / "audit.json", audit)
    print(json.dumps({"outputs": len(outputs), "valid": sum(x["valid"] for x in outputs), "output": str(args.output)}))


if __name__ == "__main__":
    main()
