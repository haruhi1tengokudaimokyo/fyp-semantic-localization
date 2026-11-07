"""Offline metrics in a fixed map frame; never called by the online backend.

No alignment, scale fit, interpolation, or removal of missing estimates is
performed. Percentiles describe valid outputs only; success uses every GT row.
GT is used only in this evaluation module and in ``diagnostics``.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np

IDENTITY_FIELDS = ("source_id", "building_id", "map_id", "camera_id")


def identity(row: dict) -> tuple[str, ...]:
    result = tuple(row.get(k) for k in IDENTITY_FIELDS)
    if not all(isinstance(v, str) and v for v in result):
        raise ValueError(f"Nonempty identity fields required: {IDENTITY_FIELDS}")
    return result


def finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name}: finite number required")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name}: nonfinite value")
    return value


def se3(value) -> np.ndarray:
    """Same rigid-pose convention as knownmap.contracts, with tighter tolerance."""
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Finite 4x4 SE(3) required")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("Invalid homogeneous row")
    rot = pose[:3, :3]
    if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5, rtol=0) or not np.isclose(
        np.linalg.det(rot), 1, atol=1e-5, rtol=0
    ):
        raise ValueError("Invalid SO(3); scale/reflection is not a rotation")
    return pose


def pose_error(estimate, truth) -> tuple[float, float]:
    estimate, truth = se3(estimate), se3(truth)
    translation = float(np.linalg.norm(estimate[:3, 3] - truth[:3, 3]))
    relative = truth[:3, :3].T @ estimate[:3, :3]
    # atan2 gives stable principal angles near both zero and pi.
    skew = np.array([relative[2, 1] - relative[1, 2],
                     relative[0, 2] - relative[2, 0],
                     relative[1, 0] - relative[0, 1]])
    rotation = float(np.degrees(np.arctan2(np.linalg.norm(skew) / 2,
                                         np.clip((np.trace(relative) - 1) / 2, -1, 1))))
    return translation, rotation


def distribution(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite statistic inputs must not be silently dropped")
    if not len(array):
        return dict(n=0, mean=None, p50=None, p95=None, maximum=None, rmse=None)
    return dict(n=len(array), mean=float(array.mean()),
                p50=float(np.quantile(array, .5)), p95=float(np.quantile(array, .95)),
                maximum=float(array.max()), rmse=float(np.sqrt(np.mean(array**2))))


def _fraction(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def validate_truth(rows: list[dict]) -> dict[str, dict]:
    if not rows:
        raise ValueError("Ground truth must define every evaluation instant")
    index = {}
    group_times = defaultdict(set)
    for original in rows:
        row = dict(original)
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id or query_id in index:
            raise ValueError("GT query_id must be a nonempty globally unique string")
        group = identity(row)
        timestamp = row.get("timestamp_s", row.get("timestamp"))
        row["timestamp_s"] = finite_number(timestamp, "GT timestamp_s")
        if "timestamp" in row and abs(row["timestamp_s"] - finite_number(row["timestamp"], "timestamp")) > 1e-6:
            raise ValueError("Conflicting GT timestamp aliases")
        if row["timestamp_s"] in group_times[group]:
            raise ValueError("Duplicate evaluation instant within a map/camera stream")
        group_times[group].add(row["timestamp_s"])
        se3(row["T_map_camera"])
        index[query_id] = row
    return index


def validate_outputs(rows: list[dict], truth: dict[str, dict]) -> dict[str, dict]:
    result = {}
    for original in rows:
        row = dict(original)
        qid = row.get("query_id")
        if qid not in truth or qid in result:
            raise ValueError("Unknown/duplicate system output query_id")
        gt = truth[qid]
        if identity(row) != identity(gt):
            raise ValueError("Output and GT source/building/map/camera mismatch")
        timestamp = finite_number(row.get("timestamp_s"), "output timestamp_s")
        if abs(timestamp - gt["timestamp_s"]) > 1e-6:
            raise ValueError("Output timestamp does not match evaluation query")
        if not isinstance(row.get("valid"), bool):
            raise ValueError("Output valid must be an explicit bool")
        output_time = finite_number(row.get("output_time_s"), "output_time_s")
        if output_time < timestamp:
            raise ValueError("Output cannot precede its evaluation instant")
        if row["valid"]:
            pose_time = finite_number(row.get("pose_time_s"), "pose_time_s")
            if abs(pose_time - timestamp) > 1e-6:
                raise ValueError("Stale pose must be propagated to evaluation time before scoring")
            se3(row.get("T_map_camera"))
        elif row.get("T_map_camera") is not None:
            # A finite internal hypothesis may exist, but it is not a valid output.
            se3(row["T_map_camera"])
        if row.get("T_before_correction") is not None:
            se3(row["T_before_correction"])
        if not isinstance(row.get("accepted_observations", []), list):
            raise ValueError("accepted_observations must be a list")
        result[qid] = row
    return result


def _failure_runs(rows: list[dict]) -> list[dict]:
    """Durations stop at next success, or last sampled time (right-censored)."""
    runs, start = [], None
    for i, row in enumerate(rows):
        if not row["success"] and start is None:
            start = i
        if row["success"] and start is not None:
            runs.append(dict(start_query_id=rows[start]["query_id"],
                             end_query_id=row["query_id"], frames=i-start,
                             duration_s=row["timestamp_s"]-rows[start]["timestamp_s"],
                             right_censored=False))
            start = None
    if start is not None:
        runs.append(dict(start_query_id=rows[start]["query_id"],
                         end_query_id=rows[-1]["query_id"], frames=len(rows)-start,
                         duration_s=rows[-1]["timestamp_s"]-rows[start]["timestamp_s"],
                         right_censored=True))
    return runs


def _system_aggregate(rows: list[dict]) -> dict:
    valid = [r for r in rows if r["valid"]]
    successes = sum(r["success"] for r in rows)
    return dict(evaluation_instants=len(rows), valid_outputs=len(valid),
                missing_or_invalid_outputs=len(rows)-len(valid), successes=successes,
                output_coverage=_fraction(len(valid), len(rows)),
                global_success_coverage=_fraction(successes, len(rows)),
                failure_fraction=_fraction(len(rows)-successes, len(rows)),
                error_denominator="valid_outputs_only; missing outputs remain failures in coverage",
                translation_m=distribution(r["translation_m"] for r in valid),
                rotation_deg=distribution(r["rotation_deg"] for r in valid),
                output_age_s=distribution(r["output_age_s"] for r in valid))


def system_metrics(output_rows: list[dict], gt_rows: list[dict], *,
                   translation_threshold_m: float, rotation_threshold_deg: float,
                   max_output_age_s: float | None = None) -> dict:
    truth = validate_truth(gt_rows)
    outputs = validate_outputs(output_rows, truth)
    translation_threshold_m = finite_number(translation_threshold_m, "translation threshold")
    rotation_threshold_deg = finite_number(rotation_threshold_deg, "rotation threshold")
    if translation_threshold_m <= 0 or rotation_threshold_deg <= 0:
        raise ValueError("Positive success thresholds required")
    if max_output_age_s is not None:
        max_output_age_s = finite_number(max_output_age_s, "max_output_age_s")
        if max_output_age_s < 0:
            raise ValueError("Output age limit must be nonnegative")
    per_query, groups = [], defaultdict(list)
    for qid, gt in truth.items():
        out = outputs.get(qid)
        row = {k: gt[k] for k in IDENTITY_FIELDS}
        row.update(query_id=qid, timestamp_s=gt["timestamp_s"],
                   valid=bool(out and out["valid"]), success=False,
                   translation_m=None, rotation_deg=None, output_age_s=None,
                   missing_output=out is None)
        if row["valid"]:
            dt, dr = pose_error(out["T_map_camera"], gt["T_map_camera"])
            age = out["output_time_s"] - out["pose_time_s"]
            row.update(translation_m=dt, rotation_deg=dr, output_age_s=age,
                       success=dt <= translation_threshold_m and dr <= rotation_threshold_deg
                       and (max_output_age_s is None or age <= max_output_age_s))
        per_query.append(row)
        groups[identity(row)].append(row)
    by_stream = []
    for group, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["timestamp_s"])
        runs = _failure_runs(rows)
        stream = dict(zip(IDENTITY_FIELDS, group))
        stream.update(_system_aggregate(rows))
        stream.update(failure_runs=runs,
                      longest_failure_observed_s=max((r["duration_s"] for r in runs), default=0),
                      longest_failure_frames=max((r["frames"] for r in runs), default=0))
        relative_translation, relative_rotation, pair_durations = [], [], []
        for first, second in zip(rows, rows[1:]):
            if not first["valid"] or not second["valid"]:
                continue
            first_id, second_id = first["query_id"], second["query_id"]
            relative_estimate = np.linalg.inv(se3(outputs[first_id]["T_map_camera"])) @ se3(outputs[second_id]["T_map_camera"])
            relative_truth = np.linalg.inv(se3(truth[first_id]["T_map_camera"])) @ se3(truth[second_id]["T_map_camera"])
            dt, dr = pose_error(relative_estimate, relative_truth)
            relative_translation.append(dt)
            relative_rotation.append(dr)
            pair_durations.append(second["timestamp_s"] - first["timestamp_s"])
        stream["rpe_adjacent_evaluation_instants"] = dict(
            possible_pairs=max(0, len(rows)-1), valid_pairs=len(relative_translation),
            invalid_pairs=max(0, len(rows)-1)-len(relative_translation),
            translation_m=distribution(relative_translation), rotation_deg=distribution(relative_rotation),
            pair_duration_s=distribution(pair_durations),
            note="Adjacent GT inventory only; no interpolation or bridging invalid outputs. Global correction jumps remain visible.")
        by_stream.append(stream)
    return dict(alignment="none: fixed-map absolute poses, no GT fit or scale correction",
                thresholds=dict(translation_m=translation_threshold_m,
                                rotation_deg=rotation_threshold_deg,
                                max_output_age_s=max_output_age_s),
                aggregate=_system_aggregate(per_query), by_stream=by_stream,
                per_query=per_query,
                failure_duration_note="Intervals use real timestamps; final failure run is right-censored at last sample. No extrapolated last-frame duration.")


def recovery_metrics(events: list[dict], per_query: list[dict], *,
                     consecutive_successes: int = 3) -> dict:
    """Evaluate prespecified recovery windows; never discard unrecovered events.

    A failure breaks the confirmation streak. End time is a fixed observation
    horizon, not a value selected from the system output. Detection/recovery time
    is the completion time of the Nth successful output, including output age.
    """
    if isinstance(consecutive_successes, bool) or not isinstance(consecutive_successes, int) or consecutive_successes < 1:
        raise ValueError("consecutive_successes must be a positive integer")
    groups = defaultdict(list)
    for row in per_query:
        groups[identity(row)].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: r["timestamp_s"])
    details, seen, windows = [], set(), defaultdict(list)
    for event in events:
        eid = event.get("event_id")
        if not isinstance(eid, str) or not eid or eid in seen:
            raise ValueError("Unique nonempty recovery event_id required")
        seen.add(eid)
        group = identity(event)
        start = finite_number(event.get("start_time_s"), "event start_time_s")
        end = finite_number(event.get("end_time_s"), "event end_time_s")
        if end < start or group not in groups:
            raise ValueError("Recovery event has invalid time range or unknown stream")
        if not event.get("kind"):
            raise ValueError("Recovery event kind is required")
        if any(start < previous_end and end > previous_start for previous_start, previous_end in windows[group]):
            raise ValueError("Overlapping recovery events in one stream")
        windows[group].append((start, end))
        available = groups[group]
        if start < available[0]["timestamp_s"] or end > available[-1]["timestamp_s"]:
            raise ValueError("Recovery window extends beyond observed trajectory")
        rows = [r for r in available if start <= r["timestamp_s"] <= end]
        if not rows:
            raise ValueError("Recovery event has no evaluation instants")
        streak, confirmed, first, completion_window = 0, None, None, []
        for row in rows:
            if row["success"]:
                if not streak:
                    first = row["timestamp_s"]
                completion_window.append((row["timestamp_s"], row["timestamp_s"] + row["output_age_s"]))
                completion_window = completion_window[-consecutive_successes:]
                streak += 1
            else:
                streak, first, completion_window = 0, None, []
            if streak >= consecutive_successes:
                completion = max(item[1] for item in completion_window)
                if completion <= end:
                    confirmed = (row, completion, completion_window[0][0])
                    break
        detail = {k: event[k] for k in IDENTITY_FIELDS}
        detail.update(event_id=eid, kind=event["kind"], start_time_s=start,
                      end_time_s=end, recovered=confirmed is not None,
                      recovery_time_s=None if confirmed is None else confirmed[1]-start,
                      first_correct_in_confirmed_streak_s=None if confirmed is None else confirmed[2]-start,
                      confirmation_query_id=None if confirmed is None else confirmed[0]["query_id"],
                      right_censored=confirmed is None,
                      observation_horizon_s=end-start, evaluation_instants=len(rows))
        details.append(detail)
    recovered = [r for r in details if r["recovered"]]
    return dict(events=len(details), recovered=len(recovered),
                unrecovered=len(details)-len(recovered),
                recovery_success_rate=_fraction(len(recovered), len(details)),
                consecutive_successes=consecutive_successes,
                recovery_time_successful_events_only_s=distribution(r["recovery_time_s"] for r in recovered),
                event_details=details,
                note="Events must be prescribed independently of evaluated predictions; unrecovered events are right-censored, never assigned zero latency.")
