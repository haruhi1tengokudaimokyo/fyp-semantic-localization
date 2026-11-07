"""Strict, offline complementarity and fixed-map system evaluation.

Example (GT is never passed to ``replay`` or the online backend)::

    python -m research3d.fusion.diagnostics --observations observations.jsonl \
        --ground-truth ground_truth.jsonl --outputs outputs.jsonl \
        --events recovery_events.jsonl --translation-m 1 --rotation-deg 30 \
        --output diagnostics.json

Top-3 is an *offline oracle*, not a deployed selection policy. Missing method
outputs count as failures against the independently defined GT query inventory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from .metrics import (IDENTITY_FIELDS, distribution, finite_number, identity,
                      pose_error, recovery_metrics, se3, system_metrics,
                      validate_outputs, validate_truth)

_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
_OK = {"ok", "valid", "success"}
_FAILED = {"failed", "unavailable", "invalid", "not_run", "timeout", "no_candidate", "skipped"}


def read_jsonl(path: Path | str) -> list[dict]:
    """Reject malformed/nonfinite input and silently overwritten duplicate keys."""
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    result = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=object_pairs,
                             parse_constant=bad_constant)
            if not isinstance(row, dict):
                raise ValueError("JSONL row must be an object")
            result.append(row)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return result


def validate_observations(rows: list[dict], truth: dict[str, dict]) -> dict[tuple[str, str], dict]:
    result, image_bindings = {}, {}
    for row in rows:
        if row.get("schema_version") != 1 or isinstance(row.get("schema_version"), bool):
            raise ValueError("Observation schema_version must equal integer 1")
        source, qid = row.get("source"), row.get("query_id")
        if source not in {"ace", "spvloc"}:
            raise ValueError("Absolute observation source must be ace or spvloc")
        if qid not in truth or (source, qid) in result:
            raise ValueError("Unknown or duplicate observation source/query_id")
        gt = truth[qid]
        if identity(row) != identity(gt):
            raise ValueError("Observation and GT source/building/map/camera mismatch")
        captured = finite_number(row.get("capture_time_s"), "capture_time_s")
        completed = finite_number(row.get("completed_time_s"), "completed_time_s")
        if abs(captured - gt["timestamp_s"]) > 1e-6:
            raise ValueError("Observation timestamp is not the matching query timestamp")
        if completed < captured:
            raise ValueError("Observation cannot finish before image capture")
        status = row.get("status")
        if not isinstance(status, str) or status.lower() not in _OK | _FAILED:
            raise ValueError("Unknown observation status")
        candidates = row.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("Candidates must be a list")
        if bool(candidates) != (status.lower() in _OK):
            raise ValueError("Successful observations need candidates; failed observations must have none")
        for rank, candidate in enumerate(candidates, 1):
            if not isinstance(candidate, dict) or type(candidate.get("rank")) is not int or candidate["rank"] != rank:
                raise ValueError("Candidate ranks must be ordered, unique, consecutive and 1-based")
            se3(candidate.get("T_map_camera"))
            if not isinstance(candidate.get("quality"), dict):
                raise ValueError("Source-specific quality dictionary required")
        provenance = row.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("Observation provenance required")
        for field in ("image_sha256", "model_sha256", "calibration_sha256"):
            if not isinstance(provenance.get(field), str) or not _SHA256.fullmatch(provenance[field]):
                raise ValueError(f"Missing or malformed provenance {field}")
        binding = (provenance["image_sha256"].lower(), provenance["calibration_sha256"].lower())
        if qid in image_bindings and image_bindings[qid] != binding:
            raise ValueError("Methods did not consume the same query image/calibration")
        image_bindings[qid] = binding
        expected_image = gt.get("image_sha256")
        if expected_image is not None and expected_image.lower() != binding[0]:
            raise ValueError("Query image hash differs from independent query inventory")
        result[source, qid] = row
    return result


def _correct(candidate, truth, translation_m, rotation_deg):
    dt, dr = pose_error(candidate["T_map_camera"], truth["T_map_camera"])
    return dt <= translation_m and dr <= rotation_deg


def _complement_aggregate(rows):
    n = len(rows)
    cells = dict(both_correct=0, ace_only_correct=0, spvloc_only_correct=0, both_wrong=0)
    for row in rows:
        a, s = row["ace_top1_correct"], row["spvloc_top1_correct"]
        cell = "both_correct" if a and s else ("ace_only_correct" if a else ("spvloc_only_correct" if s else "both_wrong"))
        cells[cell] += 1
    result = dict(evaluation_instants=n, four_cells=cells,
                  ace_top1_successes=sum(r["ace_top1_correct"] for r in rows),
                  spvloc_top1_successes=sum(r["spvloc_top1_correct"] for r in rows),
                  spvloc_top3_oracle_successes=sum(r["spvloc_top3_correct"] for r in rows),
                  top1_union_oracle_successes=sum(r["ace_top1_correct"] or r["spvloc_top1_correct"] for r in rows),
                  ace_plus_spvloc_top3_oracle_successes=sum(r["ace_top1_correct"] or r["spvloc_top3_correct"] for r in rows),
                  potential_top1_gain_over_ace_count=cells["spvloc_only_correct"],
                  ace_missing_or_failed=sum(not r["ace_has_output"] for r in rows),
                  spvloc_missing_or_failed=sum(not r["spvloc_has_output"] for r in rows),
                  spvloc_fewer_than_three_candidates=sum(r["spvloc_candidate_count"] < 3 for r in rows))
    for key in ("ace_top1", "spvloc_top1", "spvloc_top3_oracle", "top1_union_oracle", "ace_plus_spvloc_top3_oracle"):
        result[key + "_recall"] = result[key + "_successes"] / n if n else None
    result["potential_top1_gain_over_ace"] = cells["spvloc_only_correct"] / n if n else None
    return result


def complementarity(observations: list[dict], gt_rows: list[dict], *,
                    translation_threshold_m: float = 1., rotation_threshold_deg: float = 30.) -> dict:
    translation_threshold_m = finite_number(translation_threshold_m, "translation threshold")
    rotation_threshold_deg = finite_number(rotation_threshold_deg, "rotation threshold")
    if translation_threshold_m <= 0 or rotation_threshold_deg <= 0:
        raise ValueError("Success thresholds must be positive")
    truth = validate_truth(gt_rows)
    index = validate_observations(observations, truth)
    per_query, groups = [], defaultdict(list)
    for qid, gt in truth.items():
        ace = index.get(("ace", qid), {}).get("candidates", [])
        spv = index.get(("spvloc", qid), {}).get("candidates", [])
        a = bool(ace and _correct(ace[0], gt, translation_threshold_m, rotation_threshold_deg))
        s = bool(spv and _correct(spv[0], gt, translation_threshold_m, rotation_threshold_deg))
        s3 = any(_correct(c, gt, translation_threshold_m, rotation_threshold_deg) for c in spv[:3])
        row = {k: gt[k] for k in IDENTITY_FIELDS}
        row.update(query_id=qid, timestamp_s=gt["timestamp_s"],
                   ace_top1_correct=a, spvloc_top1_correct=s, spvloc_top3_correct=s3,
                   ace_has_output=bool(ace), spvloc_has_output=bool(spv),
                   spvloc_candidate_count=len(spv))
        per_query.append(row)
        groups[(gt["source_id"], gt["building_id"])].append(row)
    by_building = []
    for (source, building), rows in sorted(groups.items()):
        item = dict(source_id=source, building_id=building)
        item.update(_complement_aggregate(rows))
        by_building.append(item)
    aggregate = _complement_aggregate(per_query)
    recall_keys = [k for k in aggregate if k.endswith("_recall")]
    return dict(interpretation="Offline candidate oracle only; GT is not available to online selection. Missing predictions are failures.",
                thresholds=dict(translation_m=translation_threshold_m, rotation_deg=rotation_threshold_deg),
                aggregate=aggregate, by_building=by_building,
                building_macro={key: sum(b[key] for b in by_building) / len(by_building) for key in recall_keys},
                latency_s={source: distribution(row["completed_time_s"]-row["capture_time_s"]
                                                for row in observations if row["source"] == source)
                           for source in ("ace", "spvloc")},
                per_query=per_query)


def correction_metrics(output_rows: list[dict], observations: list[dict], gt_rows: list[dict], *,
                       translation_threshold_m: float, rotation_threshold_deg: float) -> dict:
    translation_threshold_m = finite_number(translation_threshold_m, "translation threshold")
    rotation_threshold_deg = finite_number(rotation_threshold_deg, "rotation threshold")
    if translation_threshold_m <= 0 or rotation_threshold_deg <= 0:
        raise ValueError("Positive success thresholds required")
    truth = validate_truth(gt_rows)
    outputs = validate_outputs(output_rows, truth)
    index = validate_observations(observations, truth)
    accepted, before_missing, no_prior, bad, details = 0, 0, 0, 0, []
    for qid, out in outputs.items():
        refs = out.get("accepted_observations", [])
        if not refs:
            continue
        if not out["valid"]:
            raise ValueError("An accepted correction must have a valid post-correction pose")
        seen, wrong_candidates = set(), 0
        for ref in refs:
            source, original_qid, rank = ref.get("source"), ref.get("query_id"), ref.get("rank")
            key = (source, original_qid)
            if key not in index or type(rank) is not int or rank < 1:
                raise ValueError("Accepted observation is not traceable to a candidate")
            obs = index[key]
            if identity(obs) != identity(out) or obs["capture_time_s"] > out["timestamp_s"]:
                raise ValueError("Accepted observation has wrong stream or future capture time")
            if obs["completed_time_s"] > out["output_time_s"]:
                raise ValueError("A candidate was accepted before inference completed")
            if rank > len(obs["candidates"]) or (source, original_qid, rank) in seen:
                raise ValueError("Duplicate or nonexistent accepted candidate")
            seen.add((source, original_qid, rank))
            candidate = obs["candidates"][rank-1]
            correct = _correct(candidate, truth[original_qid], translation_threshold_m, rotation_threshold_deg)
            wrong_candidates += int(not correct)
            accepted += 1
        after_correct = _correct({"T_map_camera": out["T_map_camera"]}, truth[qid], translation_threshold_m, rotation_threshold_deg)
        before = out.get("T_before_correction")
        if before is None:
            if "T_before_correction" in out:
                no_prior += 1
            else:
                before_missing += 1
            before_correct, harmful = None, None
        else:
            before_correct = _correct({"T_map_camera": before}, truth[qid], translation_threshold_m, rotation_threshold_deg)
            harmful = before_correct and not after_correct
            bad += int(harmful)
        details.append(dict(query_id=qid, accepted_candidates=len(refs),
                            wrong_accepted_candidates=wrong_candidates,
                            before_correct=before_correct, after_correct=after_correct,
                            correct_to_wrong=harmful,
                            no_prior_global_pose="T_before_correction" in out and before is None))
    wrong = sum(row["wrong_accepted_candidates"] for row in details)
    return dict(accepted_candidate_count=accepted, wrong_accepted_candidate_count=wrong,
                wrong_acceptance_fraction=wrong / accepted if accepted else None,
                correction_event_count=len(details),
                correction_events_missing_before_pose=before_missing,
                initialization_or_no_prior_pose_events=no_prior,
                bad_corrections_assessable=len(details)-before_missing-no_prior,
                correct_to_wrong_count=bad if len(details) > before_missing+no_prior else None,
                status="COMPLETE" if before_missing == 0 else "PARTIAL_MISSING_PRE_CORRECTION_POSE",
                details=details,
                definition="Candidate correctness at original capture time; harmful correction compares before/after at the same current timestamp. Explicit null prior means initialization/no prior global pose, whereas absent prior field is missing evidence; neither is counted as a correct-to-wrong trial.")


def evaluate(observations: list[dict], gt_rows: list[dict], *, output_rows=None,
             events=None, translation_threshold_m=1., rotation_threshold_deg=30.,
             max_output_age_s=None, consecutive_successes=3) -> dict:
    result = dict(schema_version=1,
                  result_type="synthetic_software_test" if any(r.get("synthetic_test") or r.get("provenance", {}).get("synthetic_test") for r in observations + gt_rows) else "offline_evaluation",
                  complementarity=complementarity(observations, gt_rows,
                      translation_threshold_m=translation_threshold_m,
                      rotation_threshold_deg=rotation_threshold_deg))
    if output_rows is not None:
        result["system"] = system_metrics(output_rows, gt_rows,
            translation_threshold_m=translation_threshold_m,
            rotation_threshold_deg=rotation_threshold_deg, max_output_age_s=max_output_age_s)
        result["corrections"] = correction_metrics(output_rows, observations, gt_rows,
            translation_threshold_m=translation_threshold_m,
            rotation_threshold_deg=rotation_threshold_deg)
        if events is not None:
            result["recovery"] = recovery_metrics(events, result["system"]["per_query"],
                                                   consecutive_successes=consecutive_successes)
    elif events is not None:
        raise ValueError("Recovery events require system outputs")
    return result


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--outputs", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--translation-m", type=float, default=1.)
    parser.add_argument("--rotation-deg", type=float, default=30.)
    parser.add_argument("--max-output-age-s", type=float)
    parser.add_argument("--consecutive-successes", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = {name: getattr(args, name) for name in ("observations", "ground_truth", "outputs", "events")}
    if args.output.resolve() in {p.resolve() for p in paths.values() if p is not None}:
        parser.error("Output must not overwrite an input file")
    data = {name: read_jsonl(path) if path is not None else None for name, path in paths.items()}
    result = evaluate(data["observations"], data["ground_truth"],
        output_rows=data["outputs"], events=data["events"],
        translation_threshold_m=args.translation_m, rotation_threshold_deg=args.rotation_deg,
        max_output_age_s=args.max_output_age_s, consecutive_successes=args.consecutive_successes)
    result["input_provenance"] = {
        name: dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for name, path in paths.items() if path is not None}
    atomic_json(args.output, result)
    print(json.dumps(dict(output=str(args.output), result_type=result["result_type"],
                          evaluation_instants=result["complementarity"]["aggregate"]["evaluation_instants"])))


if __name__ == "__main__":
    main()
