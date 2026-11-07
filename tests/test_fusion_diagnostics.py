"""Analytical fixtures test evaluator semantics, not localization performance."""
import json

import numpy as np
import pytest

from research3d.fusion.diagnostics import (
    complementarity, correction_metrics, main, read_jsonl,
)
from research3d.fusion.metrics import pose_error, recovery_metrics, system_metrics, validate_truth


IDENTITY = dict(source_id="synthetic_fixture", building_id="building_1",
                map_id="fixed_map", camera_id="left")


def pose(x=0., yaw=0.):
    angle = np.radians(yaw)
    out = np.eye(4)
    out[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                   [np.sin(angle), np.cos(angle)]]
    out[0, 3] = x
    return out.tolist()


def gt(i, time=None):
    return dict(IDENTITY, query_id=f"q{i}", timestamp_s=float(i if time is None else time),
                T_map_camera=pose(), synthetic_test=True)


def observation(source, i, xs=(0.,), time=None):
    timestamp = float(i if time is None else time)
    return dict(IDENTITY, schema_version=1, source=source, query_id=f"q{i}",
                capture_time_s=timestamp, completed_time_s=timestamp+.1,
                status="ok" if xs else "failed", synthetic_test=True,
                candidates=[dict(rank=rank, T_map_camera=pose(x), quality={"score": .5})
                            for rank, x in enumerate(xs, 1)],
                provenance=dict(image_sha256="a"*64, model_sha256="b"*64,
                                calibration_sha256="c"*64, synthetic_test=True))


def output(i, x=0., time=None, age=.2, valid=True, **kw):
    timestamp = float(i if time is None else time)
    return dict(IDENTITY, query_id=f"q{i}", timestamp_s=timestamp,
                pose_time_s=timestamp, output_time_s=timestamp+age, valid=valid,
                T_map_camera=pose(x) if valid else None, **kw)


def test_four_cells_missing_denominator_top3():
    truth = [gt(i) for i in range(5)]
    rows = [observation("ace", 0), observation("spvloc", 0),
            observation("ace", 1), observation("spvloc", 1, (3., 0., 4.)),
            observation("ace", 2, (3.,)), observation("spvloc", 2),
            observation("ace", 3, (3.,)), observation("spvloc", 3, (3.,))]
    result = complementarity(rows, truth)
    stats = result["aggregate"]
    assert stats["four_cells"] == dict(both_correct=1, ace_only_correct=1,
                                       spvloc_only_correct=1, both_wrong=2)
    assert stats["ace_top1_recall"] == .4
    assert stats["top1_union_oracle_recall"] == .6
    assert stats["potential_top1_gain_over_ace"] == .2
    assert stats["spvloc_top3_oracle_recall"] == .6
    assert stats["ace_missing_or_failed"] == 1
    assert stats["spvloc_fewer_than_three_candidates"] == 4


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(map_id="different_map"),
    lambda r: r.update(capture_time_s=10),
    lambda r: r["candidates"][0].update(rank=0),
    lambda r: r["candidates"][0].update(rank=True),
    lambda r: r["candidates"][0]["T_map_camera"][0].__setitem__(0, 2.),
    lambda r: r["candidates"][0]["T_map_camera"][0].__setitem__(3, float("nan")),
    lambda r: r["provenance"].update(model_sha256="unknown"),
    lambda r: r.update(status="failed"),
])
def test_bad_observations_fail_closed(mutation):
    row = observation("ace", 0)
    mutation(row)
    with pytest.raises(ValueError):
        complementarity([row], [gt(0)])


def test_same_query_requires_same_image_and_calibration():
    a, s = observation("ace", 0), observation("spvloc", 0)
    s["provenance"]["image_sha256"] = "d"*64
    with pytest.raises(ValueError, match="same query image"):
        complementarity([a, s], [gt(0)])


def test_missing_output_is_failure_and_no_trajectory_alignment():
    truth = [gt(i) for i in range(4)]
    result = system_metrics([output(0), output(1, x=10), output(2, valid=False)], truth,
                            translation_threshold_m=1., rotation_threshold_deg=30.)
    stats = result["aggregate"]
    assert stats["output_coverage"] == .5
    assert stats["global_success_coverage"] == .25
    assert stats["missing_or_invalid_outputs"] == 2
    assert stats["translation_m"]["p50"] == 5.
    assert stats["translation_m"]["p95"] == 9.5
    run = result["by_stream"][0]["failure_runs"][0]
    assert run["frames"] == 3 and run["duration_s"] == 2.
    assert run["right_censored"]


def test_pose_error_rotation_and_reject_reflection():
    assert pose_error(pose(3, 179), pose()) == pytest.approx((3, 179))
    invalid = pose()
    invalid[0][0] = -1
    with pytest.raises(ValueError, match="SO"):
        pose_error(invalid, pose())


def test_output_age_limit_and_stale_pose_rejected():
    result = system_metrics([output(0, age=2)], [gt(0)],
                            translation_threshold_m=1., rotation_threshold_deg=30.,
                            max_output_age_s=.5)
    assert result["aggregate"]["global_success_coverage"] == 0
    assert result["aggregate"]["output_coverage"] == 1
    stale = output(0)
    stale["pose_time_s"] = -.1
    with pytest.raises(ValueError, match="Stale pose"):
        system_metrics([stale], [gt(0)], translation_threshold_m=1., rotation_threshold_deg=30.)


def test_harmful_correction_and_delayed_candidate_capture_evaluation():
    truth = [gt(0), gt(1)]
    obs = [observation("ace", 0, (5.,))]
    out = output(1, x=5., T_before_correction=pose(),
                 accepted_observations=[dict(source="ace", query_id="q0", rank=1)])
    result = correction_metrics([out], obs, truth,
                                translation_threshold_m=1, rotation_threshold_deg=30)
    assert result["wrong_accepted_candidate_count"] == 1
    assert result["correct_to_wrong_count"] == 1
    del out["T_before_correction"]
    result = correction_metrics([out], obs, truth,
                                translation_threshold_m=1, rotation_threshold_deg=30)
    assert result["correct_to_wrong_count"] is None
    assert result["status"].startswith("PARTIAL")


def test_future_completion_cannot_be_used():
    obs = observation("ace", 0)
    obs["completed_time_s"] = 5
    out = output(0, accepted_observations=[dict(source="ace", query_id="q0", rank=1)])
    with pytest.raises(ValueError, match="before inference completed"):
        correction_metrics([out], [obs], [gt(0)],
                           translation_threshold_m=1, rotation_threshold_deg=30)


def test_recovery_includes_never_recovered_and_real_irregular_times():
    times = [0., .5, 2., 4., 6., 8., 9.]
    truth = [gt(i, t) for i, t in enumerate(times)]
    outputs = [output(i, time=t, x=0 if i in (1, 2, 3) else 4.)
               for i, t in enumerate(times)]
    system = system_metrics(outputs, truth, translation_threshold_m=1, rotation_threshold_deg=30)
    events = [dict(IDENTITY, event_id="recovered", kind="kidnapped", start_time_s=0., end_time_s=6.),
              dict(IDENTITY, event_id="failed", kind="kidnapped", start_time_s=6., end_time_s=9.)]
    result = recovery_metrics(events, system["per_query"], consecutive_successes=3)
    assert result["events"] == 2 and result["unrecovered"] == 1
    assert result["recovery_success_rate"] == .5
    assert result["event_details"][0]["recovery_time_s"] == 4.2
    assert result["event_details"][1]["recovery_time_s"] is None
    assert result["event_details"][1]["right_censored"]


def test_recovery_not_confirmed_if_final_output_arrives_after_window():
    truth = [gt(i) for i in range(4)]
    outputs = [output(i, age=2) for i in range(4)]
    system = system_metrics(outputs, truth, translation_threshold_m=1, rotation_threshold_deg=30)
    events = [dict(IDENTITY, event_id="e", kind="initialization", start_time_s=0., end_time_s=3.)]
    result = recovery_metrics(events, system["per_query"])
    assert result["unrecovered"] == 1


def test_recovery_requires_all_confirmation_outputs_to_have_arrived():
    truth = [gt(i) for i in range(5)]
    outputs = [output(i, age=20 if i == 0 else .2) for i in range(5)]
    system = system_metrics(outputs, truth, translation_threshold_m=1, rotation_threshold_deg=30)
    events = [dict(IDENTITY, event_id="e", kind="initialization", start_time_s=0., end_time_s=4.)]
    result = recovery_metrics(events, system["per_query"])
    assert result["event_details"][0]["recovery_time_s"] == 3.2
    assert result["event_details"][0]["first_correct_in_confirmed_streak_s"] == 1.


def test_rpe_never_bridges_missing_estimates():
    truth = [gt(i) for i in range(4)]
    outputs = [output(0), output(2), output(3, x=2.)]
    system = system_metrics(outputs, truth, translation_threshold_m=1, rotation_threshold_deg=30)
    rpe = system["by_stream"][0]["rpe_adjacent_evaluation_instants"]
    assert rpe["possible_pairs"] == 3
    assert rpe["valid_pairs"] == 1 and rpe["invalid_pairs"] == 2
    assert rpe["translation_m"]["rmse"] == 2.


def test_nan_threshold_cannot_turn_every_output_into_unexplained_failure():
    with pytest.raises(ValueError, match="nonfinite"):
        system_metrics([output(0)], [gt(0)], translation_threshold_m=float("nan"), rotation_threshold_deg=30)


def test_jsonl_duplicate_keys_and_nonfinite_rejected(tmp_path):
    path = tmp_path / "input.jsonl"
    for text in ('{"query_id":"a","query_id":"b"}\n', '{"value":NaN}\n'):
        path.write_text(text)
        with pytest.raises(ValueError):
            read_jsonl(path)


def test_cli_writes_atomic_traceable_synthetic_result(tmp_path, capsys):
    observations_path, gt_path = tmp_path / "observations.jsonl", tmp_path / "gt.jsonl"
    observations_path.write_text(json.dumps(observation("ace", 0)) + "\n")
    gt_path.write_text(json.dumps(gt(0)) + "\n")
    result_path = tmp_path / "report.json"
    main(["--observations", str(observations_path), "--ground-truth", str(gt_path),
          "--output", str(result_path)])
    result = json.loads(result_path.read_text())
    assert result["result_type"] == "synthetic_software_test"
    assert len(result["input_provenance"]["ground_truth"]["sha256"]) == 64
    assert result["complementarity"]["aggregate"]["evaluation_instants"] == 1
    assert not list(tmp_path.glob(".report.json.*"))
    assert "synthetic_software_test" in capsys.readouterr().out


def test_duplicate_truth_unknown_observation_and_rank_gaps_rejected():
    with pytest.raises(ValueError, match="globally unique"):
        validate_truth([gt(0), gt(0)])
    with pytest.raises(ValueError, match="Unknown"):
        complementarity([observation("ace", 1)], [gt(0)])
    row = observation("spvloc", 0, (0., 1.))
    row["candidates"][1]["rank"] = 3
    with pytest.raises(ValueError, match="ranks"):
        complementarity([row], [gt(0)])


def test_actual_replay_schema_to_offline_metrics_isolated_gt():
    from research3d.fusion.replay import run_replay
    from research3d.fusion.diagnostics import evaluate
    from pathlib import Path
    config_path = Path(__file__).resolve().parents[1] / "configs/fusion/experiment.json"
    config = json.loads(config_path.read_text())
    config = {**config["replay_defaults"], **config["variants"]["E_three_recovery"]}
    truth, motions, observations = [], [], []
    for i in range(5):
        truth_row = gt(i)
        truth_row["T_map_camera"] = pose(.1*i)
        truth.append(truth_row)
        obs = observation("ace", i, (.1*i,))
        obs["status"] = "VALID"
        obs["candidates"][0]["quality"] = {"pnp_inliers": 100}
        observations.append(obs)
        motions.append(dict(IDENTITY, schema_version=1, query_id=f"q{i}", source="macvo",
                            capture_time_s=float(i), completed_time_s=float(i)+.2,
                            valid=True, segment_id="segment0", T_odom_camera=pose(.1*i),
                            provenance=dict(obs["provenance"])))
    # Only motions and GT-free observations enter the backend.
    outputs, audit = run_replay(motions, observations, config)
    result = evaluate(observations, truth, output_rows=outputs)
    assert result["result_type"] == "synthetic_software_test"
    assert not audit["estimate_alignment_to_gt"]
    assert result["system"]["aggregate"]["valid_outputs"] == 3
    assert result["corrections"]["wrong_accepted_candidate_count"] == 0
    assert result["corrections"]["initialization_or_no_prior_pose_events"] == 1
    assert result["corrections"]["status"] == "COMPLETE"
