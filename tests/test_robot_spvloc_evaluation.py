"""Offline scoring must not promote oracle candidate choice to localization."""
import copy

import numpy as np
import pytest

from research3d.robot_sim.evaluate_spvloc import score_rows


def fixture_rows():
    frames = [dict(query_id=f"q{i}", timestamp_ns=i, map_id="map", image_sha256="image") for i in range(3)]
    truths = [dict(query_id=f"q{i}", timestamp_ns=i, T_map_camera=np.eye(4).tolist()) for i in range(3)]
    wrong = np.eye(4)
    wrong[0, 3] = 2.
    prediction = dict(query_id="q0", timestamp_ns=0, map_id="map", map_sha256="world", image_sha256="image",
                      status="ok", online_gt_used=False, refined_top1_T_map_camera_optical=wrong.tolist(),
                      candidates=[dict(rank=1, T_map_camera_optical=wrong.tolist(), viewport_score=.8),
                                  dict(rank=2, T_map_camera_optical=np.eye(4).tolist(), viewport_score=.6)])
    return frames, truths, [prediction]


def test_full_denominator_and_oracle_recall_do_not_change_top1():
    frames, truths, predictions = fixture_rows()
    result, rows = score_rows(frames, truths, predictions, "map", "world")
    assert result["top1"]["valid"] == 1
    assert result["top1"]["success_1m30deg"] == 0
    assert result["refined_top1"]["success_1m30deg"] == 0
    assert result["top3_oracle_candidate_recall"]["1m30deg"] == 1/3
    assert result["failure_counts"]["top1"]["missing_prediction"] == 2
    assert rows[0]["top1_error"] == [2., 0.]


def test_duplicate_and_identity_mismatches_fail_closed():
    frames, truths, predictions = fixture_rows()
    with pytest.raises(ValueError, match="duplicate"):
        score_rows(frames, truths, predictions*2, "map", "world")
    altered = copy.deepcopy(predictions)
    altered[0]["map_sha256"] = "different-world"
    with pytest.raises(ValueError, match="binding mismatch"):
        score_rows(frames, truths, altered, "map", "world")


def test_reflection_is_invalid_not_silently_projected_to_rotation():
    frames, truths, predictions = fixture_rows()
    predictions[0]["candidates"][0]["T_map_camera_optical"][0][0] = -1.
    result, rows = score_rows(frames, truths, predictions, "map", "world")
    assert result["top1"]["valid"] == 0
    assert rows[0]["top1_error"] is None
    assert "proper SO(3)" in rows[0]["top1_failure"]
