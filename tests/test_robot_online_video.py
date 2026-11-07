import numpy as np
import pytest

from research3d.robot_sim.make_online_video import available_count, candidate_base_points, first_origin


def test_available_count_uses_return_time_and_keeps_equal_time():
    rows = [{"received": 3.0}, {"received": 4.0}, {"received": 4.0}, {"received": 5.0}]
    assert available_count(rows, "received", 2.0) == 0
    assert available_count(rows, "received", 4.0) == 3
    with pytest.raises(ValueError):
        available_count([{"received": 2.0}, {"received": 1.0}], "received", 3.0)


def test_candidate_camera_to_base_preserves_all_spv_modes():
    rig = np.eye(4); rig[0, 3] = .2
    camera = np.eye(4); camera[0, 3] = 2.2
    row = {"candidate": {"status": "ok", "source": "official_spvloc",
        "refined_top1_T_map_camera_optical": camera.tolist(),
        "candidates": [{"rank": 1, "T_map_camera_optical": camera.tolist()}]}}
    points = candidate_base_points(row, rig)
    assert [p[0] for p in points] == ["refined_top1", "rank1"]
    assert np.allclose([p[1] for p in points], [[2., 0.], [2., 0.]])


def test_first_origin_does_not_take_future_output_with_old_pose_time():
    frames = [{"query_id": "a", "timestamp_ns": 2_000_000_000}]
    gt = np.eye(4); gt[0, 3] = 10.
    before = np.eye(4); before[0, 3] = 1.
    future = np.eye(4); future[0, 3] = 100.
    trace = [dict(timestamp_s=1.9, pose_time_s=1.8, local_valid=True, T_odom_base=before.tolist()),
             dict(timestamp_s=2.1, pose_time_s=1.9, local_valid=True, T_odom_base=future.tolist())]
    origin, audit = first_origin(frames, {"a": {"T_map_base": gt.tolist()}}, trace)
    assert np.isclose(origin[0, 3], 9.)
    assert audit["output_timestamp_s"] == 1.9
