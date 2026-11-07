"""Geometry and rejection tests; these synthetic fixtures are not data evidence."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from research3d.fusion.capture import (
    CaptureConfig, camera_pose, densify_route, plan_routes, render_pair,
    stereo_positions, check_storage,
)
from research3d.fusion.data_validation import validate_capture, stereo_depth_check, structural_depth_check


def test_eye_baseline_is_real_translation_and_proper_rotation():
    for yaw in (0, 37, 90, 180, 270):
        left, right = stereo_positions([2, 1.3, 4], yaw, .1)
        a, b = camera_pose(left, yaw), camera_pose(right, yaw)
        relative = np.linalg.inv(a) @ b
        assert np.linalg.det(a[:3, :3]) == pytest.approx(1)
        np.testing.assert_allclose(relative[:3, :3], np.eye(3), atol=1e-10)
        np.testing.assert_allclose(relative[:3, 3], [.1, 0, 0], atol=1e-10)
    with pytest.raises(ValueError, match="nonzero"):
        stereo_positions([0, 0, 0], 0, 0)


def test_route_densification_bounds_translations_and_rotations():
    cfg = CaptureConfig(fps=10., angular_speed_deg_s=45.)
    poses = densify_route(np.array([[0., 0, 0], [.25, 0, 0], [.25, 0, .25]]), cfg)
    translations = np.array([x[0] for x in poses])
    yaws = np.array([x[1] for x in poses])
    assert np.max(np.linalg.norm(np.diff(translations, axis=0), axis=1)) <= .025 + 1e-9
    assert np.max(np.abs((np.diff(yaws)+180) % 360-180)) <= 4.5 + 1e-9


def test_reference_query_graph_routes_are_disjoint():
    points = [dict(x=x*.25, y=0., z=z*.25) for x in range(5) for z in range(5)]
    routes = plan_routes(points, CaptureConfig())
    assert len(routes["reference"]) >= 3 and len(routes["query"]) >= 3
    assert not set(map(tuple, routes["reference"])) & set(map(tuple, routes["query"]))


def test_final_event_supplies_both_eyes_not_two_different_times():
    class Controller:
        cameras = []
        sequence = 0

        def step(self, **params):
            self.sequence += 1
            camera = {"position": params["position"], "rotation": params["rotation"], "fieldOfView": params["fieldOfView"]}
            if params["action"] == "AddThirdPartyCamera":
                self.cameras.append(camera)
            else:
                self.cameras[params["thirdPartyCameraId"]] = camera
            return SimpleNamespace(metadata={"lastActionSuccess": True, "sequenceId": self.sequence,
                                             "thirdPartyCameras": self.cameras.copy()},
                                   third_party_camera_frames=[np.full((240, 320, 3), self.sequence, np.uint8) for _ in self.cameras],
                                   third_party_depth_frames=[np.ones((240, 320), np.float32) for _ in self.cameras])
    controller = Controller()
    event, images, _, _ = render_pair(controller, [0, 1.3, 0], 0., CaptureConfig(width=320, height=240), False)
    assert event.metadata["sequenceId"] == 2
    assert images[0][0, 0, 0] == images[1][0, 0, 0] == 2


def test_depth_reprojection_requires_metric_stereo_consistency():
    depth = np.full((240, 320), 2., np.float32)
    K = np.array([[156, 0, 159.5], [0, 156, 119.5], [0, 0, 1]])
    correct = stereo_depth_check(depth, depth, K, .1)
    assert correct["samples"] > 100
    assert correct["p95_depth_residual_m"] == 0
    assert correct["median_disparity_px"] == pytest.approx(7.8)
    wrong_units = stereo_depth_check(depth, depth * 1000, K, .1)
    assert wrong_units["samples"] == 0


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _fixture(root):
    config = CaptureConfig(width=320, height=240, baseline_m=.1, fps=10.)
    _write(root / "calibration.json", config.calibration("map1"))
    _write(root / "manifest.json", dict(map_id="map1", captured_frames=dict(reference=3, query=3),
                                       config=dict(translation_speed_m_s=.25, angular_speed_deg_s=45), pilot_truncated=True))
    _write(root / "map/annotation_3d.json", {"planes": [{"type": x} for x in ("wall", "floor", "ceiling")]})
    _write(root / "map/procthor_house.json", {"rooms": [], "walls": [dict(polygon=[
        dict(x=-10, y=0, z=2), dict(x=10, y=0, z=2), dict(x=-10, y=4, z=2), dict(x=10, y=4, z=2)])]})
    for split, x in (("reference", 0.), ("query", 1.)):
        rows, offline = [], []
        for index in range(3):
            identity = dict(source_id="procthor", building_id="b1", map_id="map1", camera_id="left",
                            query_id=f"{split}/{index}", frame_id=f"{split}/{index}", timestamp_s=index/10)
            row = dict(identity, capture_time_s=index/10, render_completed_time_s=index,
                       event_sequence_id=index+1, same_event_stereo=True)
            left, right = stereo_positions([x+index*.02, 1.3, 0], 0, .1)
            off = dict(identity, T_map_camera=camera_pose(left, 0).tolist(), T_map_right=camera_pose(right, 0).tolist(),
                       provenance=dict(source="synthetic unit fixture", role="offline",
                                       calibration_sha256=hashlib.sha256((root / "calibration.json").read_bytes()).hexdigest(),
                                       synthetic_test=True))
            instance_path = f"offline/{split}/instance/{index}.png"
            (root / instance_path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((240, 320, 3), [1, 2, 3], np.uint8)).save(root / instance_path)
            dictionary_path = f"offline/{split}/dictionary/{index}.json"
            _write(root / dictionary_path, {"wall|0": [1, 2, 3]})
            off.update(left_instance=instance_path, instance_dictionary=dictionary_path)
            for eye in ("left", "right"):
                relative = f"online/{split}/{eye}/{index}.png"
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((240, 320, 3), index, np.uint8)).save(path)
                row[f"{eye}_rgb"] = relative
                row[f"{eye}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                depth = f"offline/{split}/{eye}/{index}.npy"
                (root / depth).parent.mkdir(parents=True, exist_ok=True)
                np.save(root / depth, np.full((240, 320), 2., np.float32))
                off[f"{eye}_depth"] = depth
            rows.append(row)
            offline.append(off)
        (root / "online" / split / "frames.jsonl").write_text("\n".join(map(json.dumps, rows)))
        (root / "offline" / f"{split}_poses.jsonl").write_text("\n".join(map(json.dumps, offline)))
    return root


def test_disk_validation_accepts_consistent_fixture(tmp_path):
    audit = validate_capture(_fixture(tmp_path))
    assert audit["passed"], audit
    assert audit["metrics"]["reference_query_min_rig_center_separation_m"] > .9
    assert not audit["metrics"]["formal_sequence_complete"]


@pytest.mark.parametrize("failure", ["gt_leak", "reflection", "zero_baseline", "image_corruption", "camera_mismatch",
                                     "map_coordinate_shift", "invented_structural_labels"])
def test_disk_validation_rejects_bad_capture(tmp_path, failure):
    root = _fixture(tmp_path)
    if failure == "gt_leak":
        path = root / "online/query/frames.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        rows[0]["T_map_camera"] = np.eye(4).tolist()
        path.write_text("\n".join(map(json.dumps, rows)))
    elif failure == "zero_baseline":
        path = root / "calibration.json"
        data = json.loads(path.read_text()); data["baseline_m"] = 0
        _write(path, data)
    elif failure in {"reflection", "camera_mismatch", "map_coordinate_shift"}:
        path = root / "offline/query_poses.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        if failure == "reflection":
            rows[0]["T_map_camera"][0][0] *= -1
        elif failure == "camera_mismatch":
            rows[0]["T_map_right"][0][3] += .2
        else:
            # Stereo is still internally consistent, but its map is wrong.
            rows[0]["T_map_camera"][0][3] += 100
            rows[0]["T_map_right"][0][3] += 100
        path.write_text("\n".join(map(json.dumps, rows)))
    elif failure == "invented_structural_labels":
        _write(root / "offline/query/dictionary/0.json", {"Sofa|0": [1, 2, 3]})
    else:
        (root / "online/query/left/0.png").write_bytes(b"not an image")
    assert not validate_capture(root)["passed"]


def test_storage_budget_never_deletes_existing_files(tmp_path):
    keep = tmp_path / "keep"
    keep.write_bytes(b"existing")
    with pytest.raises(RuntimeError, match="budget"):
        check_storage(tmp_path, CaptureConfig(max_new_gib=1e-10, min_free_gib=0), 1024)
    assert keep.read_bytes() == b"existing"
