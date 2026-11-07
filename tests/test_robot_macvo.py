import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from research3d.robot_sim.prepare_ace import sha256
from research3d.robot_sim.run_macvo import load_stereo_manifest, ned_to_optical, evaluate_run


def stereo_capture(tmp_path):
    root = tmp_path / "capture"
    (root / "query").mkdir(parents=True)
    camera = dict(width=640, height=480, K=[[320, 0, 320], [0, 320, 240], [0, 0, 1]])
    (root / "world.sdf").write_text("world geometry")
    relative = np.eye(4)
    relative[0, 3] = .1
    manifest = dict(map_id="gazebo/test", map_sha256=sha256(root / "world.sdf"),
                    world_path="world.sdf", camera=camera, stereo_baseline_m=.1,
                    calibration=dict(rectified=True, T_left_right=relative.tolist()))
    (root / "map_manifest.json").write_text(json.dumps(manifest))
    rows = []
    for i in range(3):
        row = dict(query_id=f"query/{i}", timestamp_ns=i * 100000000,
                   right_timestamp_ns=i * 100000000, camera=camera)
        for side, field, key in (("left", "image_path", "image_sha256"), ("right", "right_image_path", "right_image_sha256")):
            p = root / "query" / f"{i}_{side}.png"
            Image.new("RGB", (640, 480), (i * 20, 40, 80)).save(p)
            row[field], row[key] = str(p.relative_to(root)), sha256(p)
        rows.append(row)
    write_rows(root, rows)
    return root, rows


def write_rows(root, rows):
    (root / "query/frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_stereo_native_calibration_no_frame_cap(tmp_path):
    root, _ = stereo_capture(tmp_path)
    _, _, rows, cam, baseline = load_stereo_manifest(root)
    assert len(rows) == 3 and baseline == .1 and cam["width"] == 640


def test_stereo_rejects_temporal_mismatch_and_gt(tmp_path):
    root, rows = stereo_capture(tmp_path)
    rows[1]["right_timestamp_ns"] += 1
    write_rows(root, rows)
    with pytest.raises(ValueError, match="synchronized"):
        load_stereo_manifest(root)
    rows[1]["right_timestamp_ns"] -= 1
    rows[1]["T_map_camera"] = np.eye(4).tolist()
    write_rows(root, rows)
    with pytest.raises(ValueError, match="supervision"):
        load_stereo_manifest(root)


def test_stereo_rejects_duplicate_time_and_hash_change(tmp_path):
    root, rows = stereo_capture(tmp_path)
    rows[1]["timestamp_ns"] = rows[1]["right_timestamp_ns"] = 0
    write_rows(root, rows)
    with pytest.raises(ValueError, match="nonmonotonic"):
        load_stereo_manifest(root)
    rows[1]["timestamp_ns"] = rows[1]["right_timestamp_ns"] = 100000000
    rows[0]["image_sha256"] = "0" * 64
    write_rows(root, rows)
    with pytest.raises(ValueError, match="checksum"):
        load_stereo_manifest(root)


def test_ned_optical_basis_and_identity():
    T = np.eye(4)
    T[:3, 3] = [1, 2, 3]  # native NED: forward/right/down
    cv = ned_to_optical(T)
    np.testing.assert_allclose(cv[:3, 3], [2, 3, 1])
    np.testing.assert_allclose(cv[:3, :3], np.eye(3))


def test_stereo_rejects_baseline_sign_or_rotation(tmp_path):
    root, _ = stereo_capture(tmp_path)
    p = root / "map_manifest.json"
    m = json.loads(p.read_text())
    m["calibration"]["T_left_right"][0][3] = -.1
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="extrinsic"):
        load_stereo_manifest(root)


def test_independent_metrics_require_saved_hash_and_exact_gt(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    traj = np.tile(np.eye(4), (3, 1, 1))
    traj[:, 0, 3] = [0, 1, 2]
    np.savez_compressed(run / "trajectory.npz", T_odom_camera=traj, lost=[False] * 3,
                        query_ids=np.array(["q0", "q1", "q2"]), timestamp_ns=np.array([0, 1, 2]))
    (run / "protocol.json").write_text(json.dumps(dict(map_id="m")))
    (run / "summary.json").write_text(json.dumps(dict(trajectory_sha256=sha256(run / "trajectory.npz"))))
    truth = tmp_path / "gt.jsonl"
    truth.write_text("".join(json.dumps(dict(query_id=f"q{i}", timestamp_ns=i, T_map_camera=T.tolist())) + "\n" for i, T in enumerate(traj)))
    result = evaluate_run(run, truth)
    assert result["ate_se3_aligned_rmse_m"] < 1e-12 and result["GT_online"] is False
    assert result["world_absolute_localization"] is False
    truth.write_text(truth.read_text().replace('"q2"', '"q3"'))
    with pytest.raises(ValueError, match="identity"):
        evaluate_run(run, truth)
