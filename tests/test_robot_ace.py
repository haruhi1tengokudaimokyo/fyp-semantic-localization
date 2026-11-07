"""ACE adapter boundaries: independent split, map binding, no online GT access."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from research3d.robot_sim.prepare_ace import export_scene, sha256, validate_export
from research3d.robot_sim.ace_worker import DatasetReadGuard, Worker, load_binding


CAM = dict(width=640, height=480, K=[[320, 0, 320], [0, 320, 240], [0, 0, 1]])


def capture(tmp_path):
    root = tmp_path / "capture"
    for split in ("reference", "query", "offline_gt"):
        (root / split).mkdir(parents=True)
    (root / "world.sdf").write_text("test scene geometry")
    m = dict(map_id="gazebo/test", map_sha256=sha256(root / "world.sdf"), world_path="world.sdf", camera=CAM)
    (root / "map_manifest.json").write_text(json.dumps(m))
    for split, color in (("reference", "red"), ("query", "blue")):
        image = root / split / "000.png"
        Image.new("RGB", (640, 480), color).save(image)
        row = dict(query_id=split + "/000", timestamp_ns=1, image_path=str(image.relative_to(root)), camera=CAM)
        if split == "reference":
            row["T_map_camera"] = np.eye(4).tolist()
        (root / split / "frames.jsonl").write_text(json.dumps(row) + "\n")
    (root / "offline_gt/query_poses.jsonl").write_text("must not be read")
    return root


def test_export_reference_only_and_changed_input(tmp_path):
    root = capture(tmp_path)
    out = tmp_path / "ace_scene"
    audit = export_scene(root, out, min_reference_frames=1)
    assert audit["query_gt_read"] is False
    assert len(list((out / "train/poses").glob("*"))) == 1
    assert not (out / "test").exists()
    np.testing.assert_allclose(np.loadtxt(out / "train/poses/frame-000000.pose.txt"), np.eye(4))
    validate_export(out)
    (root / "reference/frames.jsonl").write_text("[]\n")
    with pytest.raises(ValueError, match="changed"):
        validate_export(out)


def test_export_rejects_query_supervision_and_overlap(tmp_path):
    root = capture(tmp_path)
    p = root / "query/frames.jsonl"
    row = json.loads(p.read_text())
    row["T_map_camera"] = np.eye(4).tolist()
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="supervision"):
        export_scene(root, tmp_path / "e1", 1)
    del row["T_map_camera"]
    row["image_path"] = "reference/000.png"
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="overlap"):
        export_scene(root, tmp_path / "e2", 1)


def test_export_rejects_crossmap_and_uncentered_camera(tmp_path):
    root = capture(tmp_path)
    p = root / "reference/frames.jsonl"
    row = json.loads(p.read_text())
    row["map_id"] = "other"
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="another map"):
        export_scene(root, tmp_path / "e1", 1)
    del row["map_id"]
    row["camera"]["K"][0][2] += 2
    p.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="centered"):
        export_scene(root, tmp_path / "e2", 1)


class Predictor:
    def predict(self, observation):
        with Image.open(observation.rgb_path) as im:
            im.load()
        state = SimpleNamespace(
            hypotheses=[SimpleNamespace(T_world_camera=np.eye(4))],
            diagnostics=SimpleNamespace(status="estimated", inliers=150, median_reprojection_px=1.2, runtime_ms=3.0),
        )
        return state, None


def worker_request(tmp_path):
    root = capture(tmp_path)
    binding = dict(camera=CAM, map_id="gazebo/test", map_sha256="a" * 64, head_sha256="b" * 64, dataset_root=str(root))
    worker = Worker(binding, root / "query", Predictor(), install_guard=False)
    request = dict(op="localize", request_id="r", query_id="query/000", map_id="gazebo/test", timestamp_ns=1,
                   image_path=str(root / "query/000.png"), camera=CAM)
    return root, worker, request


def test_rpc_pose_provenance_and_timestamps(tmp_path):
    _, worker, request = worker_request(tmp_path)
    result = worker.localize(request)
    assert result["status"] == "ok" and result["inliers"] == 150
    assert result["covariance_se3"] is None and result["accepted"] is False
    assert result["image_sha256"] == sha256(request["image_path"])
    assert result["T_map_camera_optical"] == np.eye(4).tolist()
    with pytest.raises(ValueError, match="increase"):
        worker.localize(request)


@pytest.mark.parametrize("field", ["T_map_camera", "ground_truth", "pose", "segmentation", "gt_path"])
def test_rpc_rejects_supervision_fields(tmp_path, field):
    _, worker, request = worker_request(tmp_path)
    request[field] = "hidden"
    with pytest.raises(ValueError, match="unexpected"):
        worker.localize(request)


def test_rpc_rejects_path_escape_and_hidden_metadata(tmp_path):
    root, worker, request = worker_request(tmp_path)
    request["metadata"] = {"kidnapped": True}
    with pytest.raises(ValueError, match="metadata"):
        worker.localize(request)
    del request["metadata"]
    request["image_path"] = str(root / "reference/000.png")
    with pytest.raises(ValueError, match="online-root"):
        worker.localize(request)


def test_gt_read_guard_and_symlink(tmp_path):
    root = capture(tmp_path)
    guard = DatasetReadGuard([root])
    guard.allowed = (root / "query/000.png").resolve()
    guard("open", (str(guard.allowed), "rb", 0))
    with pytest.raises(PermissionError):
        guard("open", (str(root / "offline_gt/query_poses.jsonl"), "r", 0))
    link = root / "query/fake.png"
    link.symlink_to(root / "offline_gt/query_poses.jsonl")
    with pytest.raises(PermissionError):
        guard("open", (str(link), "rb", 0))


def test_binding_rejects_weight_tampering(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    head = tmp_path / "head.pt"
    head.write_bytes(b"scene-specific")
    (source / "ace_encoder_pretrained.pt").write_bytes(b"encoder")
    (source / "ace_trainer.py").write_bytes(b"trainer")
    binding = dict(schema="robot_ace_model_v1", status="TRAINED_NOT_LOCALIZATION_VALIDATED",
                   head_path=str(head), head_sha256=sha256(head), encoder_sha256=sha256(source / "ace_encoder_pretrained.pt"),
                   trainer_sha256=sha256(source / "ace_trainer.py"))
    p = tmp_path / "binding.json"
    p.write_text(json.dumps(binding))
    assert load_binding(p, source) == binding
    head.write_bytes(b"wrong-map-head")
    with pytest.raises(ValueError, match="head hash"):
        load_binding(p, source)


def test_gt_open_rejected_inside_inference_subprocess(tmp_path):
    root = capture(tmp_path)
    program = '''
import json, sys
from pathlib import Path
from research3d.robot_sim.ace_worker import Worker
root=Path(sys.argv[1])
cam=json.loads((root/'map_manifest.json').read_text())['camera']
class LeakingPredictor:
    def predict(self, observation):
        (root/'offline_gt/query_poses.jsonl').read_text()
        raise AssertionError('GT should never be readable')
w=Worker(dict(camera=cam,map_id='m',map_sha256='a'*64,head_sha256='b'*64,dataset_root=str(root)),root/'query',LeakingPredictor())
try:
    w.localize(dict(op='localize',map_id='m',timestamp_ns=1,query_id='q',image_path=str(root/'query/000.png'),camera=cam))
except PermissionError:
    print('GT_READ_BLOCKED')
else:
    raise AssertionError('leak guard failed')
'''
    result = subprocess.run([sys.executable, "-c", program, str(root)], text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "GT_READ_BLOCKED"


def test_export_tampered_training_pose_rejected(tmp_path):
    root = capture(tmp_path)
    out = tmp_path / "scene"
    export_scene(root, out, 1)
    (out / "train/poses/frame-000000.pose.txt").write_text("invalid pose")
    with pytest.raises(ValueError, match="training file changed"):
        validate_export(out)


def test_offline_ace_evaluation_has_no_gt_alignment(tmp_path):
    from research3d.robot_sim.evaluate_ace import evaluate
    root = capture(tmp_path)
    m = json.loads((root / "map_manifest.json").read_text())
    query = json.loads((root / "query/frames.jsonl").read_text())
    query["image_sha256"] = sha256(root / query["image_path"])
    (root / "query/frames.jsonl").write_text(json.dumps(query))
    truth = dict(query_id=query["query_id"], timestamp_ns=1, T_map_camera=np.eye(4).tolist())
    (root / "offline_gt/query_poses.jsonl").write_text(json.dumps(truth))
    shifted = np.eye(4)
    shifted[0, 3] = 2.
    run = tmp_path / "predicted"
    run.mkdir()
    prediction = dict(query_id=query["query_id"], timestamp_ns=1, map_id=m["map_id"], map_sha256=m["map_sha256"],
                      image_sha256=query["image_sha256"], status="ok", T_map_camera=shifted.tolist(), inliers=500, latency_ms=5)
    (run / "predictions.jsonl").write_text(json.dumps(prediction))
    metrics = evaluate(root, run)
    assert metrics["translation_rmse_m"] == 2.
    assert metrics["success_count"]["1m30deg"] == 0
    assert metrics["failure_count"]["1m30deg"] == 1
    assert metrics["pose_output_coverage"] == 1.0
