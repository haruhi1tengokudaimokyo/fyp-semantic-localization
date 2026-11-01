"""Prevent completed results being relabelled as a changed scientific experiment."""

import copy, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
from core import db, register, fingerprint


def spec():
    return dict(
        task_id="isolated",
        experiment_id="R",
        dataset_id="scene",
        model_dataset_id="scene",
        command=["python", "method.py"],
        config={"batch": 5120},
        inputs=[],
        seed=1,
        validation=[],
        depends_on=[],
    )


def test_completed_command_change_requires_new_task_and_deduplicates_event(tmp_path):
    s = spec()
    register([s], tmp_path)
    with db(tmp_path) as c:
        c.execute("UPDATE tasks SET status='COMPLETED'")
    changed = copy.deepcopy(s)
    changed["command"] = ["python", "different.py"]
    register([changed], tmp_path)
    register([changed], tmp_path)
    with db(tmp_path) as c:
        row = c.execute("SELECT * FROM tasks").fetchone()
        assert json.loads(row["spec"]) == s and row["status"] == "COMPLETED"
        assert (
            c.execute(
                "SELECT COUNT(*) FROM events WHERE event='COMPLETED_SPEC_CHANGE_REJECTED'"
            ).fetchone()[0]
            == 1
        )


def test_running_worker_keeps_its_original_spec(tmp_path):
    s = spec()
    register([s], tmp_path)
    with db(tmp_path) as c:
        c.execute("UPDATE tasks SET status='RUNNING'")
    changed = copy.deepcopy(s)
    changed["config"]["batch"] = 1
    register([changed], tmp_path)
    with db(tmp_path) as c:
        assert json.loads(c.execute("SELECT spec FROM tasks").fetchone()[0]) == s


def test_runtime_library_environment_changes_fingerprint():
    s = spec()
    changed = copy.deepcopy(s)
    changed["env"] = {"LD_LIBRARY_PATH": "/isolated/library"}
    assert fingerprint(s) != fingerprint(changed)


def test_old_binary_needs_a_fresh_matching_hash_receipt(tmp_path):
    import hashlib, os, time
    from core import validate

    blob = tmp_path / "model.bin"
    blob.write_bytes(b"unchanged scientific checkpoint")
    os.utime(blob, (1, 1))
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"sha256": hashlib.sha256(blob.read_bytes()).hexdigest()})
    )
    s = {
        "validation": [
            {"path": str(receipt), "kind": "json"},
            {
                "path": str(blob),
                "fresh": False,
                "sha256_from_json": {"path": str(receipt), "key": "sha256"},
            },
        ]
    }
    assert validate(s, started=time.time())["accepted"]
    blob.write_bytes(b"same path but changed checkpoint")
    import pytest

    with pytest.raises(ValueError, match="content-hash receipt"):
        validate(s, started=time.time())
