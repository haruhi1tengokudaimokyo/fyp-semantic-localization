"""Actual isolated subprocess failures distinguish data quarantine from transient retry."""

import json, sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
import worker
from core import db, register


@pytest.mark.parametrize("exit_code,expected", [(65, "FAILED"), (17, "RETRYING")])
def test_declared_nonretryable_payload_exit_keeps_failure_evidence(
    tmp_path, monkeypatch, exit_code, expected
):
    s = dict(
        task_id="data_quality_fixture",
        experiment_id="INFRASTRUCTURE_ONLY",
        dataset_id="no_scientific_data",
        command=[sys.executable, "-c", f"raise SystemExit({exit_code})"],
        depends_on=[],
        inputs=[],
        validation=[],
        max_attempts=3,
        nonretryable_exit_codes=[65],
        resources={"gpu": False},
    )
    register([s], tmp_path)
    with db(tmp_path) as c:
        c.execute("UPDATE tasks SET status='RUNNING',attempt=1")
    monkeypatch.setattr(worker, "resources", lambda: {"infrastructure_only": True})
    actual_sleep = worker.time.sleep
    monkeypatch.setattr(worker.time, "sleep", lambda _: actual_sleep(0.01))
    with pytest.raises(RuntimeError, match=f"Payload exit code {exit_code}"):
        worker.run(s["task_id"], tmp_path)
    with db(tmp_path) as c:
        row = c.execute("SELECT status,attempt,failure FROM tasks").fetchone()
        assert row["status"] == expected and row["attempt"] == 1
        assert str(exit_code) in json.loads(row["failure"])["message"]
    assert (tmp_path / "artifacts/data_quality_fixture.attempt1.failure.json").exists()
