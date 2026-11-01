"""Time checkpoints must not fabricate a baseline or research completion."""

import importlib.util
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "reproduction/progress.py"
s = importlib.util.spec_from_file_location("progress", p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


def test_unknown_or_partial_total_does_not_trigger():
    assert m.due_milestones({}, None, 1e8) == []
    assert (
        m.due_milestones(
            {},
            dict(
                scope="ACE_queue",
                representative_core_measurements_complete=True,
                wall_seconds=400,
                start_epoch=0,
            ),
            1e8,
        )
        == []
    )
    assert (
        m.due_milestones(
            {},
            dict(
                scope="four_paper_total",
                representative_core_measurements_complete=False,
                wall_seconds=400,
                start_epoch=0,
            ),
            1e8,
        )
        == []
    )


def test_time_boundaries_once_and_catch_up_after_offline():
    b = dict(
        scope="four_paper_total",
        representative_core_measurements_complete=True,
        wall_seconds=400,
        start_epoch=1000,
    )
    assert m.due_milestones({}, b, 1099.99) == []
    assert m.due_milestones({}, b, 1100) == [25]
    assert m.due_milestones({"emitted": [25]}, b, 1200) == [50]
    assert m.due_milestones({"emitted": [25]}, b, 1600) == [50, 75, 100]
    assert m.due_milestones({"emitted": [25, 50, 75, 100]}, b, 1600) == []
