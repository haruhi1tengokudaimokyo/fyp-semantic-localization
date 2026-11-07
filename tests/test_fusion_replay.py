"""Safety/causality contracts, not simulated localization benchmark results."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from research3d.fusion.contracts import validate_observation, validate_motion
from research3d.fusion.protocol import validate_config
from research3d.fusion.replay import run_replay


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/fusion/experiment.json").read_text())
IDENTITY = dict(source_id="synthetic_unit_test", building_id="house", map_id="map_v1", camera_id="left")
PROVENANCE = dict(image_sha256="a" * 64, calibration_sha256="b" * 64, model_sha256="c" * 64)


def T(x):
    t = np.eye(4)
    t[0, 3] = x
    return t.tolist()


def motion(i, x=None, **kw):
    return dict(schema_version=1, source="macvo", **IDENTITY, query_id=f"q{i}", capture_time_s=float(i),
                completed_time_s=float(i) + .1, valid=True, segment_id="s0",
                T_odom_camera=T(i if x is None else x), provenance=PROVENANCE.copy(), **kw)


def observation(i, xs, source="ace", delay=.05):
    return dict(schema_version=1, source=source, **IDENTITY, query_id=f"q{i}",
                capture_time_s=float(i), completed_time_s=float(i) + delay, status="VALID",
                provenance=PROVENANCE.copy(),
                candidates=[dict(rank=r + 1, T_map_camera=T(x),
                                 quality={"pnp_inliers": 100} if source == "ace" else {"viewport_score": .9})
                            for r, x in enumerate(xs)])


def cfg(**kw):
    return {**copy.deepcopy(CONFIG["replay_defaults"]), "absolute_sources": ["ace", "spvloc"], **kw}


def test_config_and_no_test_tuning():
    validate_config(copy.deepcopy(CONFIG))
    bad = copy.deepcopy(CONFIG)
    bad["data"]["development_scene_ids"].append(3250)
    with pytest.raises(ValueError):
        validate_config(bad)


def test_late_observation_attaches_capture_then_propagates():
    rows, _ = run_replay([motion(0), motion(1), motion(2)], [observation(0, [10], delay=1.5)], cfg(confirmation_frames=1))
    assert not rows[0]["valid"] and not rows[1]["valid"]
    assert rows[2]["T_map_camera"][0][3] == pytest.approx(12)
    assert rows[2]["accepted_observations"][0]["query_id"] == "q0"


def test_future_capture_never_used_for_earlier_estimate():
    m = [motion(0), motion(1)]
    m[0]["completed_time_s"] = 1.2
    m[1]["completed_time_s"] = 1.3
    rows, _ = run_replay(m, [observation(1, [11])], cfg(confirmation_frames=1))
    assert not rows[0]["valid"] and rows[1]["valid"]


def test_correlated_sources_same_image_do_not_count_as_multiple_frames():
    rows, _ = run_replay([motion(0)], [observation(0, [10]), observation(0, [10], "spvloc")], cfg(confirmation_frames=2))
    assert not rows[0]["valid"]


def test_two_room_modes_are_not_averaged_or_arbitrarily_chosen():
    obs = [observation(i, [i + 10, i + 30], "spvloc") for i in range(3)]
    rows, audit = run_replay([motion(i) for i in range(3)], obs, cfg())
    assert not any(r["valid"] for r in rows)
    assert audit["decisions"][-1]["recovery_modes"] == 2


def test_recovery_is_not_locked_to_wrong_old_location():
    m = [motion(i) for i in range(7)]
    obs = [observation(i, [i + (0 if i < 3 else 20)]) for i in range(7)]
    rows, _ = run_replay(m, obs, cfg(confirmation_frames=2, conflicts_before_lost=2))
    assert rows[1]["valid"]
    assert rows[4]["status"] == "RECOVERING"
    assert rows[5]["T_map_camera"][0][3] == pytest.approx(25)


def test_lost_motion_breaks_anchor_no_interpolation():
    m = [motion(i) for i in range(4)]
    m[1].update(valid=False, T_odom_camera=None)
    m[2].update(segment_id="s1", T_odom_camera=T(0))
    m[3].update(segment_id="s1", T_odom_camera=T(1))
    rows, _ = run_replay(m, [observation(0, [10])], cfg(confirmation_frames=1))
    assert rows[0]["valid"] and all(not r["valid"] for r in rows[1:])


def test_wrong_map_and_wrong_image_rejected():
    for mutate in [lambda o: o.update(map_id="wrong"), lambda o: o["provenance"].update(image_sha256="d" * 64)]:
        ob = observation(0, [10])
        mutate(ob)
        with pytest.raises(ValueError):
            run_replay([motion(0)], [ob], cfg())


def test_truth_and_reflection_rejected():
    ob = observation(0, [10])
    ob["candidates"][0]["quality"]["is_correct"] = True
    with pytest.raises(ValueError):
        validate_observation(ob)
    ob = observation(0, [10])
    ob["candidates"][0]["T_map_camera"][0][0] = -1
    with pytest.raises(ValueError):
        validate_observation(ob)


def test_stale_results_do_not_relocalize():
    rows, audit = run_replay([motion(i) for i in range(5)], [observation(0, [10], delay=4)], cfg(confirmation_frames=1))
    assert not any(x["valid"] for x in rows)
    assert audit["decisions"][-1]["rejected"][0]["reason"] == "stale_observation"


def test_only_one_absolute_source_selected_per_output():
    rows, _ = run_replay([motion(0)], [observation(0, [10]), observation(0, [10.1], "spvloc")], cfg(confirmation_frames=1))
    assert len(rows[0]["accepted_observations"]) == 1


def test_no_global_ground_truth_origin_for_macvo_only():
    rows, _ = run_replay([motion(0), motion(1)], [], cfg(absolute_sources=[]))
    assert not any(x["valid"] for x in rows)
    assert all(x["local_odometry_valid"] for x in rows)


def test_late_observation_cannot_cross_invalid_motion_even_if_segment_reused():
    m = [motion(i) for i in range(3)]
    m[1].update(valid=False, T_odom_camera=None)
    rows, audit = run_replay(m, [observation(0, [10], delay=2)], cfg(confirmation_frames=1))
    assert not any(x["valid"] for x in rows)
    assert audit["decisions"][-1]["rejected"][0]["reason"] == "broken_motion_chain"


def test_boolean_version_and_rank_not_accepted_as_one():
    ob = observation(0, [10])
    ob["schema_version"] = True
    with pytest.raises(ValueError):
        validate_observation(ob)
    ob = observation(0, [10])
    ob["candidates"][0]["rank"] = True
    with pytest.raises(ValueError):
        validate_observation(ob)


def test_motion_hashes_required_even_without_absolute_branch():
    m = motion(0)
    m["provenance"].pop("model_sha256")
    with pytest.raises((KeyError, ValueError)):
        validate_motion(m)


def test_recovery_support_cannot_accumulate_beyond_temporal_window():
    m = [motion(i) for i in range(5)]
    obs = [observation(i, [10 + i]) for i in [0, 2, 4]]
    rows, _ = run_replay(m, obs, cfg(confirmation_frames=3, hypothesis_ttl_s=2))
    assert not any(r["valid"] for r in rows)
