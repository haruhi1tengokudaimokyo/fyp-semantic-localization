import numpy as np
from scipy.spatial.transform import Rotation
from motion.recovery import RecoveryGate


def test_recovery_decisions_and_poses_are_world_gauge_invariant():
    gauge = np.eye(4)
    gauge[:3, :3] = Rotation.from_euler("xyz", [20, 40, 70], degrees=True).as_matrix()
    gauge[:3, 3] = [5, -3, 1]
    first = RecoveryGate()
    second = RecoveryGate()
    for i in range(12):
        candidate = np.eye(4)
        candidate[0, 3] = i * 0.03
        motion = np.eye(4)
        motion[0, 3] = 0.03
        if i == 7:
            candidate[1, 3] = (
                1.0  # Same isolated false absolute hypothesis in both gauges.
            )
        a = first.step(motion, i > 0, candidate, 200)
        b = second.step(motion, i > 0, gauge @ candidate, 200)
        assert (
            a["status"] == b["status"]
            and a["accepted_absolute"] == b["accepted_absolute"]
        )
        if a["T_world_camera"] is not None:
            np.testing.assert_allclose(
                b["T_world_camera"], gauge @ a["T_world_camera"], atol=1e-10
            )


def test_missing_visual_inputs_are_explicit_lost_state_not_fake_confidence():
    gate = RecoveryGate()
    for _ in range(3):
        gate.step(np.eye(4), True, np.eye(4), 200)
    state = gate.step(np.eye(4), False, None, 0)
    assert (
        state["status"] == "LOST"
        and not state["accepted_absolute"]
        and state["pose_covariance"] is None
    )
    np.testing.assert_equal(state["T_world_camera"], np.eye(4))
