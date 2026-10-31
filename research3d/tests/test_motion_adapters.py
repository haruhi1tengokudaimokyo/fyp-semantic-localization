import numpy as np
import pytest
from motion.adapters import IndexedMotionConstraint


def test_missing_timestamps_and_lost_motion_cannot_enter_timed_fusion():
    c = IndexedMotionConstraint(0, 1, np.eye(4), True, "test")
    with pytest.raises(ValueError, match="physical sensor timestamps"):
        c.for_timed_estimator()
    c = IndexedMotionConstraint(0, 1, np.eye(4), False, "test", 100, 200)
    with pytest.raises(ValueError, match="lost-track"):
        c.for_timed_estimator()


def test_real_timestamps_preserve_metric_transform_without_fake_pose_covariance():
    T = np.eye(4)
    T[0, 3] = 0.37
    c = IndexedMotionConstraint(2, 3, T, True, "test", 1000000000, 1100000000)
    m = c.for_timed_estimator()
    np.testing.assert_equal(m.T_previous_current, T)
    assert (
        m.covariance_se3 is None
        and m.current_stamp_ns - m.previous_stamp_ns == 100000000
    )
    with pytest.raises(ValueError):
        IndexedMotionConstraint(2, 3, T, True, "test", 100, None)
