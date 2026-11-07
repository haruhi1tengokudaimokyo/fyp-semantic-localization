from research3d.robot_sim.camera_fault import blackout_active


def test_actual_24_4_second_start_boundary_uses_integer_sensor_clock():
    # Actual pilot evidence: 24.4 - 9.4 rounds below 15.0 in binary float.
    origin=9_400_000_000
    assert 24.4-9.4 < 15.0
    assert blackout_active(24_400_000_000,origin,15,35)
    assert not blackout_active(24_399_999_999,origin,15,35)
    assert blackout_active(44_399_999_999,origin,15,35)
    assert not blackout_active(44_400_000_000,origin,15,35)
