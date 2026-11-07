from types import SimpleNamespace as NS

import pytest

from research3d.robot_sim.replay_sensors import merge_events,sensor_message,stamp_parts,validate_sensor


def imu(t):
    return dict(timestamp_ns=t,frame_id='base_link',angular_velocity=[0.,0.,.2],
                linear_acceleration=[0.,0.,9.81],angular_velocity_covariance=[4e-6,0.,0.,0.,4e-6,0.,0.,0.,4e-6],orientation_excluded=True)


def joints(t):
    return dict(timestamp_ns=t,names=['left_wheel_joint','right_wheel_joint'],positions=[1.,2.],velocities=[3.,4.])


def message():
    return NS(header=NS(stamp=NS(sec=0,nanosec=0),frame_id=''),orientation=NS(x=0.,y=0.,z=0.,w=0.),
              angular_velocity=NS(x=0.,y=0.,z=0.),linear_acceleration=NS(x=0.,y=0.,z=0.))


def test_deterministic_clock_order_and_strict_stereo_sync():
    images=[dict(timestamp_ns=10,right_timestamp_ns=10),dict(timestamp_ns=30,right_timestamp_ns=30)]
    events=merge_events(images,[joints(10),joints(20)],[imu(10),imu(25)])
    assert [(t,k) for t,k,_ in events]==[(10,'joint_states'),(10,'imu'),(10,'stereo'),(20,'joint_states'),(25,'imu'),(30,'stereo')]
    with pytest.raises(ValueError): merge_events([dict(timestamp_ns=10,right_timestamp_ns=11)],[],[])
    with pytest.raises(ValueError): merge_events([dict(timestamp_ns=10,right_timestamp_ns=10,T_map_camera=[])],[],[])
    with pytest.raises(ValueError): merge_events([],[],[imu(20),imu(10)])


def test_imu_excludes_absolute_orientation_but_preserves_gyro():
    out=sensor_message('imu',imu(1_000_000_007),message)
    assert out.header.stamp.sec==1 and out.header.stamp.nanosec==7
    assert out.angular_velocity.z==.2
    assert out.orientation_covariance[0]==-1
    assert out.header.frame_id=='base_link'
    with pytest.raises(ValueError): validate_sensor(dict(imu(0),orientation=[0,0,0,1]),'imu')
    with pytest.raises(ValueError): validate_sensor(dict(imu(0),orientation_excluded=False),'imu')
    with pytest.raises(ValueError): validate_sensor(dict(imu(0),ground_truth={}), 'imu')


def test_actual_joint_rates_not_world_odometry():
    out=sensor_message('joint_states',joints(7),message)
    assert out.name==['left_wheel_joint','right_wheel_joint']
    assert out.velocity==[3.,4.]
    assert out.position==[1.,2.]
    with pytest.raises(ValueError): validate_sensor(dict(joints(7),T_map_base=[]),'joint_states')
    with pytest.raises(ValueError): validate_sensor(dict(joints(7),velocities=[1.]),'joint_states')
    with pytest.raises(ValueError): stamp_parts(1.)
    with pytest.raises(ValueError): stamp_parts(-1)
