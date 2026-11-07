import numpy as np

from research3d.robot_sim.evaluate import interpolate, statistics, path_statistics, recovery_statistics


def test_offline_association_does_not_bridge_missing_truth_or_extrapolate():
    a=np.eye(4);b=np.eye(4);b[0,3]=1
    assert interpolate(np.array([0.,1.]),[a,b],.5) is None
    assert interpolate(np.array([0.,.1]),[a,b],-.01) is None
    assert interpolate(np.array([0.,.1]),[a,b],.11) is None
    assert np.isclose(interpolate(np.array([0.,.1]),[a,b],.05)[0,3],.5)


def test_missing_predictions_are_failures_in_success_denominator():
    values=statistics([[.01,1.],[2.,1.]],4)
    assert values['coverage']==.5
    assert values['success_1m30deg']==.25
    assert statistics([],4)['success_1m30deg']==0


def test_teleport_is_not_counted_as_physical_path_or_interpolated_rotation():
    from scipy.spatial.transform import Rotation
    poses=[]
    for x in [0.,.1,5.,5.1]:
        T=np.eye(4);T[0,3]=x;poses.append(T)
    result=path_statistics([0,.1,.2,.3],poses)
    assert np.isclose(result['sampled_continuous_path_length_m'],.2)
    assert np.isclose(result['relocation_jump_distance_m'],4.9)
    rotated=np.eye(4);rotated[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix()
    assert interpolate(np.array([0.,.1]),[np.eye(4),rotated],.05) is None


def test_recovery_uses_third_output_availability_not_first_correct_capture():
    times=np.arange(0.,1.,.1);poses=[]
    for t in times:
        T=np.eye(4);T[0,3]=0 if t<.2 else 5;poses.append(T)
    rows=[dict(timestamp_s=float(t),pose_time_s=float(t),output_time_s=float(t+.03),absolute_error=[0.,0.]) for t in times]
    result=recovery_statistics([dict(type='kidnap',requested_at_timestamp_ns=150_000_000)],rows,times,poses)[0]
    assert result['success']
    assert np.isclose(result['confirmed_output_timestamp_s'],.43)
    assert np.isclose(result['recovery_time_s'],.23)
    assert np.isclose(result['first_correct_delay_s'],.03)


def test_one_event_cannot_claim_recovery_after_next_kidnapping():
    times=np.arange(0.,1.,.1);poses=[]
    for t in times:
        T=np.eye(4);T[0,3]=0 if t<.2 else (5 if t<.5 else 9);poses.append(T)
    rows=[dict(timestamp_s=float(t),pose_time_s=float(t),output_time_s=float(t+.01),absolute_error=None if t<.5 else [0.,0.]) for t in times]
    events=[dict(type='kidnap',requested_at_timestamp_ns=t) for t in [150_000_000,450_000_000]]
    result=recovery_statistics(events,rows,times,poses)
    assert result[0]['success'] is False and result[0]['censored']
    assert result[1]['success'] is True
