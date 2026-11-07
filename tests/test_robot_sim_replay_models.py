import copy
import numpy as np
import pytest

from research3d.robot_sim.replay_models import (adapt_absolute, adapt_motion,
    match_input, schedule)


def fixture():
    camera=dict(width=640,height=480,K=[[320,0,320],[0,320,240],[0,0,1]])
    manifest=dict(map_id='unit_test_map',map_sha256='a'*64,camera=camera,
                  source={'scene':'unit_test'},calibration={'stereo_baseline_m':.1})
    frames={f'query/{i}':dict(query_id=f'query/{i}',timestamp_ns=i*100_000_000,
                            image_sha256=str(i)*64,camera=camera) for i in range(3)}
    raw=[dict(query_id=q,timestamp_ns=f['timestamp_ns'],map_id=manifest['map_id'],map_sha256=manifest['map_sha256'],
              image_sha256=f['image_sha256'],status='ok',head_sha256='c'*64,T_map_camera_optical=np.eye(4).tolist(),
              inliers=100,latency_ms=200.) for q,f in frames.items()]
    return manifest,frames,raw


def test_observation_bindings_reject_other_map_image_time_and_truth():
    m,f,raw=fixture()
    good=adapt_absolute(raw,f,m,'ace')
    assert len(good)==3 and good[0]['candidates'][0]['quality']['pnp_inliers']==100
    for key,value in [('map_id','other'),('map_sha256','b'*64),('image_sha256','b'*64),('timestamp_ns',7),('ground_truth',{})]:
        bad=copy.deepcopy(raw);bad[0][key]=value
        with pytest.raises(ValueError): adapt_absolute(bad,f,m,'ace')


def test_lost_macvo_pose_not_interpolated_and_chain_restarts():
    m,f,_=fixture()
    raw=[dict(query_id=q,timestamp_ns=x['timestamp_ns'],map_id=m['map_id'],map_sha256=m['map_sha256'],
              status='lost' if i==1 else 'estimated',T_odom_camera=np.eye(4).tolist()) for i,(q,x) in enumerate(f.items())]
    times=[dict(query_id=q,timestamp_ns=x['timestamp_ns'],inference_ms=1300.) for q,x in f.items()]
    out=adapt_motion(raw,f,m,dict(checkpoint_sha256='d'*64,motion_checkpoint_sha256='e'*64),times)
    assert out[1]['valid'] is False and out[1]['T_odom_camera'] is None
    assert out[0]['segment_id']!=out[2]['segment_id']
    with pytest.raises(ValueError): adapt_motion(raw[:-1],f,m,{},times)


def test_real_service_backlog_not_relabelled_realtime():
    motion=[dict(source='macvo',capture_time_s=t,measured_service_s=1.3) for t in (0.,.1)]
    absolute=[dict(source='ace',capture_time_s=0.,measured_service_s=.2),dict(source='spvloc',capture_time_s=0.,measured_service_s=.4)]
    m,o=schedule(motion,absolute,['ace','spvloc'],'measured_service_fifo')
    assert m[0]['completed_time_s']==pytest.approx(1.3)
    assert o[0]['completed_time_s']==pytest.approx(1.5)
    assert o[1]['completed_time_s']==pytest.approx(1.9)
    assert m[1]['completed_time_s']==pytest.approx(3.2)
    assert m[1]['queued_service_wait_s']==pytest.approx(1.8)
    # Removing a branch removes its compute burden; original rows stay intact.
    m2,_=schedule(motion,absolute,[],'measured_service_fifo')
    assert m2[1]['completed_time_s']==pytest.approx(2.6)
    ideal,_=schedule(motion,absolute,['ace'],'offline_accuracy')
    assert ideal[1]['completed_time_s']==.1
    assert 'completed_time_s' not in motion[0]


def test_spvloc_refinement_keeps_original_ranks_without_gt_selection():
    m,f,_=fixture();T=np.eye(4);refined=np.eye(4);refined[0,3]=.3
    raw=[dict(query_id='query/0',timestamp_ns=0,map_id=m['map_id'],map_sha256=m['map_sha256'],image_sha256='0'*64,
              status='ok',checkpoint_sha256='f'*64,latency_s=.5,
              candidates=[dict(rank=i,T_map_camera_optical=T.tolist(),viewport_score=.9-i*.1) for i in (1,2,3)],
              refined_top1_T_map_camera_optical=refined.tolist())]
    out=adapt_absolute(raw,f,m,'spvloc')[0]
    assert out['candidates'][0]['T_map_camera'][0][3]==.3
    assert out['candidates'][1]['T_map_camera'][0][3]==0
    assert [x['rank'] for x in out['candidates']]==[1,2,3]
    assert out['candidates'][0]['quality']['viewport_score']==pytest.approx(.8)
