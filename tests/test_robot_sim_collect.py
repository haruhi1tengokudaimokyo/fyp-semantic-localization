from types import SimpleNamespace as NS

import numpy as np
import pytest

from research3d.robot_sim.collect import (image_record, matrix, rgb_array,
                                        stamp_ns, synchronized_base)
from research3d.robot_sim.world import T_BASE_CAMERA_OPTICAL


def test_padded_rgb_rows_are_not_interpreted_as_pixels():
    msg=NS(encoding='rgb8',height=2,width=2,step=8,
           data=bytes([1,2,3,4,5,6,250,250,7,8,9,10,11,12,251,251]))
    assert rgb_array(msg).tolist()==[[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]


@pytest.mark.parametrize('encoding,pixel', [('bgr8',[30,20,10]),('bgra8',[30,20,10,0]),('rgba8',[10,20,30,0])])
def test_channel_order(encoding,pixel):
    assert rgb_array(NS(encoding=encoding,height=1,width=1,step=len(pixel),data=bytes(pixel))).tolist()==[[[10,20,30]]]


def test_unsupported_depth_or_short_buffer_not_silently_accepted():
    with pytest.raises(ValueError): rgb_array(NS(encoding='16UC1',height=1,width=1,step=2,data=b'00'))
    with pytest.raises(ValueError): rgb_array(NS(encoding='rgb8',height=2,width=2,step=6,data=b'000'))


def test_world_base_to_optical_composition():
    s=np.sqrt(.5)
    T=matrix(NS(translation=NS(x=2,y=3,z=.1),rotation=NS(x=0,y=0,z=s,w=s)))
    optical=T@np.asarray(T_BASE_CAMERA_OPTICAL)
    np.testing.assert_allclose(optical[:3,3],[2,3.2,.6],atol=1e-12)
    np.testing.assert_allclose(optical[:3,2],[0,1,0],atol=1e-12)
    assert np.isclose(np.linalg.det(optical[:3,:3]),1.)
    assert stamp_ns(NS(sec=2,nanosec=17))==2_000_000_017


def test_teleport_transition_cannot_label_old_renderer_frame():
    old=np.eye(4);new=np.eye(4);new[0,3]=2
    old[2,3]=new[2,3]=.1
    history=[(t*1_000_000, old if t<100 else new) for t in range(0,201,20)]
    target=[2,0,.1,0]
    assert synchronized_base(history,100_000_000,target) is None
    assert synchronized_base(history,200_000_000,target) is None
    history.extend((t*1_000_000,new) for t in range(220,301,20))
    np.testing.assert_array_equal(synchronized_base(history,300_000_000,target),new)


def test_pose_gap_and_too_far_timestamp_are_rejected():
    T=np.eye(4)
    assert synchronized_base([(0,T)],16_000_000) is None
    assert synchronized_base([(0,T),(200_000_000,T)],200_000_000,[0,0,.1,0]) is None
    assert synchronized_base([(0,T)],10_000_000) is not None


def test_stereo_pairing_and_online_record_do_not_leak_pose():
    args=dict(query_id='query/000001',timestamp_ns=100,image_path='query/rgb/1.png',image_sha256='a'*64,
              camera={'K':[[320,0,320],[0,320,240],[0,0,1]]},map_id='known_map',
              right_image_path='query/rgb_right/1.png',right_image_sha256='b'*64,right_timestamp_ns=100)
    row=image_record(**args)
    assert not any('pose' in k.lower() or k.startswith('T_') or k=='gt' for k in row)
    with pytest.raises(ValueError): image_record(**dict(args,right_timestamp_ns=101))
    with pytest.raises(ValueError): image_record(**dict(args,right_image_path=args['image_path']))
