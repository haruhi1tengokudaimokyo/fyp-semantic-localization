"""Coordinate/ray checks for the official SPVLoc Gazebo adapter."""
import json

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

from research3d.robot_sim.run_spvloc import reflection_to_optical,crop_camera,online_frames,adapt_annotation_to_actual_world


@pytest.mark.parametrize("yaw",[0.,.5,-2.2])
def test_original_parse_camera_reflection_matches_ros_optical(yaw):
    # Reconstruct official parse_camera_info from physically defined forward/up.
    forward=Rotation.from_euler("z",yaw).apply([1.,0.,0.])
    up=np.array([0.,0.,1.])
    u=np.cross(forward,up)
    v=-np.cross(forward,u)
    spv=np.eye(4)
    spv[:3,:3]=np.stack([u,v,forward],axis=1)
    spv[:3,3]=[1200.,2300.,600.]
    result=reflection_to_optical(spv)
    optical=Rotation.from_euler("z",yaw).as_matrix()@np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    np.testing.assert_allclose(result[:3,:3],optical,atol=1e-12)
    np.testing.assert_allclose(result[:3,3],[1.2,2.3,.6])
    assert np.isclose(np.linalg.det(result[:3,:3]),1.)
    # Positive optical x goes camera-right; positive optical y goes down.
    assert np.dot(result[:3,1],[0,0,1])<0


def test_conversion_does_not_silently_flip_proper_input():
    with pytest.raises(ValueError,match="reflected"):
        reflection_to_optical(np.eye(4))


def test_center_crop_preserves_horizontal_and_vertical_rays():
    im=Image.fromarray(np.zeros((480,640,3),np.uint8))
    camera={"width":640,"height":480,"K":[[320,0,320],[0,320,240],[0,0,1]]}
    crop,k,metadata=crop_camera(im,camera)
    assert crop.size==(480,480)
    assert metadata["left"]==80 and metadata["top"]==0
    old_ray=np.array([(400-320)/320,(300-240)/320,1.])
    normalized_new=np.array([2*((400-80)/480-.5),2*(300/480-.5),1.])
    new_ray=np.linalg.inv(k)@normalized_new
    np.testing.assert_allclose(new_ray,old_ray,atol=1e-12)
    assert abs(metadata["horizontal_fov_deg"]-73.739795)<1e-5


def test_online_spvloc_rejects_query_pose_labels(tmp_path):
    path=tmp_path/"frames.jsonl"
    path.write_text(json.dumps({"query_id":"q1","gt_pose":[1,2,3]})+"\n")
    with pytest.raises(ValueError,match="GT field"):
        online_frames(path,tmp_path,"building")


def test_openroof_adapter_reindexes_planes_and_semantic_references(tmp_path):
    sdf=tmp_path/"world.sdf"
    sdf.write_text('<sdf><world><model name="floor"/><model name="robot"/><light name="ceiling"/></world></sdf>')
    annotation={"planes":[{"ID":0,"type":"floor"},{"ID":1,"type":"ceiling"},{"ID":2,"type":"wall"}],
                "planeLineMatrix":[[1,0],[1,1],[0,1]],"semantics":[{"ID":0,"planeID":[0,1,2]}]}
    result,audit=adapt_annotation_to_actual_world(annotation,{"world_path":str(sdf),"objects":[{"name":"floor","category":"floor"}]})
    assert audit["source_ceiling_plane_ids_removed"]==[1]
    assert result["semantics"][0]["planeID"]==[0,1]
    assert result["planeLineMatrix"]==[[1,0],[0,1]]
    assert result["planes"][1]=={"ID":1,"type":"wall"}
    assert len(annotation["planes"])==3  # original source is immutable
