import json
import math
import xml.etree.ElementTree as ET

from research3d.robot_sim.world import build_world, procthor_structure, reference_poses, T_BASE_CAMERA_OPTICAL
from research3d.robot_sim.sim_bridge import bridge_command


def test_physics_and_online_sensor_contract(tmp_path):
    sdf=build_world(tmp_path)
    world=ET.parse(sdf).getroot().find('world')
    robot=world.find("model[@name='robot']")
    assert robot.find('static') is None
    assert len(robot.findall("joint[@type='revolute']")) == 2
    joint_plugin=robot.find("plugin[@name='gz::sim::systems::JointStatePublisher']")
    assert joint_plugin.findtext('topic') == '/robot/joint_states'
    camera=robot.find("link[@name='base_link']/sensor[@name='camera']")
    assert camera.findtext('gz_frame_id') == 'camera_optical_frame'
    assert math.isclose(float(camera.findtext('camera/horizontal_fov')), math.pi/2)
    for name in ('odom_topic','tf_topic'):
        assert robot.find("plugin[@name='gz::sim::systems::DiffDrive']").findtext(name).startswith('/offline/')
    online=' '.join(bridge_command())
    assert '/model/robot/pose' not in online
    assert 'Odometry' not in online
    assert '/model/robot/pose' in ' '.join(bridge_command(True))


def test_optical_basis_and_map_manifest(tmp_path):
    build_world(tmp_path)
    # Optical right=-base_y, down=-base_z, forward=base_x.
    assert [r[0] for r in T_BASE_CAMERA_OPTICAL[:3]] == [0,-1,0]
    assert [r[1] for r in T_BASE_CAMERA_OPTICAL[:3]] == [0,0,-1]
    assert [r[2] for r in T_BASE_CAMERA_OPTICAL[:3]] == [1,0,0]
    m=json.loads((tmp_path/'map_manifest.json').read_text())
    assert len(m['world_sha256']) == 64
    assert {'floor','wall','door','window','furniture','sign'} <= {x['category'] for x in m['objects']}
    assert len(list((tmp_path/'textures').glob('*.png'))) == 4


def test_source_door_window_holes_are_not_filled(tmp_path):
    polygon=[dict(x=0,y=0,z=0),dict(x=5,y=0,z=0),dict(x=0,y=3,z=0),dict(x=5,y=3,z=0)]
    house=dict(rooms=[dict(floorPolygon=[dict(x=0,z=0),dict(x=5,z=0),dict(x=5,z=4),dict(x=0,z=4)])],
               walls=[dict(id='a',polygon=polygon),dict(id='b',polygon=polygon)],
               doors=[dict(id='door',wall0='a',wall1='b',holePolygon=[dict(x=1,y=0),dict(x=2,y=2.1)])],
               windows=[dict(id='window',wall0='a',wall1='b',holePolygon=[dict(x=3,y=1),dict(x=4,y=2)])])
    p=tmp_path/'house.json';p.write_text(json.dumps(house))
    _,objects,source=procthor_structure(p,[tmp_path/'texture.png']*4)
    walls=[o for o in objects if o['category']=='wall']
    assert source['walls_unique_nonempty']==1
    assert len(source['openings'])==2
    def solid_at(x,z):
        return any(abs(x-o['center_m'][0])<o['size_m'][0]/2 and abs(z-o['center_m'][2])<o['size_m'][2]/2 for o in walls)
    assert not solid_at(1.5,1.)  # Open door is traversable.
    assert not solid_at(3.5,1.5) # Window retains its correct hole.
    assert solid_at(3.5,.5)     # Wall below the window is retained.
    assert solid_at(1.5,2.5)    # Lintel above the door is retained.


def test_mapping_viewpoints_are_separate_and_unique():
    poses=reference_poses()
    assert len(poses)==192 and len(set(poses))==192
    assert all(.5<x<8 and 4.7<y<10 and z==.101 for x,y,z,yaw in poses)
