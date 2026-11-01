"""Frozen continuous RGB-D query and disjoint mapping views from real ReplicaCAD assets.

All observations are Habitat renders, not real recordings. Map GT is offline;
query poses are stored separately for explicitly GT-motion-conditioned papers.
"""
import argparse, json, math, os
from pathlib import Path
os.environ.setdefault('MAGNUM_LOG','quiet');os.environ.setdefault('HABITAT_SIM_LOG','quiet')
import numpy as np
from PIL import Image
from blocked_data import atomic


def main(scene,out,frames=600,references=128,mesh_only=False):
    if mesh_only:
        finish_mesh(scene,Path(out));return
    import habitat_sim,magnum as mn
    from habitat_sim.utils.common import quat_from_angle_axis
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'acceptance.json').exists(), 'Accepted sequence is immutable'
    src=Path('/data/fyp_3d/datasets/replica_independent')/scene
    original=json.loads((src/'habitat/replica_stage.stage_config.json').read_text())
    stage=dict(render_asset=str(src/'habitat/mesh_semantic.ply'),semantic_asset=str(src/'habitat/mesh_semantic.ply'),semantic_descriptor_filename=str(src/'habitat/info_semantic.json'),up=original.get('up',[0,0,1]),front=original.get('front',[0,1,0]),requires_lighting=False)
    atomic(out/'colored_replica.stage_config.json',stage)
    sc=habitat_sim.SimulatorConfiguration();sc.scene_id=str(out/'colored_replica.stage_config.json');sc.enable_physics=False
    sensors=[]
    for name,kind in [('rgb',habitat_sim.SensorType.COLOR),('depth',habitat_sim.SensorType.DEPTH),('instance',habitat_sim.SensorType.SEMANTIC)]:
        spec=habitat_sim.CameraSensorSpec();spec.uuid=name;spec.sensor_type=kind;spec.resolution=[270,480];spec.position=mn.Vector3(0,1.6,0);spec.hfov=mn.Deg(90);sensors.append(spec)
    ac=habitat_sim.agent.AgentConfiguration();ac.sensor_specifications=sensors
    sim=habitat_sim.Simulator(habitat_sim.Configuration(sc,[ac]));sim.seed(2089)
    # Recompute from the actual loaded mesh so the navmesh shares its coordinates.
    ns=habitat_sim.NavMeshSettings();ns.set_defaults();ns.agent_height=1.7;ns.agent_radius=.15
    assert sim.recompute_navmesh(sim.pathfinder,ns)
    labels=json.loads((src/'habitat/info_semantic.json').read_text())
    atomic(out/'instances.json',dict(scene=scene,source_semantic_information=labels,instance_id_policy='Source face object_id +2; invalid semantic pixels become background 0',source_mesh=str(src/'habitat/mesh_semantic.ply')))
    agent=sim.initialize_agent(0);K=np.array([[240.,0,239.5],[0,240.,134.5],[0,0,1.]])
    np.save(out/'K.npy',K)
    rows=[];hashes=set()
    import hashlib
    def save(split,i,pos,yaw):
        state=habitat_sim.AgentState();state.position=pos;state.rotation=quat_from_angle_axis(float(yaw),np.array([0,1,0]));agent.set_state(state)
        obs=sim.get_sensor_observations();T=np.asarray(sim._sensors['rgb']._sensor_object.node.absolute_transformation(),dtype=float)@np.diag([1,-1,-1,1])
        name=f'{i:06d}';folder=out/split;folder.mkdir(exist_ok=True)
        rgb=obs['rgb'][...,:3];digest=hashlib.sha256(rgb.tobytes()).hexdigest()
        if split=='reference': hashes.add(digest)
        else: assert digest not in hashes, 'Exact reference-query RGB duplicate'
        Image.fromarray(rgb).save(folder/(name+'.png'));np.savez_compressed(folder/(name+'.npz'),depth=obs['depth'].astype(np.float32),instance=remap_ids(obs['instance']))
        return T
    refs=[]
    for i in range(references):
        pos=sim.pathfinder.get_random_navigable_point();refs.append(save('reference',i,pos,2*np.pi*(i%16)/16))
    np.save(out/'reference_poses.npy',refs)
    # The route is frozen before any localization; constant <=4cm translation per frame.
    route=[];p=sim.pathfinder.get_random_navigable_point()
    while len(route)<frames+1:
        target=sim.pathfinder.get_random_navigable_point();path=habitat_sim.ShortestPath();path.requested_start=p;path.requested_end=target
        if not sim.pathfinder.find_path(path) or path.geodesic_distance<1:continue
        for a,b in zip(path.points[:-1],path.points[1:]):
            n=max(1,math.ceil(float(np.linalg.norm(b-a))/.04))
            route.extend(a+(b-a)*(i/n) for i in range(n))
        p=target
    query=[];yaw=None
    for i in range(frames):
        delta=np.asarray(route[i+1])-route[i]
        desired=math.atan2(-float(delta[0]),-float(delta[2]))
        yaw=desired if yaw is None else yaw+float(np.clip((desired-yaw+np.pi)%(2*np.pi)-np.pi,-.08,.08))
        query.append(save('query',i,route[i],yaw))
        if i%100==0:print('RENDER',scene,i,flush=True)
    poses=np.stack(query);assert np.isfinite(poses).all();assert np.max(np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1))<.081
    np.savez_compressed(out/'ground_truth_conditioned_motion_and_eval.npz',T_world_camera=poses,timestamp_s=np.arange(frames)*.1)
    nav=np.array(sim.pathfinder.build_navmesh_vertices());np.save(out/'navmesh_vertices.npy',nav)
    sim.close()
    finish_mesh(scene,out)


def remap_ids(values):
    raw=np.asarray(values,dtype=np.int64);return np.where((raw<0)|(raw>=2**31),0,raw+2).astype(np.int32)


def finish_mesh(scene,out):
    from plyfile import PlyData
    src=Path('/data/fyp_3d/datasets/replica_independent')/scene
    ply=PlyData.read(src/'habitat/mesh_semantic.ply');v=ply['vertex'].data
    positions=np.stack([v[k] for k in ['x','y','z']],axis=1)
    rgb=np.stack([v[k] for k in ['red','green','blue']],axis=1).astype(np.uint8)
    faces=[];ids=[]
    for row in ply['face'].data:
        polygon=row['vertex_indices'];obj=int(row['object_id'])
        for i in range(1,len(polygon)-1):faces.append([polygon[0],polygon[i],polygon[i+1]]);ids.append(obj)
    faces=np.asarray(faces,dtype=np.int64);ids=remap_ids(np.asarray(ids))
    cfg=json.loads((out/'colored_replica.stage_config.json').read_text());up=np.asarray(cfg['up'],float);front=np.asarray(cfg['front'],float)
    up/=np.linalg.norm(up);front/=np.linalg.norm(front);right=np.cross(front,up);right/=np.linalg.norm(right)
    rotation=np.stack([right,up,-front]);assert np.allclose(rotation.T@rotation,np.eye(3)) and np.isclose(np.linalg.det(rotation),1)
    vertices=(positions[faces].reshape(-1,3)@rotation.T).astype(float)
    triangles=np.arange(len(vertices)).reshape(-1,3)
    np.savez_compressed(out/'annotated_asset_mesh.npz',vertices=vertices,triangles=triangles,obj_ids=np.repeat(ids,3),vertex_colors=rgb[faces].reshape(-1,3))
    assert np.load(out/'ground_truth_conditioned_motion_and_eval.npz')['T_world_camera'].shape==(600,4,4)
    assert len(list((out/'query').glob('*.png')))==600
    atomic(out/'acceptance.json',dict(status='accepted',dataset='Original Replica',scene=scene,frames=600,references=128,simulated=True,source_instance_identity_preserved=True,triangles=len(triangles),HDR_Ptex_used=False,independent_scene=True,source_acquisition=str(src/'acquisition.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scene');p.add_argument('output');p.add_argument('--mesh-only',action='store_true');a=p.parse_args();main(a.scene,a.output,mesh_only=a.mesh_only)
