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
    src=Path('/data/fyp_3d/datasets/replicacad')
    sc=habitat_sim.SimulatorConfiguration();sc.scene_dataset_config_file=str(src/'replicaCAD.scene_dataset_config.json');sc.scene_id=scene;sc.enable_physics=True
    sensors=[]
    for name,kind in [('rgb',habitat_sim.SensorType.COLOR),('depth',habitat_sim.SensorType.DEPTH),('instance',habitat_sim.SensorType.SEMANTIC)]:
        spec=habitat_sim.CameraSensorSpec();spec.uuid=name;spec.sensor_type=kind;spec.resolution=[270,480];spec.position=mn.Vector3(0,1.6,0);spec.hfov=mn.Deg(90);sensors.append(spec)
    ac=habitat_sim.agent.AgentConfiguration();ac.sensor_specifications=sensors
    sim=habitat_sim.Simulator(habitat_sim.Configuration(sc,[ac]));sim.seed(2089)
    assert sim.pathfinder.is_loaded
    mgr=sim.get_rigid_object_manager();instances=[]
    for i,handle in enumerate(sorted(mgr.get_object_handles())):
        obj=mgr.get_object_by_handle(handle);obj.semantic_id=i+2
        instances.append(dict(id=i+2,handle=handle,template=obj.creation_attributes.handle,T_world_object=np.asarray(obj.transformation).tolist(),semantic_category_id=obj.creation_attributes.semantic_id))
    assert len(instances)>5
    atomic(out/'instances.json',dict(scene=scene,objects=instances,background_id_policy='Unannotated stage/background excluded from object graph; floor sampled from actual navmesh'))
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
        Image.fromarray(rgb).save(folder/(name+'.png'));np.savez_compressed(folder/(name+'.npz'),depth=obs['depth'].astype(np.float32),instance=obs['instance'].astype(np.int32))
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


def asset_colors(mesh):
    color=np.asarray(mesh.visual.to_color().vertex_colors)
    if color.ndim!=2:
        # Actual constant PBR baseColorFactor, not an invented semantic color.
        factor=getattr(mesh.visual.material,'baseColorFactor',None)
        assert factor is not None
        color=np.tile(np.asarray(factor,dtype=np.uint8),(len(mesh.vertices),1))
    assert color.shape==(len(mesh.vertices),4)
    return color[:,:3]


def finish_mesh(scene,out):
    src=Path('/data/fyp_3d/datasets/replicacad')
    instances=json.loads((out/'instances.json').read_text())['objects']
    frames=len(list((out/'query').glob('*.png')));references=len(np.load(out/'reference_poses.npy'))
    assert frames==600 and references==128
    assert np.load(out/'ground_truth_conditioned_motion_and_eval.npz')['T_world_camera'].shape==(600,4,4)
    assert len(list((out/'query').glob('*.npz')))==600
    # Export actual transformed asset triangles, not a reconstruction from query depth.
    import trimesh
    vertices=[];triangles=[];ids=[];colors=[];offset=0
    for obj in instances:
        config=Path(obj['template']);config=config if config.is_absolute() else src/'configs/objects'/(config.name+'.object_config.json')
        if not config.exists():
            candidates=list((src/'configs/objects').glob('*'+Path(obj['template']).stem+'*.json'));assert len(candidates)==1;(config,)=candidates
        d=json.loads(config.read_text());asset=(config.parent/d['render_asset']).resolve();mesh=trimesh.load(asset,force='scene').to_geometry()
        T=np.asarray(obj['T_world_object']);V=np.asarray(mesh.vertices)@T[:3,:3].T+T[:3,3]
        vertices.append(V);triangles.append(np.asarray(mesh.faces)+offset);ids.append(np.full(len(V),obj['id'],np.int32));offset+=len(V)
        colors.append(asset_colors(mesh))
    # Stage architectural mesh, explicitly unknown semantic ID (zero).
    stage_inst=json.loads((src/'configs/scenes'/(scene+'.scene_instance.json')).read_text())['stage_instance']
    stage_cfg=next(p for p in (src/'configs/stages').glob('*.json') if stage_inst['template_name'].lower() in p.stem.lower() or p.stem.lower() in stage_inst['template_name'].lower())
    d=json.loads(stage_cfg.read_text());mesh=trimesh.load((stage_cfg.parent/d['render_asset']).resolve(),force='scene').to_geometry()
    V=np.asarray(mesh.vertices);vertices.append(V);triangles.append(np.asarray(mesh.faces)+offset);ids.append(np.zeros(len(V),np.int32));colors.append(asset_colors(mesh))
    np.savez_compressed(out/'annotated_asset_mesh.npz',vertices=np.vstack(vertices),triangles=np.vstack(triangles),obj_ids=np.concatenate(ids),vertex_colors=np.vstack(colors))

    atomic(out/'acceptance.json',dict(status='accepted',dataset='ReplicaCAD',scene=scene,simulated=True,frames=frames,references=references,instances=len(instances),query_poses_separate=True,reference_query_exact_rgb_overlap=0,scope='Alternative-data original-method evaluation; shared FRL apartment layouts are not independent physical buildings'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scene');p.add_argument('output');p.add_argument('--mesh-only',action='store_true');a=p.parse_args();main(a.scene,a.output,mesh_only=a.mesh_only)
