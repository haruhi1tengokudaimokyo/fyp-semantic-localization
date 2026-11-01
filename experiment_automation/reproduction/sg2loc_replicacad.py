"""Dataset adapter around the actual SG2Loc SceneGraphLoc/PF/refinement classes.

ReplicaCAD alternative-data evaluation, RGB-D + measured RGB-D motion + level
camera mounting. This differs from upstream's GT-relative-motion condition.
"""
import argparse,json,sys,hashlib,time
from pathlib import Path
import numpy as np
from blocked_data import atomic
SRC=Path('/data/fyp_3d/sources/sg2loc');sys.path.insert(0,str(SRC))
A=np.array([[1.,0,0,0],[0,0,-1.,0],[0,1.,0,0],[0,0,0,1.]])


def save_window_checkpoint(path,identity,cursor,poses,coarse):
    state=np.random.get_state();temporary=path.with_suffix('.part.npz')
    np.savez_compressed(temporary,identity=identity,cursor=cursor,poses=poses,
        coarse=json.dumps(coarse),rng_kind=state[0],rng_keys=state[1],
        rng_pos=state[2],rng_has_gauss=state[3],rng_cached_gauss=state[4])
    temporary.replace(path)


def load_window_checkpoint(path,identity,length):
    with np.load(path,allow_pickle=False) as z:
        assert str(z['identity'])==identity,'Checkpoint source/configuration mismatch; retained for diagnosis'
        cursor=int(z['cursor']);assert 0<=cursor<=600 and cursor%length==0
        poses=z['poses'].copy();assert poses.shape==(600,4,4)
        coarse=json.loads(str(z['coarse']))
        np.random.set_state((str(z['rng_kind']),z['rng_keys'].copy(),int(z['rng_pos']),int(z['rng_has_gauss']),float(z['rng_cached_gauss'])))
    return cursor,poses,coarse


def cfg_for(data,out,length):
    from omegaconf import OmegaConf
    cfg=OmegaConf.merge(OmegaConf.load(SRC/'sg2loc/scannet/configs/val.yaml'),OmegaConf.load(SRC/'sg2loc/scannet/configs/sg2loc.yaml'))
    cfg.model.global_descriptor_dim=1024;cfg.train.loss.use_global_descriptor=False
    cfg.data.img.w=480;cfg.data.img.h=270
    cfg.data.img_encoding.patch_w=32;cfg.data.img_encoding.patch_h=18
    cfg.particle_filter.sequence_length=length
    cfg.particle_filter.scans_scenes_dir=str(data)
    cfg.particle_filter.preprocess.output_dir=str(out/'bvh')
    cfg.particle_filter.preprocess.embeddings_dir=str(out/'object_embeddings')
    # Only CUDA pose batches are smaller on 6GiB; particle count/pass/noise unchanged.
    obs=cfg.particle_filter.observation_model
    obs.first_pose_batch_size=128;obs.second_pose_batch_size=64;obs.third_pose_batch_size=128
    obs.threads_per_block=128
    cfg.other.use_resume=False
    return cfg


def prepare(data,out):
    from sg2loc.preprocessing.bvh_build import build_bvh,flatten_bvh
    mesh=np.load(data/'annotated_asset_mesh.npz');vertices=mesh['vertices']@A[:3,:3].T
    triangles=mesh['triangles'];nodes=[];tris=[];orig=[]
    root=build_bvh(triangles,vertices,8);flatten_bvh(root,nodes,tris,orig)
    bvh=out/'bvh/map';bvh.mkdir(parents=True,exist_ok=True)
    for name,value in dict(vertices=vertices,bvh_nodes=np.asarray(nodes),bvh_triangles=np.asarray(tris),obj_ids=mesh['obj_ids'],uv_coords=np.array([[0,0],[1,0],[0,1.]],np.float32),uv_indices=np.tile([0,1,2],(len(triangles),1)),triangle_orig_indices=orig,vertex_colors=mesh['vertex_colors']).items():np.save(bvh/(name+'.npy'),value)
    # Only nodes with actual multiview RGB observations enter the encoder. No zero/random feature placeholders.
    counts={}
    for p in sorted((data/'reference').glob('*.npz')):
        labels=np.load(p)['instance']
        for i,n in zip(*np.unique(labels,return_counts=True)):
            if i>1 and n>=16:counts.setdefault(int(i),[]).append((int(n),p.stem))
    object_ids=sorted(counts);assert len(object_ids)>=2
    centers=[];object_points=[];rng=np.random.default_rng(2089)
    for obj in object_ids:
        pts=vertices[mesh['obj_ids']==obj];assert len(pts)>0
        centers.append(pts.mean(0));object_points.append(pts[rng.choice(len(pts),512,replace=len(pts)<512)])
    center=vertices.mean(0);centers=np.asarray(centers)
    # Real spatial adjacency; explicit 'none' relation where there is no supported semantic relation annotation.
    edges=np.array([(i,j) for i in range(len(centers)) for j in range(len(centers)) if i!=j and np.linalg.norm(centers[i]-centers[j])<3],np.int64)
    assert len(edges)>0
    rel=np.zeros((len(object_ids),9),np.float32)
    names=(SRC/'sg2loc/scannet/preprocessing/labels/scannet8_relationships.txt').read_text().splitlines()
    none=names.index('none')
    for i,j in edges:rel[i,none]+=1
    np.savez_compressed(out/'scene_graph.npz',obj_ids=object_ids,points=np.asarray(object_points)-center,centers=centers-center,edges=edges,relations=rel)
    atomic(out/'reference_votes.json',{str(k):[n for _,n in sorted(v,reverse=True)[:10]] for k,v in counts.items()})
    atomic(out/'geometry.json',dict(status='accepted',nodes=len(object_ids),edges=len(edges),actual_asset_triangles=len(triangles),world_conversion=A.tolist(),relation_policy='Measured metric-neighborhood adjacency; unavailable semantic relations explicitly none in official ScanNet vocabulary, no invented annotations',attribute_policy='Original ScanNet checkpoint omits attr module',dropped_unobserved_objects=True))


def features(data,out):
    import torch
    from PIL import Image
    from torchvision import transforms as tvf
    from sg2loc.preprocessing.dinov2_utils import DinoV2ExtractFeatures
    from sg2loc.scan3r.preprocessing.generate_object_visual_embeddings import mask2box_multi_level
    torch.set_num_threads(4);torch.hub.set_dir('/data/fyp_3d/models/torch_hub')
    original_hub_load=torch.hub.load
    def local_dino(repo,name,*args,**kwargs):
        if repo=='facebookresearch/dinov2':
            return original_hub_load('/data/fyp_3d/sources/dinov2',name,*args,source='local',**kwargs)
        return original_hub_load(repo,name,*args,**kwargs)
    torch.hub.load=local_dino
    extractor=DinoV2ExtractFeatures('dinov2_vitg14',31,'value',use_cls=True,device='cuda')
    transform=tvf.Compose([tvf.ToTensor(),tvf.Normalize([.485,.456,.406],[.229,.224,.225])])
    votes=json.loads((out/'reference_votes.json').read_text());object_features={}
    for obj,frames in votes.items():
        values=[]
        for frame in frames:
            mask=np.load(data/'reference'/(frame+'.npz'))['instance']==int(obj)
            image=Image.open(data/'reference'/(frame+'.png')).convert('RGB');crop_feats=[]
            for level in range(3):
                box=mask2box_multi_level(torch.from_numpy(mask).float(),level,.2)
                rgb=image.crop(box).resize((224,224),Image.BICUBIC)
                with torch.inference_mode():crop_feats.append(extractor(transform(rgb)[None].cuda())[0,0].cpu())
            values.append(torch.stack(crop_feats).mean(0))
        object_features[int(obj)]=torch.stack(values);assert torch.isfinite(object_features[int(obj)]).all()
        print('OBJECT_DINO',obj,len(values),flush=True)
    torch.save(object_features,out/'object_features.pt')
    q=out/'patch_features';q.mkdir(exist_ok=True)
    for i,p in enumerate(sorted((data/'query').glob('*.png'))):
        target=q/(p.stem+'.npy')
        if target.exists():continue
        image=Image.open(p).convert('RGB').resize((32*14,18*14),Image.BICUBIC)
        with torch.inference_mode():value=extractor(transform(image)[None].cuda())[0,1:].reshape(18,32,1536).cpu().numpy()
        assert np.isfinite(value).all()
        tmp=target.with_name(target.stem+'.partial.npy');np.save(tmp,value);tmp.replace(target)
        if i%20==0:print('PATCH_DINO',i,flush=True)
    import subprocess
    receipt=json.loads(Path('/data/fyp_3d/models/torch_hub/checkpoints/dinov2_vitg14_pretrain.pth.json').read_text())
    atomic(out/'features.json',dict(status='accepted',frames=600,DINO='dinov2_vitg14',layer=31,facet='value',actual_multilevel_object_crops=True,zero_or_random_features=False,weight_sha256=receipt['sha256'],DINO_source_commit=subprocess.check_output(['git','-C','/data/fyp_3d/sources/dinov2','rev-parse','HEAD'],text=True).strip()))


def graph_tensors(out):
    import torch
    g=np.load(out/'scene_graph.npz');n=len(g['obj_ids']);e=len(g['edges'])
    return dict(batch_size=1,obj_ids=g['obj_ids'],tot_obj_count=np.array([n]),graph_per_obj_count=np.array([[n]]),graph_per_edge_count=np.array([[e]]),tot_obj_pts=torch.from_numpy(g['points']).float().cuda(),tot_rel_pose=torch.from_numpy(g['centers']).float().cuda(),edges=torch.from_numpy(g['edges']).cuda(),tot_bow_vec_object_edge_feats=torch.from_numpy(g['relations']).float().cuda(),scene_ids=[['map']],obj_img_patches={'map':{k:v.cuda() for k,v in torch.load(out/'object_features.pt',weights_only=True).items()}})


def locate(data,out,length):
    import torch
    from sg2loc.particle_filter.runner import ParticleFilterRunner
    from sg2loc.particle_filter.filter import ParticleFilter
    from sg2loc.particle_filter.scene import load_scene_geometry
    from sg2loc.particle_filter.observation import ObservationModel
    cfg=cfg_for(data,out,length);torch.set_num_threads(4);np.random.seed(2089);torch.manual_seed(2089)
    runner=ParticleFilterRunner.__new__(ParticleFilterRunner);runner.cfg=cfg;runner.device=torch.device('cuda');runner.build_model(cfg)
    checkpoint=json.loads(Path('/data/fyp_3d/models/sg2loc_scannet/checkpoint.json').read_text())
    state=torch.load(checkpoint['checkpoint'],map_location='cpu',weights_only=False)['model']
    for layer in [0,1]:
        stem=f'sg_encoder.structure_encoder.layer_stack.{layer}.'
        # PyG renamed homogeneous GAT's shared lin_src/lin_dst to lin. Only
        # collapse them after proving byte-identical learned weights.
        if stem+'lin.weight' not in state:
            assert torch.equal(state[stem+'lin_src.weight'],state[stem+'lin_dst.weight'])
            state[stem+'lin.weight']=state[stem+'lin_src.weight']
    model_state=runner.model.state_dict();missing=set(dict(runner.model.named_parameters()))-set(state)
    assert not missing, f'Missing learned SceneGraphLoc parameters: {missing}'
    assert all(tuple(v.shape)==tuple(model_state[k].shape) for k,v in state.items() if k in model_state)
    runner.model.load_state_dict({k:v for k,v in state.items() if k in model_state},strict=False);runner.model.eval()
    graph=graph_tensors(out)
    with torch.inference_mode():emb=torch.nn.functional.normalize(runner.model.obj_embedding_encoder(runner.model.forward_scene_graph({'scene_graphs':graph})),dim=-1).cpu()
    scene=load_scene_geometry(str(out/'bvh/map'));truth=np.load(out/'measured_motion.npz')['T_local_camera'];K=np.load(data/'K.npy');K[:2,2]+=.5
    # Correct row-major patch orientation for unrotated landscape images; upstream
    # helper unconditionally uses the portrait 3RScan flip, even in its ScanNet path.
    def labels(self,i):
        v,idx=torch.topk(self.pf.patch_obj_sim_T[i]+1,k=1,dim=1)
        obj=self.pf.obj_ids_slices[self.pf.scene_graph_to_index['map']]
        return v.numpy().flatten().astype(np.float32),obj[idx.numpy().flatten()]
    ObservationModel.semantic_patch_labels=labels
    class CADFilter(ParticleFilter):
        gravity_axis_col=1
        def load_query_poses(self,scan_id,frame_idxs):return truth[np.asarray(frame_idxs,dtype=int)]
        def map_to_query_transform(self,scan_id):return np.eye(4)
        def load_intrinsics(self,scan_id):return K
        def query_image_path(self,scan_id,frame_idx):return str(data/'query'/f'{int(frame_idx):06d}.png')
        def map_texture_path(self,scan_id):return None
        def floor_mask(self,vertices,obj_ids):
            z=np.percentile((np.load(data/'navmesh_vertices.npy')@A[:3,:3].T)[:,2],50)
            return np.abs(vertices[:,2]-z)<.15
        def load_query_sensor_depth(self,scan_id,frame_idx):
            y,x=np.mgrid[:270,:480]
            raynorm=np.sqrt(1+((x+.5-K[0,2])/K[0,0])**2+((y+.5-K[1,2])/K[1,1])**2)
            return np.load(data/'query'/f'{int(frame_idx):06d}.npz')['depth']*raynorm
    poses=np.full((600,4,4),np.nan);coarse=[]
    run=out/f'length{length}';run.mkdir(exist_ok=True)
    # The released PF uses NumPy randomness. Retain its exact RNG state at each
    # complete sequence window, bound to inputs, learned weights and adapter code.
    from omegaconf import OmegaConf
    digest=hashlib.sha256(OmegaConf.to_yaml(cfg).encode())
    for source in [Path(__file__),data/'acceptance.json',out/'features.json',out/'motion.json',Path('/data/fyp_3d/models/sg2loc_scannet/checkpoint.json')]:digest.update(source.read_bytes())
    identity=digest.hexdigest();checkpoint_path=run/'window_checkpoint.npz';cursor=0
    if checkpoint_path.exists():
        cursor,poses,coarse=load_window_checkpoint(checkpoint_path,identity,length)
        print('RESUME_COMPLETE_WINDOWS',cursor,flush=True)
    for st in range(cursor,600,length):
        window_start=time.monotonic()
        frames=list(range(st,min(st+length,600)));assert len(frames)==length
        raw=torch.from_numpy(np.stack([np.load(out/'patch_features'/f'{i:06d}.npy') for i in frames])).cuda()
        with torch.inference_mode():
            patch=runner.encode_patches({'patch_features':raw});sims=[torch.nn.functional.normalize(x,dim=-1).cpu()@emb.T for x in patch]
        d=dict(scan_ids=['query']*length,scan_ids_temp=['map']*length,frame_idxs=frames,scene_graphs=graph)
        pf=CADFilter(cfg,d,sims,scene);pose,_,_=pf.run_particle_filter()
        for frame,p in getattr(pf,'frame_mle_poses',{}).items():poses[int(frame)]=np.linalg.inv(A)@p
        if pose is not None:poses[frames[-1]]=np.linalg.inv(A)@pose;coarse.append({'last_frame':frames[-1],'pose_zup':pose.tolist()})
        np.savez_compressed(run/'trajectory.partial.npz',T_world_camera=poses,processed_frames=st+length)
        save_window_checkpoint(checkpoint_path,identity,st+length,poses,coarse)
        print('SG2LOC',length,st,'valid',np.isfinite(poses).all((1,2)).sum(),'window_seconds',time.monotonic()-window_start,flush=True)
    np.savez_compressed(run/'trajectory.npz',T_world_camera=poses)
    atomic(run/'coarse.json',coarse)
    atomic(run/'execution.json',dict(status='executed',localization_executed=True,frames=600,sequence_length=length,GT_online=False,GT_scope='Query GT isolated; measured RGB-D odometry and declared level mounting replace upstream GT-relative-motion condition',refinement_executed=False,SceneGraphLoc_checkpoint=checkpoint,learned_parameters_missing=[],original_SemanticParticleFilter=True,original_ObservationModel=True,patch_orientation_adapter='Landscape row-major, no portrait flip; +0.5 principal-point shift for upstream pixel-center rays',dataset=json.loads((data/'acceptance.json').read_text())['dataset']+' alternative-data evaluation, not original ScanNet benchmark'))


def refine(data,out,length):
    import torch,pandas as pd
    import dataclasses
    from scipy.spatial.transform import Rotation
    from romav2 import RoMaV2
    from sg2loc.scannet.refinement import ScannetSequenceRefiner
    from sg2loc.particle_filter.scene import load_scene_geometry
    cfg=cfg_for(data,out,length);torch.set_num_threads(4);torch.hub.set_dir('/data/fyp_3d/models/torch_hub')
    truth=np.load(out/'measured_motion.npz')['T_local_camera'];K=np.load(data/'K.npy');K[:2,2]+=.5
    run=out/f'length{length}';dest=run/'refinement';dest.mkdir(exist_ok=True)
    coarse=json.loads((run/'coarse.json').read_text());lookup={}
    for row in coarse:
        p=np.array(row['pose_zup']);i=row['last_frame'];values={f'MLEPose_{j}':v for j,v in enumerate(p.flatten())}
        values['PositionError']=np.nan;values['RotationError']=np.nan
        lookup[('query',i)]=pd.Series(values)
    scene=load_scene_geometry(str(out/'bvh/map'))
    class CADRefiner(ScannetSequenceRefiner):
        def query_image_path(self,scan_id,frame_idx):return str(data/'query'/f'{int(frame_idx):06d}.png')
        def load_query_poses(self,scan_id,frame_idxs):return truth[np.asarray(frame_idxs,dtype=int)]
        def map_to_query_transform(self,scan_id):return np.eye(4)
        def load_intrinsics(self,scan_id):return K.copy()
    appearance=CADRefiner.load_appearance(cfg,'map')
    # Turing has no BF16 support. Keep learned parameters FP32 and use FP16
    # autocast for upstream AMP blocks; architecture/weights/thresholds unchanged.
    original_autocast=torch.autocast
    class TuringAutocast(original_autocast):
        def __init__(self,device_type,dtype=None,*args,**kwargs):
            if dtype is torch.bfloat16:dtype=torch.float16
            super().__init__(device_type,dtype,*args,**kwargs)
    torch.autocast=TuringAutocast
    roma_cfg=RoMaV2.Cfg(setting=cfg.refinement.roma_setting)
    roma_cfg=dataclasses.replace(roma_cfg,compile=False,descriptor=dataclasses.replace(roma_cfg.descriptor,enable_amp=False))
    base_roma=RoMaV2(roma_cfg)
    class PairBatchOne:
        # Each view pair is independent. Serialize identical original matcher
        # calls to fit 6GiB without changing RoMa/PnP acceptance thresholds.
        def __getattr__(self,name):return getattr(base_roma,name)
        def match(self,q,r):
            predictions=[base_roma.match(q[i:i+1],r[i:i+1]) for i in range(len(q))]
            return {k:torch.cat([p[k] for p in predictions],dim=0) if torch.is_tensor(predictions[0][k]) else predictions[0][k] for k in predictions[0]}
    roma=PairBatchOne()
    framesfile=dest/'frame_poses.txt';framesfile.write_text('')
    for st in range(0,600,length):
        frames=list(range(st,st+length));d=dict(scan_ids=['query']*length,scan_ids_temp=['map']*length,frame_idxs=frames)
        refiner=CADRefiner(d,cfg,lookup,cfg.refinement.pnp_thresh,scene,appearance);refiner.out_dir=str(dest)
        refiner.localize_seq(st//length,roma);print('SG2LOC_REFINEMENT',st,flush=True)
    # The official refinement file contains WORLD-TO-CAMERA PnP poses; invert
    # before exporting. Failed/low-inlier frames fall back to their actual PF pose.
    coarseposes=np.load(run/'trajectory.npz')['T_world_camera'];poses=coarseposes.copy();inliers=np.zeros(600,int)
    for line in framesfile.read_text().splitlines():
        v=line.split();i=int(v[1]);inliers[i]=int(v[9]);p=np.eye(4);p[:3,:3]=Rotation.from_quat(np.array(v[2:6],float)).as_matrix();p[:3,3]=np.array(v[6:9],float)
        if inliers[i]>cfg.refinement.pnp_thresh:poses[i]=np.linalg.inv(A)@np.linalg.inv(p)
    np.savez_compressed(dest/'trajectory.npz',T_world_camera=poses,inliers=inliers)
    atomic(dest/'execution.json',dict(status='executed',localization_executed=True,refinement_executed=True,frames=600,original_RoMa_refinement=True,matcher_pair_batch=1,refined_frames=int((inliers>cfg.refinement.pnp_thresh).sum()),GT_online=False,GT_scope='Measured RGB-D relative-motion backward propagation; query GT first read by separate offline metrics task'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['geometry','features','localize','refine','refine_accept']);p.add_argument('data');p.add_argument('output');p.add_argument('--length',type=int,choices=[5,10,25],default=5);a=p.parse_args();d=Path(a.data);o=Path(a.output);o.mkdir(parents=True,exist_ok=True)
    if a.phase=='geometry':prepare(d,o)
    elif a.phase=='features':features(d,o)
    elif a.phase=='localize':locate(d,o,a.length)
    else:
        refine(d,o,a.length)
        if a.phase=='refine_accept':
            from localization_metrics import evaluate
            evaluate(d,o/f'length{a.length}/refinement')
