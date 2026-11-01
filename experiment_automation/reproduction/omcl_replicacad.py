"""Official OMCL kernels and original LSeg, applied to disjoint ReplicaCAD renders.

The adapter replaces filesystem/visualization plumbing, not CLIP retrieval or PnP.
Motion is explicitly noisy GT-relative motion, as in the released OMCL driver.
No GT-free claim is made. Global initialization uses map reference poses only.
"""
import argparse,json,sys,time,math
from pathlib import Path
import numpy as np
from blocked_data import atomic

SOURCE=Path('/data/fyp_3d/sources/omcl')
sys.path[:0]=[str(SOURCE/'src'),'/data/fyp_3d/sources/CLIP']


def extract(data,out):
    import torch,clip,timm
    from PIL import Image
    from torchvision import transforms
    from omcl.models.image_encoders.lseg.modules.models.lseg_net import LSegEncNet
    from omcl.models.image_encoders.lseg.lseg import get_lseg_feat
    from omcl.models.processing import get_unique_features
    torch.set_num_threads(4);torch.manual_seed(2089)
    old=timm.create_model
    def create(*a,**kw):kw['pretrained']=False;return old(*a,**kw)
    timm.create_model=create
    oldclip=clip.load
    def load(*a,**kw):kw.update(download_root='/data/fyp_3d/models/clip',device='cpu');return oldclip(*a,**kw)
    clip.load=load
    model=LSegEncNet('',arch_option=0,block_depth=0,activation='lrelu',crop_size=480)
    weight=next(Path('/data/fyp_3d/models/lseg').glob('*.ckpt'))
    state=torch.load(weight,map_location='cpu',weights_only=False)['state_dict'];model.load_state_dict({k.removeprefix('net.'):v for k,v in state.items()},strict=True);del state
    model=model.cuda().eval();clip.model.convert_weights(model.clip_pretrained)
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])
    bank=torch.empty(0,512,device='cuda',dtype=torch.float16)
    for split in ['reference','query']:
        folder=out/split;folder.mkdir(parents=True,exist_ok=True)
        # Reference dictionary is frozen. Queries carry their own real feature dictionaries.
        if split=='reference' and (out/'reference_bank.pt').exists():
            bank=torch.load(out/'reference_bank.pt',weights_only=True).cuda()
        for i,p in enumerate(sorted((data/split).glob('*.png'))):
            target=folder/(p.stem+'.pt')
            if target.exists():continue
            rgb=np.asarray(Image.open(p).convert('RGB'))
            with torch.inference_mode():
                feat,_=get_lseg_feat(model,rgb,[''],tf,'cuda',crop_size=480,base_size=480)
                feat=feat[0].permute(1,2,0).reshape(-1,512).half()
                assert torch.isfinite(feat).all()
                known=bank if split=='reference' else torch.empty(0,512,device='cuda',dtype=torch.float16)
                new=get_unique_features(known,feat,similarity_threshold=.1)
                if len(new):known=torch.cat([known,new.half()])
                ids=[]
                for chunk in feat.split(4096):ids.append((chunk@known.T).argmax(-1).cpu())
                ids=torch.cat(ids).reshape(rgb.shape[:2])
                if split=='reference':
                    bank=known;tmp=out/'reference_bank.part';torch.save(bank.cpu(),tmp);tmp.replace(out/'reference_bank.pt')
                tmp=target.with_suffix('.part');torch.save(dict(ids=ids,features=known.cpu() if split=='query' else None),tmp);tmp.replace(target)
            print('LSEG',split,i,len(known),flush=True)
    acceptance=json.loads((data/'acceptance.json').read_text())
    atomic(out/'features.json',dict(status='accepted',frames=acceptance['frames'],references=acceptance['references'],original_LSeg_strict_weights=True,base_size=480,crop_size=480,batch=1,dictionary_threshold=.1,query_GT_read=False))


def mapping(data,out):
    import torch
    from omcl.models.map import init_map,increment_map
    from omegaconf import OmegaConf
    torch.set_num_threads(4)
    bank=torch.load(out/'reference_bank.pt',weights_only=True).cuda();K=np.load(data/'K.npy');poses=np.load(data/'reference_poses.npy')
    points,features,counts=init_map(None,'cuda',512)
    for i,p in enumerate(sorted((data/'reference').glob('*.npz'))):
        depth=np.load(p)['depth'];obs=torch.load(out/'reference'/(p.stem+'.pt'),weights_only=True)
        y,x=np.mgrid[0:270:8,0:480:8];d=depth[::8,::8];ok=np.isfinite(d)&(d>.1)&(d<15)
        xyz=np.stack([(x[ok]-K[0,2])*d[ok]/K[0,0],(y[ok]-K[1,2])*d[ok]/K[1,1],d[ok]],axis=1)
        world=xyz@poses[i,:3,:3].T+poses[i,:3,3]
        assert np.max(np.abs(world))<25.6, 'Do not silently clip a scene outside the declared octree cube'
        labels=obs['ids'][::8,::8][ok];newfeatures=bank[labels.cuda()]
        points,features,counts=increment_map(points,features,torch.tensor(world,dtype=torch.float32,device='cuda'),newfeatures,counts,max_level=10,resolution=.05,device='cuda')
        print('OCTREE_MAP',i,len(points),flush=True)
    # Preserve the original averaged language vectors (no semantic label substitution).
    torch.save(dict(points=points.cpu(),features=features.cpu(),counts=counts.cpu(),max_level=10,resolution=.05),out/'octree_language_map.pt')
    atomic(out/'map.json',dict(status='accepted',nodes=len(points),references=len(poses),original_increment_map=True,query_used=False,mapping_pixel_stride=8,resolution=.05,max_level=10))


def locate(data,out,mode):
    import torch,kaolin as kal
    from omegaconf import OmegaConf
    from omcl.models import pf
    from omcl.models.map import points_features2octree
    from omcl.utils.rays import create_rays,transform_rays_batched,ray_trace_pose_batched
    from omcl.models.odom import get_mean_pose
    from omcl.models.processing import get_unique_features
    torch.set_num_threads(4);torch.manual_seed(2089);np.random.seed(2089)
    run=out/mode;run.mkdir(exist_ok=True)
    m=torch.load(out/'octree_language_map.pt',weights_only=True)
    # A dictionary of actual averaged map features, compressed by original LiLMaps rule.
    real=m['features'].cuda().half();bank=get_unique_features(torch.empty(0,512,device='cuda',dtype=torch.float16),real,similarity_threshold=.05,mean=True)
    labels=torch.cat([(c@bank.T).argmax(-1).cpu() for c in real.split(4096)])
    spc,spc_labels,scale=points_features2octree(m['points'].cuda(),labels,10,.05,vectorize=True)
    hierarchy=kal.ops.spc.generate_points(spc.octrees,spc.pyramids,spc.exsum)
    # Actual OpenGL camera rays; stored CV poses converted on the right only.
    rays_o,rays_d,_=create_rays(torch.eye(4,device='cuda'),270,480,240.,240.)
    truth=np.load(data/'ground_truth_conditioned_motion_and_eval.npz')['T_world_camera']
    cv_gl=np.diag([1,-1,-1,1]);poses=truth@cv_gl
    cfg=OmegaConf.load(SOURCE/'src/omcl/configs/mp3d_config.yaml')
    cfg.num_particles=1024;cfg.num_samples=1024;cfg.particles_batch_size=32
    if mode=='zero_motion_noise':
        for k in cfg.odom:cfg.odom[k]=0.
    if mode=='global':
        refs=np.load(data/'reference_poses.npy')@cv_gl
        particles=torch.tensor(refs[np.random.choice(len(refs),1024,replace=True)],dtype=torch.float32,device='cuda')
        angle=np.random.uniform(-np.pi,np.pi,1024)
        from scipy.spatial.transform import Rotation
        rr=torch.tensor(Rotation.from_euler('y',angle).as_matrix(),dtype=torch.float32,device='cuda');particles[:,:3,:3]=particles[:,:3,:3]@rr
        weights=torch.ones(1024,dtype=torch.float64,device='cuda')/1024
    else:particles,weights=pf.init_particles(torch.tensor(poses[0],dtype=torch.float32,device='cuda'),[cfg.init_std_x,cfg.init_std_y,cfg.init_std_th],1024)
    estimates=np.full((len(poses),4,4),np.nan);failures=[];start=time.monotonic()
    files=sorted((out/'query').glob('*.pt'));assert len(files)==len(poses)
    # Original axis convention retained. The released estimate_loss drops its last
    # ray in a chunk; this run preserves that code and records it rather than silently tuning.
    for i,p in enumerate(files):
        obs=torch.load(p,weights_only=True);features=obs['features'].cuda();image=features[obs['ids'].cuda()]
        image=image.reshape(270,480,512)
        # RGB-only query condition: all image pixels are eligible; no query depth mask.
        _,mask,success=pf.get_masks_features(torch.ones(270,480,dtype=torch.bool),image,features,torch.arange(len(features)),'cuda',cfg,num_samples=cfg.num_samples)
        if i:
            odom=torch.tensor(np.linalg.inv(poses[i-1])@poses[i],dtype=torch.float32,device='cuda');particles=pf.predict(particles,odom,cfg)
        losses=torch.zeros(1024,device='cuda')
        if success:
            measured=image.reshape(-1,512)[mask]
            for st in range(0,1024,32):
                end=min(st+32,1024);o,d=transform_rays_batched(rays_o[mask],rays_d[mask],particles[st:end])
                idx,_,hit,valid=ray_trace_pose_batched(o/scale,d,spc,hierarchy,spc_labels)
                if len(hit):pf.estimate_loss(valid,hit,measured,idx,bank,bs=16,output_tensor=losses,output_slice=slice(st,end))
        if not torch.isfinite(losses).all() or losses.max()<=0:
            failures.append(dict(frame=i,reason='No finite language-map ray support'));print('NO_SUPPORT',i,flush=True)
            weights=torch.ones_like(weights)/len(weights)
        else:
            weights=pf.estimate_weights(weights,losses)
            mean,rot=get_mean_pose(particles,weights)
            estimated=torch.eye(4);estimated[:3,:3]=rot.cpu();estimated[:3,3]=mean.cpu();estimates[i]=estimated.numpy()@cv_gl
            samples=pf.torch_stratified_resample(weights.cpu());particles=particles[samples.to('cuda')];weights=weights[samples.to('cuda')]
        # Real progress checkpoint; all failures remain NaN, no interpolation.
        if i%20==0 or i==len(files)-1:
            np.savez_compressed(run/'trajectory.partial.npz',T_world_camera=estimates,processed_frames=i+1);print('OMCL',mode,i,'failures',len(failures),flush=True)
    np.savez_compressed(run/'trajectory.npz',T_world_camera=estimates)
    atomic(run/'execution.json',dict(status='executed',localization_executed=True,frames=len(files),mode=mode,failures=failures,wall_seconds=time.monotonic()-start,original_functions=['increment_map','points_features2octree','get_masks_features','ray_trace_pose_batched','estimate_loss','estimate_weights','predict','torch_stratified_resample','get_mean_pose'],GT_online=True,GT_scope='Noisy ground-truth relative motion; tracking initialization near GT except global uses map-only reference sampling',alternative_dataset=json.loads((data/'acceptance.json').read_text())['dataset'],not_original_paper_dataset=True,LSeg_used=True,released_loss_chunk_last_ray_behavior_preserved=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['features','map','tracking','global','global_accept','zero_motion_noise']);p.add_argument('data');p.add_argument('output');a=p.parse_args();data=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.phase=='features':extract(data,out)
    elif a.phase=='map':mapping(data,out)
    elif a.phase=='global_accept':
        locate(data,out,'global')
        from localization_metrics import evaluate
        evaluate(data,out/'global')
    else:locate(data,out,a.phase)
