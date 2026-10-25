"""RGB -> official depth/semantic rays -> official map matching -> metric pose."""
from common import *
from matching import atlas,match
import argparse,time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def run(sample,out,step=.1,occlusion=0.,mode='official',person_mask=False):
    torch.manual_seed(42); np.random.seed(42)
    meta,image=read_sample(sample); device='cuda' if torch.cuda.is_available() else 'cpu'
    depth,semantic=models(); depth.to(device).eval(); semantic.to(device).eval()
    a=atlas(meta,step)
    image=image.to(device); mask=torch.ones((1,360,640),device=device)
    if occlusion:
        width=int(640*occlusion); start=(640-width)//2
        image[:,:,:,start:start+width]=.25
    if device=='cuda': torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start=time.perf_counter()
    with torch.inference_mode():
        d,_,dp=depth(image,mask); s,room,_=semantic(image,mask)
    if device=='cuda': torch.cuda.synchronize()
    network_ms=1000*(time.perf_counter()-start)
    d=d[0].cpu().numpy(); sp=s.softmax(-1)[0].cpu().numpy(); hard=sp.argmax(-1)
    conf=np.ones(40)
    if occlusion:
        width=int(40*occlusion); lo=(40-width)//2; conf[lo:lo+width]=0
    if person_mask:
        from person_mask import person_ray_confidence
        detected_conf,person_pixels=person_ray_confidence(image,device)
        conf*=detected_conf
        mode='weighted_jsd'
    start=time.perf_counter()
    candidates,heat=match(a,d,hard,conf,sp,mode)
    matching_ms=1000*(time.perf_counter()-start)
    gt=np.array(meta['pose']) if meta.get('pose') is not None else None; pred=np.array(candidates[0]['pose'])
    err=float(np.linalg.norm(pred[:2]-gt[:2])) if gt is not None else None; angle=float(abs((pred[2]-gt[2]+np.pi)%(2*np.pi)-np.pi)*180/np.pi) if gt is not None else None
    result={'sample':str(sample),'building_id':meta.get('building_id','provided_map'),'scope':meta.get('usage','external_query'),'mode':mode,'grid_m':step,'occlusion_fraction':occlusion,'occlusion_mask_source':'pretrained_lraspp_person' if person_mask else ('known_rectangle_oracle_mask' if occlusion else 'none'),'pose':pred.tolist(),'ground_truth':gt.tolist() if gt is not None else None,'position_error_m':err,'angle_error_deg':angle,'network_ms':network_ms,'matching_ms':matching_ms,'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,'gpu_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20 if device=='cuda' else None,'candidates':candidates,'depth_axial_m':d.tolist(),'semantic_probabilities':sp.tolist(),'confidence':conf.tolist(),'room_type_prediction':int(room.argmax(-1).item())}
    out=Path(out); out.mkdir(parents=True,exist_ok=True); write_json(out/'result.json',result)
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    ax[0,0].imshow(image[0].cpu().permute(1,2,0)); ax[0,0].set_title(f'Official RGB input | occlusion {occlusion:.0%}'); ax[0,0].axis('off')
    ax[0,1].imshow(Image.open(meta['map'])); res=meta['map_m_per_pixel']
    for p,color,label in [(gt,'lime','GT'),(pred,'magenta','estimate')]:
        if p is None: continue
        ax[0,1].scatter(p[0]/res,p[1]/res,c=color,label=label)
        ax[0,1].arrow(p[0]/res,p[1]/res,50*np.cos(p[2]),-50*np.sin(p[2]),color=color,width=3)
    ray_angles=np.flip(np.arctan2(np.arange(40)-19.5,40*(1/np.tan(np.deg2rad(40))/2)))
    for j in range(0,40,4):
        distance=d[j]/np.cos(ray_angles[j]); direction=pred[2]+ray_angles[j]
        end=pred[:2]+[distance*np.cos(direction),-distance*np.sin(direction)]
        ax[0,1].plot([pred[0]/res,end[0]/res],[pred[1]/res,end[1]/res],color=['gray','blue','red','pink'][hard[j]],alpha=.7,lw=.8)
    map_w,map_h=Image.open(meta['map']).size; ax[0,1].set_xlim(0,map_w); ax[0,1].set_ylim(map_h,0)
    ax[0,1].legend(); ax[0,1].set_title(f'{err:.2f} m / {angle:.1f} deg' if gt is not None else 'Estimated pose on provided map'); ax[0,1].axis('off')
    ax[1,0].plot(d,label='RGB prediction'); ax[1,0].plot(meta.get('depth_axial_m',[]),label='map GT axial (if available)'); ax[1,0].set_ylabel('depth (m)'); ax[1,0].set_xlabel('image column ray'); ax[1,0].legend()
    ax[1,1].imshow(heat,cmap='magma'); ax[1,1].set_title(f'Position score, {step:g} m grid / {mode}'); ax[1,1].axis('off')
    fig.savefig(out/'localization.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,3),layout='constrained'); ax.imshow(sp.T,aspect='auto',vmin=0,vmax=1,cmap='viridis'); ax.set_yticks(range(4),['wall','window','door','unknown']); ax.set_xlabel('ray index'); ax.set_title('Official semantic head probabilities'); fig.savefig(out/'semantics.png',dpi=150); plt.close(fig)
    print(json.dumps({k:result[k] for k in ['position_error_m','angle_error_deg','network_ms','matching_ms','gpu_peak_allocated_mib']}))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--sample',required=True); p.add_argument('--out',required=True); p.add_argument('--grid',type=float,default=.1); p.add_argument('--occlusion',type=float,default=0); p.add_argument('--person-mask',action='store_true'); p.add_argument('--mode',choices=['official','weighted_jsd'],default='official'); a=p.parse_args(); run(a.sample,a.out,a.grid,a.occlusion,a.mode,a.person_mask)
