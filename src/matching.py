# Legacy V1 engineering interface, preserved for reproducibility.
# Audited P0/P1 entry points: audit.py and localize.py; see docs/AUDIT_V2.md.
"""Official coarse matcher plus confidence extension; metric image-map coordinates."""
from common import *
from utils.localization_utils import localize,get_ray_from_depth,get_ray_from_semantics

def atlas(meta,step=.1):
    """Vectorized equivalent of upstream cast_ray, verified against it separately.
    Preserve upstream door-through behavior and image-y-up bearing convention.
    """
    from PIL import Image
    if not np.isclose(meta['map_m_per_pixel'],.01): raise ValueError('Resample map to 0.01 m/pixel first')
    img=np.asarray(Image.open(meta['map']).convert('RGB'),dtype=np.float32)/255
    free=np.load(meta['free_mask']); h,w=img.shape[:2]; ratio=step/meta['map_m_per_pixel']
    cache=Path(meta['map']).parent/f'atlas_{step:g}.npz'
    import hashlib
    fingerprint=hashlib.sha256(img.tobytes()+free.tobytes()).hexdigest()
    if cache.exists():
        cached=dict(np.load(cache))
        if str(cached.get('fingerprint',''))==fingerprint and cached['valid'].any(): return cached
    gh,gw=int(h//ratio),int(w//ratio)
    gy,gx=np.meshgrid(np.arange(gh),np.arange(gw),indexing='ij')
    valid=free[np.minimum((gy*ratio).astype(int),h-1),np.minimum((gx*ratio).astype(int),w-1)]>0
    if not valid.any(): raise ValueError('Empty candidate mask; refusing a meaningless pose')
    xy=np.stack([gx[valid]*ratio,gy[valid]*ratio],-1)
    dd=np.full((gh,gw,36),15,np.float32); ss=np.full((gh,gw,36),3,np.int64)
    for o in range(36):
        theta=o*np.pi/18; n=len(xy); active=np.ones(n,bool); red=np.full(n,-1); dist=np.full(n,1500); sem=np.full(n,3)
        for t in range(5,1500):
            ids=np.flatnonzero(active)
            if not len(ids): break
            xx=(xy[ids,0]+t*np.cos(theta)).astype(int); yy=(xy[ids,1]-t*np.sin(theta)).astype(int)
            inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
            outside=ids[~inside]; dist[outside]=np.where(red[outside]>=0,red[outside],t); sem[outside]=np.where(red[outside]>=0,2,3); active[outside]=False
            ids=ids[inside]; c=img[yy[inside],xx[inside]]
            black=np.all(c<.1,axis=1); blue=(c[:,0]<.2)&(c[:,1]<.2)&(c[:,2]>.8); r=(c[:,0]>.8)&(c[:,1]<.2)&(c[:,2]<.2)
            red[ids[r]]=t
            stop=black|blue; hit=ids[stop]; dist[hit]=t; sem[hit]=np.where(red[hit]>60,2,np.where(blue[stop],1,0)); active[hit]=False
        last=np.flatnonzero(active&(red>=0)); dist[last]=red[last]; sem[last]=2
        dd[...,o][valid]=dist*.01; ss[...,o][valid]=sem
    np.savez_compressed(cache,depth=dd,semantic=ss,valid=valid,step=step,fingerprint=fingerprint)
    return dict(depth=dd,semantic=ss,valid=valid,step=step,fingerprint=fingerprint)

def match(a,depth40,semantic40,weights40=None,soft40=None,mode='official'):
    dr=get_ray_from_depth(depth40,V=9,F_W=1/np.tan(np.deg2rad(40))/2)
    sr=get_ray_from_semantics(semantic40.tolist())
    dm=torch.from_numpy(a['depth']); sm=torch.from_numpy(a['semantic'])
    if mode=='official':
        pd=localize(dm,torch.tensor(dr),return_np=False)[0]
        ps=localize(sm,torch.tensor(sr),return_np=False,localize_type='semantic')[0]
    else:
        # Same angular interpolation as depth sampling, no depth cosine correction for confidence.
        xp=np.tan(np.deg2rad(np.arange(-40,41,10)))*40*(1/np.tan(np.deg2rad(40))/2)+19.5
        xp=np.clip(xp,0,39)
        weights=np.interp(xp,np.arange(40),weights40)[::-1].copy()
        if weights.sum()<.2: raise ValueError('Insufficient reliable rays: abstain')
        obs=torch.tensor(dr[::-1].copy(),dtype=torch.float32)
        prob=np.stack([np.interp(xp,np.arange(40),soft40[:,c]) for c in range(4)],-1)[::-1].copy()
        prob=torch.tensor(prob,dtype=torch.float32); prob=prob/prob.sum(-1,keepdim=True)
        wd=torch.tensor(weights,dtype=torch.float32)
        dp=torch.nn.functional.pad(dm,[4,5],mode='circular'); sp=torch.nn.functional.pad(sm,[4,5],mode='circular')
        pd=[]; ps=[]
        for o in range(36):
            error=(dp[...,o:o+9]-obs).abs()
            pd.append(torch.exp(-(error*wd).sum(-1)*9/wd.sum()/40))
            if mode=='mask_only':
                target=sp[...,o:o+9].long(); observed=torch.tensor(sr[::-1].copy())
                penalty=(target!=observed).float()*torch.tensor([1.,5.,3.,0.])[target]
                ps.append(torch.exp(-(penalty*wd).sum(-1)*9/wd.sum()/40))
                continue
            q=torch.nn.functional.one_hot(sp[...,o:o+9].long(),4).float(); mid=(q+prob)/2
            js=.5*((prob*(prob.clamp_min(1e-8).log()-mid.clamp_min(1e-8).log())).sum(-1)+(q*(q.clamp_min(1e-8).log()-mid.clamp_min(1e-8).log())).sum(-1))
            ps.append(torch.exp(-(js*wd).sum(-1)*9/wd.sum()/4))
        pd=torch.stack(pd,-1); ps=torch.stack(ps,-1)
    score=(.6*pd+.4*ps).numpy(); score[~a['valid']]=0
    ids=np.argsort(score.ravel())[-5:][::-1]; candidates=[]
    for k in ids:
        y,x,o=np.unravel_index(k,score.shape)
        candidates.append({'pose':[float(x*a['step']),float(y*a['step']),float(o*np.pi/18)],'score':float(score[y,x,o])})
    return candidates,score.max(-1)
