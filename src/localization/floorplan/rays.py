"""Vectorized backend for the pinned upstream cast_ray behavior, not a new sensor model."""
import numpy as np

def cast_many(image,positions,angles,dist_max=1500):
    image=np.asarray(image);positions=np.broadcast_to(np.asarray(positions), (len(angles),2));angles=np.asarray(angles)
    n=len(angles);h,w=image.shape[:2];active=np.ones(n,bool);red=np.full(n,-1);depth=np.full(n,dist_max);label=np.full(n,3);hit=np.zeros((n,2),int);red_hit=hit.copy()
    for t in range(5,dist_max):
        ids=np.flatnonzero(active)
        if not len(ids):break
        xx=(positions[ids,0]+t*np.cos(angles[ids])).astype(int);yy=(positions[ids,1]-t*np.sin(angles[ids])).astype(int)
        hit[ids]=np.stack([xx,yy],-1);inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        outside=ids[~inside];depth[outside]=np.where(red[outside]>=0,red[outside],t);label[outside]=np.where(red[outside]>=0,2,3)
        has_red=outside[red[outside]>=0];hit[has_red]=red_hit[has_red];active[outside]=False
        ids=ids[inside];c=image[yy[inside],xx[inside],:3]
        black=np.all(c<.1,axis=1);blue=(c[:,0]<.2)&(c[:,1]<.2)&(c[:,2]>.8);r=(c[:,0]>.8)&(c[:,1]<.2)&(c[:,2]<.2)
        red[ids[r]]=t;red_hit[ids[r]]=hit[ids[r]]
        stop=black|blue;found=ids[stop];depth[found]=t;label[found]=np.where(red[found]>60,2,np.where(blue[stop],1,0));active[found]=False
    found=np.flatnonzero(active&(red>=0));depth[found]=red[found];label[found]=2;hit[found]=red_hit[found]
    return depth*.01,label,hit
