"""Explicit profiles: upstream coarse-to-fine versus coordinate-consistent integration."""
from pathlib import Path
import numpy as np
import torch
from common import REPO
from utils.localization_utils import localize,get_ray_from_depth,get_ray_from_semantics
from evaluation.candidate_extractor import extract_top_k_locations
from evaluation.candidate_refiner import compute_rays_from_candidate,measure_similarity
import evaluation.candidate_refiner as upstream_refiner
from unittest.mock import patch
from localization.geometry import F_W,COLUMN_ANGLES,axial_to_radial,semantic_rays_nearest,wrap_angle
from localization.floorplan.rays import cast_many

class Matcher:
    def __init__(self,atlas,image,profile,room_masks=None):
        self.atlas=atlas;self.image=image;self.profile=profile;self.cache={};self.room_masks=room_masks or {}
    def volumes(self,depth,labels,valid_weights=None,room_probability=None):
        if valid_weights is not None and np.mean(valid_weights)<.01:raise ValueError('No reliable rays')
        if valid_weights is not None and np.any(np.asarray(valid_weights)<.999):
            raise ValueError('Partial ray weighting requires explicit experimental profile; not silently applied to baseline')
        dr=get_ray_from_depth(depth,V=9,F_W=F_W)
        sr=get_ray_from_semantics(labels.tolist()) if self.profile['semantic_resampling']=='upstream_vote' else semantic_rays_nearest(labels)
        pd=localize(torch.tensor(self.atlas['depth']),torch.tensor(dr),return_np=False)[0].numpy()
        ps=localize(torch.tensor(self.atlas['semantic']),torch.tensor(sr),return_np=False,localize_type='semantic')[0].numpy()
        valid=self.atlas['valid'].copy()
        if self.profile.get('room_aware') and room_probability is not None:
            room=int(np.argmax(room_probability))
            if room_probability[room]>.7 and room in self.room_masks and self.room_masks[room].any():valid&=self.room_masks[room]
        return pd,ps,valid
    def candidates(self,score,valid):
        p=score.copy();p[~valid]=0
        if p.max()<=0:raise ValueError('No valid map hypotheses')
        step=float(self.atlas['step']);k=self.profile['top_k']
        if self.profile['candidate_policy']=='upstream_xy':
            picks=extract_top_k_locations(p.max(-1),p.argmax(-1),K=k,min_dist_m=self.profile['nms_m'],resolution_m_per_pixel=step)
            return [{'pose':[q['x']*step,q['y']*step,q['orientation_radians']],'raw_score':q['prob_value']} for q in picks]
        # Greedy SE(2) NMS: preserve distinct headings at the same position.
        picks=[];indices=np.argsort(p.ravel())[::-1]
        for index in indices:
            y,x,o=np.unravel_index(index,p.shape)
            if p[y,x,o]<=0:break
            pose=np.array([x*step,y*step,o*np.pi/18])
            duplicate=any(np.linalg.norm(pose[:2]-np.array(q['pose'])[:2])<=self.profile['nms_m'] and abs(wrap_angle(pose[2]-q['pose'][2]))<=np.deg2rad(self.profile['nms_deg']) for q in picks)
            if not duplicate:picks.append({'pose':pose.tolist(),'raw_score':float(p[y,x,o])})
            if len(picks)==k:break
        return picks
    def candidate_rays(self,pose):
        key=tuple(np.round(pose,8))
        if key in self.cache:return self.cache[key]
        # Use upstream refinement function unchanged, with a verified fast ray lookup backend.
        offsets={'0':0.,'5':np.deg2rad(5),'-5':np.deg2rad(-5)}
        angles=np.concatenate([COLUMN_ANGLES+pose[2]+off for off in offsets.values()]);pos=np.array(pose[:2])*100
        dist,sem,hit=cast_many(self.image,np.broadcast_to(pos,(120,2)),angles)
        lookup={round(float(a),10):(float(d*100),int(s),tuple(h)) for a,d,s,h in zip(angles,dist,sem,hit)}
        def cached_ray(image,position,angle,**kw):return lookup[round(float(angle),10)]
        cand={'x':pose[0]/.1,'y':pose[1]/.1,'orientation_radians':pose[2]}
        with patch.object(upstream_refiner,'ray_cast',cached_ray):
            data=compute_rays_from_candidate(None,self.image,cand,offsets)
        self.cache[key]=data;return data
    def refine(self,candidates,depth,labels,fusion):
        depth_cmp=depth if self.profile['fine_depth']=='upstream_literal' else axial_to_radial(depth)
        alpha={'depth':1.,'semantic':0.,'combined':.3}[fusion]
        records=[]
        for candidate in candidates:
            rays=self.candidate_rays(candidate['pose']);trials=[]
            for offset,(d,s) in rays.items():
                cost=measure_similarity(np.asarray(depth_cmp).copy(),np.asarray(labels).copy(),d,s,alpha=alpha)
                theta=float((candidate['pose'][2]+np.deg2rad(float(offset)))%(2*np.pi))
                trials.append({'pose':[candidate['pose'][0],candidate['pose'][1],theta],'fine_cost':float(cost),'offset_deg':float(offset)})
            records.append(dict(candidate,trials=trials,best=min(trials,key=lambda v:v['fine_cost'])))
        best=min((v['best'] for v in records),key=lambda v:v['fine_cost'])
        return best,records
    def run_from_volumes(self,depth,labels,pd,ps,valid,fusion='combined'):
        score={'depth':pd,'semantic':ps,'combined':.6*pd+.4*ps}[fusion].copy();score[~valid]=0
        candidates=self.candidates(score,valid);best,records=self.refine(candidates,depth,labels,fusion)
        total=score.sum();mass=score/total
        positive=mass[mass>0];entropy=float(-(positive*np.log(positive)).sum()/np.log(len(positive))) if len(positive)>1 else 0.
        return {'coarse_pose':candidates[0]['pose'],'estimated_pose':best['pose'],'raw_score':candidates[0]['raw_score'],'fine_cost':best['fine_cost'],'score_kind':'uncalibrated heuristic; NOT localization success probability','success_probability':None,'normalized_grid_mass_max':float(mass.max()),'normalized_grid_entropy':entropy,'candidates':records,'accepted_pose':None,'acceptance':'unvalidated: no held-out calibration buildings'},score
