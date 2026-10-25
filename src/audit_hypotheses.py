"""Controlled P0 interpretation/seed probes. Results must not select a tuned test protocol."""
import argparse,json
from pathlib import Path
import numpy as np
from common import ROOT,read_sample,write_json
from localization.geometry import radial_to_axial,pose_error
from localization.models.observation import Observation
from localization.floorplan.representation import Floorplan
from localization.matching.engine import Matcher

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 data=json.loads((ROOT/'runs/audit_v2/perception_v1.json').read_text());meta,_=read_sample(data[0]['sample']);floor=Floorplan(meta['map'],meta['free_mask']);cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/official_reproduction/zind.json').read_text());cfg['room_aware']=False;matcher=Matcher(floor.rays,floor.image,cfg)
 rows=[]
 for interpretation in ['assume_axial','assume_radial']:
  for seed in [0,1,2,42,123]:
   for policy in ['argmax','multinomial']:
    for i,q in enumerate(data):
     meta,_=read_sample(q['sample']);depth=np.array(q['pred_depth']);depth=depth if interpretation=='assume_axial' else radial_to_axial(depth)
     obs=Observation(depth,np.array(q['semantic_prob']),np.array(q['depth_prob']),np.array(q['room_prob']),np.ones(40));labels=obs.labels(policy,seed+i)
     pd,ps,valid=matcher.volumes(depth,labels)
     pose=matcher.candidates(.6*pd+.4*ps,valid)[0]['pose'];xy,yaw=pose_error(pose,meta['pose'])
     rows.append({'interpretation':interpretation,'seed':seed,'semantic_policy':policy,'sample_index':i,'xy_m':xy,'yaw_deg':yaw})
    print(interpretation,seed,policy,flush=True)
 summary={'scope':'P0 hypothesis probe on existing one-building observations; no learning and no choice of best seed or convention as validated protocol','groups':{}}
 for interpretation in ['assume_axial','assume_radial']:
  for seed in [0,1,2,42,123]:
   for policy in ['argmax','multinomial']:
    r=[x for x in rows if x['interpretation']==interpretation and x['seed']==seed and x['semantic_policy']==policy];summary['groups'][f'{interpretation}/{policy}/{seed}']={'median_xy_m':float(np.median([v['xy_m'] for v in r])),'joint_recall':float(np.mean([v['xy_m']<1 and v['yaw_deg']<30 for v in r]))}
 write_json(out/'summary.json',summary);write_json(out/'per_image.json',rows)
if __name__=='__main__':main()
