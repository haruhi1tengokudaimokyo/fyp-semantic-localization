"""Audited online query API: RGB + known map only, zero GT or oracle-mask arguments."""
import argparse,json,time
from pathlib import Path
from common import write_json
from localization.models.observation import ObservationModel
from localization.floorplan.representation import Floorplan
from localization.matching.engine import Matcher

def locate(rgb,map_path,free_mask,profile,model=None):
    model=model or ObservationModel();observation=model.predict(rgb)
    floor=Floorplan(map_path,free_mask);matcher=Matcher(floor.rays,floor.image,profile)
    labels=observation.labels(profile['semantic_prediction'],profile['seed'])
    start=time.perf_counter();pd,ps,valid=matcher.volumes(observation.depth,labels,observation.valid_weights,observation.room_probability)
    result,score=matcher.run_from_volumes(observation.depth,labels,pd,ps,valid)
    result.update(profile=profile['name'],map_sha256=floor.hash,network_ms=observation.latency_ms,matching_ms=1000*(time.perf_counter()-start),peak_allocated_mib=observation.peak_allocated_mib,peak_reserved_mib=observation.peak_reserved_mib,ground_truth_used=False,room_prior_applied=False,room_note='Room polygons not supplied by this CLI; official room-aware protocol is therefore not complete')
    return result,score,observation

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rgb',required=True);p.add_argument('--map',required=True);p.add_argument('--free-mask',required=True);p.add_argument('--profile',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    profile=json.loads(Path(a.profile).read_text());result,score,obs=locate(a.rgb,a.map,a.free_mask,profile)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);write_json(out/'result.json',result)
    import numpy as np
    np.savez_compressed(out/'likelihood.npz',raw_score=score,position_mass=(score/score.sum()).sum(-1),semantic_probability=obs.semantic_probability,depth=obs.depth)
    print(json.dumps({k:result[k] for k in ['estimated_pose','score_kind','ground_truth_used','success_probability']}))
