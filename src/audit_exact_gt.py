"""Offline exact-pose ray audit; never imported by the online inference entry point."""
import argparse,json
from pathlib import Path
import numpy as np
from common import ROOT,read_sample,write_json
from localization.geometry import COLUMN_ANGLES,axial_to_radial
from localization.floorplan.representation import Floorplan
from localization.floorplan.rays import cast_many
from localization.matching.engine import measure_similarity

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',default=str(ROOT/'data/zind_public_sample/manifest.json'));p.add_argument('--out',required=True);a=p.parse_args()
    paths=json.loads(Path(a.manifest).read_text());meta,_=read_sample(paths[0]);floor=Floorplan(meta['map'],meta['free_mask']);records=[]
    for path in paths:
        meta,_=read_sample(path);gt=np.array(meta['pose']);depth=np.array(meta['depth_axial_m']);labels=np.array(meta['semantic'])
        radial,sem,_=cast_many(floor.image,np.broadcast_to(gt[:2]*100,(40,2)),COLUMN_ANGLES+gt[2])
        records.append({'sample':path,'depth_max_radial_difference_m':float(abs(radial-axial_to_radial(depth)).max()),'semantic_disagreements':int((sem!=labels).sum()),'fine_cost_gt_at_exact_pose_upstream_axial_literal':float(measure_similarity(depth.copy(),labels.copy(),radial.copy(),sem.copy(),alpha=.3)),'fine_cost_gt_at_exact_pose_consistent_radial':float(measure_similarity(axial_to_radial(depth).copy(),labels.copy(),radial.copy(),sem.copy(),alpha=.3))})
    write_json(a.out,{'n_rays':len(paths)*40,'records':records,'max_depth_difference_m':max(r['depth_max_radial_difference_m'] for r in records),'total_semantic_disagreements':sum(r['semantic_disagreements'] for r in records),'mean_upstream_literal_fine_cost':float(np.mean([r['fine_cost_gt_at_exact_pose_upstream_axial_literal'] for r in records])),'mean_consistent_fine_cost':float(np.mean([r['fine_cost_gt_at_exact_pose_consistent_radial'] for r in records]))})
if __name__=='__main__':main()
