"""RGB-only full pipeline timing. GT is read only by the external evaluator after prediction."""
import argparse,json,time
from pathlib import Path
import numpy as np
from common import ROOT,read_sample,write_json
from localization.models.observation import ObservationModel
from localization.floorplan.representation import Floorplan
from localization.matching.engine import Matcher
from localization.geometry import pose_error

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
 base=Path(__file__).resolve().parents[1];config=json.loads((base/'configs/integration/zind.json').read_text());paths=json.loads((ROOT/'data/zind_public_sample/manifest.json').read_text());meta,_=read_sample(paths[0]);floor=Floorplan(meta['map'],meta['free_mask']);matcher=Matcher(floor.rays,floor.image,config);model=ObservationModel()
 for _ in range(2):model.predict(Path(paths[0])/'rgb.png')
 rows=[]
 for i,path in enumerate(paths):
  matcher.cache.clear();start=time.perf_counter();obs=model.predict(Path(path)/'rgb.png');labels=obs.labels();pd,ps,valid=matcher.volumes(obs.depth,labels,obs.valid_weights);match_start=time.perf_counter();result,score=matcher.run_from_volumes(obs.depth,labels,pd,ps,valid);total_ms=1000*(time.perf_counter()-start)
  # Ground truth enters only here, outside the observation and matching interfaces.
  gt=json.loads((Path(path)/'sample.json').read_text())['pose'];xy,yaw=pose_error(result['estimated_pose'],gt)
  rows.append({'sample_index':i,'network_ms':obs.latency_ms,'candidate_fine_ms':1000*(time.perf_counter()-match_start),'processing_ms_including_rgb_io':total_ms,'peak_allocated_mib':obs.peak_allocated_mib,'peak_reserved_mib':obs.peak_reserved_mib,'fine_xy_m':xy,'fine_yaw_deg':yaw})
 summary={'scope':'one-building engineering benchmark; two warmup network calls; no ROS, no model loading or atlas precomputation in processing timing; fine-ray cache cleared per image','n':len(rows),'network_mean_ms':float(np.mean([r['network_ms'] for r in rows])),'processing_mean_ms':float(np.mean([r['processing_ms_including_rgb_io'] for r in rows])),'processing_p95_ms':float(np.percentile([r['processing_ms_including_rgb_io'] for r in rows],95)),'processing_fps':float(1000/np.mean([r['processing_ms_including_rgb_io'] for r in rows])),'peak_allocated_mib':max(r['peak_allocated_mib'] for r in rows),'peak_reserved_mib':max(r['peak_reserved_mib'] for r in rows)}
 write_json(out/'summary.json',summary);write_json(out/'per_image.json',rows);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
