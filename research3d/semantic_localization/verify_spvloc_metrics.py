"""Recompute every saved pose error from separately stored offline GT."""
import json,math,sys
from pathlib import Path
import numpy as np

def verify(root,label):
 summary=json.loads((root/f'metrics_{label}.json').read_text())
 records=[json.loads(p.read_text()) for p in sorted(root.glob(f'eval_{label}_*.json'))]
 truth={int(p.stem.rsplit('_',1)[1]):json.loads(p.read_text())['poses'] for p in root.glob('offline_evaluation_gt_*.json')}
 seen=set();by_scene={};global_rows={key:[] for key in ['initial_estimates','refined_estimates']}
 for rec in records:
  sid=rec['scene_id'];index=rec['query_index'];assert (sid,index) not in seen;seen.add((sid,index))
  gt=np.array(truth[sid][index],float);scene=by_scene.setdefault(sid,{k:[] for k in global_rows})
  for key in global_rows:
   output=rec['estimates'][key];pose=np.array(output['pose'],float);assert np.isfinite(pose).all()
   delta=pose[:3,3]-gt[:3,3];translation=math.sqrt(sum(float(x*x) for x in delta))/1000
   cosine=(float(np.einsum('ij,ij->',pose[:3,:3],gt[:3,:3]))-1)/2
   rotation=math.degrees(math.acos(max(-1,min(1,cosine))))
   assert abs(translation-output['translation_error_m'])<1e-6
   assert abs(rotation-output['rotation_error_deg'])<1e-4
   row=[translation,rotation,float(translation<1 and rotation<30)];scene[key].append(row);global_rows[key].append(row)
 assert len(seen)==summary['queries'] and set(by_scene)==set(summary['split']['test'])
 def aggregate(rows):
  a=np.array(rows);return dict(queries=len(a),translation_median_m=float(np.median(a[:,0])),rotation_median_deg=float(np.median(a[:,1])),success_1m30deg=float(np.mean(a[:,2])))
 scenes={sid:{key:aggregate(rows) for key,rows in values.items()} for sid,values in by_scene.items()}
 for key,rows in global_rows.items():
  r=aggregate(rows)
  for metric in ['translation_median_m','rotation_median_deg','success_1m30deg']:assert abs(r[metric]-summary['metrics'][key][metric])<1e-4
 result=dict(status='INDEPENDENT_POSE_METRICS_ACCEPTED',queries=len(seen),per_scene=scenes,macro_success={key:float(np.mean([s[key]['success_1m30deg'] for s in scenes.values()])) for key in global_rows},GT_online=False)
 path=root/f'independent_metrics_{label}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path)
if __name__=='__main__':verify(Path(sys.argv[1]),sys.argv[2])
