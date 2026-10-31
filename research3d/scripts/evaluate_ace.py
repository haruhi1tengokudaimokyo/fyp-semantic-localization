"""Run frozen known-map prediction then read GT in the offline evaluator only."""
import argparse,csv,hashlib,json,resource,time
from pathlib import Path
import numpy as np
from knownmap.contracts import Camera,SensorObservation
from knownmap.ace import ACE

def main(a):
 import torch
 torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
 root=Path(a.scene);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 cam=Camera(640,480,np.array([[525,0,320],[0,525,240],[0,0,1]]));model=ACE(a.ace,a.head)
 images=sorted((root/'test'/'rgb').glob('*.png'))[::a.stride]
 if a.limit:images=images[:a.limit]
 # Warmup is excluded and explicitly recorded.
 warm=SensorObservation('warmup',cam,images[0]);model.predict(warm)
 torch.cuda.reset_peak_memory_stats();rows=[];predictions=[];start=time.perf_counter()
 for i,p in enumerate(images):
  obs=SensorObservation(p.name,cam,p)
  state,corr=model.predict(obs)
  T=state.hypotheses[0].T_world_camera;predictions.append(T)
  # Ground truth enters here, AFTER prediction. It is never passed to ACE.
  gt=np.loadtxt(root/'test'/'poses'/p.name.replace('.color.png','.pose.txt'))
  terr=float(np.linalg.norm(T[:3,3]-gt[:3,3]));rerr=float(np.degrees(np.arccos(np.clip((np.trace(T[:3,:3].T@gt[:3,:3])-1)/2,-1,1))))
  rows.append(dict(frame=p.name,translation_m=terr,rotation_deg=rerr,inliers=state.diagnostics.inliers,runtime_ms=state.diagnostics.runtime_ms,median_reprojection_px=state.diagnostics.median_reprojection_px,status=state.diagnostics.status))
  if i%25==0:print(i,len(images),rows[-1],flush=True)
 np.savez_compressed(out/'predictions.npz',T_world_camera=np.array(predictions),frame=np.array([p.name for p in images]))
 with (out/'per_frame.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 t=np.array([r['translation_m'] for r in rows]);r=np.array([r['rotation_deg'] for r in rows]);ms=np.array([r['runtime_ms'] for r in rows])
 summary=dict(method='official ACE + DSAC*, GT-free adapter',scene=root.name,n=len(rows),seed=a.seed,stride=a.stride,median_translation_m=float(np.median(t)),median_rotation_deg=float(np.median(r)),recall={f'{m}m_{deg}deg':float(np.mean((t<m)&(r<deg))) for m,deg in [(0.01,1),(0.02,2),(0.05,5),(0.1,5),(0.25,10),(1,30)]},failure_rate_5cm_5deg=float(np.mean((t>=.05)|(r>=5))),invalid_or_low_inlier_rate=float(np.mean([x['status']=='failed' for x in rows])),runtime_mean_ms=float(ms.mean()),runtime_p95_ms=float(np.percentile(ms,95)),warmup_frames=1,torch_peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,torch_peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2,process_maxrss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,wall_seconds=time.perf_counter()-start,head_sha256=hashlib.sha256(Path(a.head).read_bytes()).hexdigest(),timing_scope='RGB read + grayscale/normalize + CUDA forward synchronized + CPU DSAC* + diagnostic reprojection, excludes model loading; no robot transport latency',covariance='unavailable; never inferred from inlier score')
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('scene');p.add_argument('head');p.add_argument('output');p.add_argument('--ace',required=True);p.add_argument('--stride',type=int,default=1);p.add_argument('--limit',type=int);p.add_argument('--seed',type=int,default=1305);main(p.parse_args())
