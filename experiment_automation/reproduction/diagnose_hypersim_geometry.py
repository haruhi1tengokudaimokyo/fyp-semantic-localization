"""Explain quarantined RGB-D geometry without changing selection or tolerances."""
import argparse,csv,hashlib,json
from pathlib import Path
import h5py,numpy as np
from knownmap.dataset_geometry import (hypersim_camera_model,hypersim_camera_orientation,
    hypersim_quantization_bound,gl_asset_pose_to_cv_meters,radial_distance_to_z)
SOURCE=Path('/data/fyp_3d/sources/hypersim')
def array(p):
 with h5py.File(p) as f:return f['dataset'][:]
def main(scene,output):
 root=Path('/data/fyp_3d/datasets/hypersim_formal_subset')/scene
 m=next(r for r in csv.DictReader((SOURCE/'contrib/mikeroberts3000/metadata_camera_parameters.csv').open()) if r['scene_name']==scene)
 K,Q,w,h=hypersim_camera_model(m);scale=float(m['settings_units_info_meters_scale'])
 f=sorted(root.glob('images/*/*.position.hdf5'))[0]
 cam=f.parent.name.removeprefix('scene_').removesuffix('_geometry_hdf5');frame=int(f.name.split('.')[1]);cr=root/'_detail'/cam
 ids=array(cr/'camera_keyframe_frame_indices.hdf5').reshape(-1);ix=list(ids).index(frame)
 R,rs=hypersim_camera_orientation(array(cr/'camera_keyframe_orientations.hdf5')[ix]);p=array(cr/'camera_keyframe_positions.hdf5')[ix]
 T=np.eye(4);T[:3,:3]=R;T[:3,3]=p;T=gl_asset_pose_to_cv_meters(T,scale);T[:3,:3]=T[:3,:3]@Q.T
 raw=array(f);world=raw.astype(float)*scale;depth=array(str(f).replace('position.hdf5','depth_meters.hdf5'));sem=array(str(f).replace('position.hdf5','semantic.hdf5'))
 z,valid=radial_distance_to_z(depth,K);v,u=np.indices(depth.shape);rays=np.stack([u,v,np.ones_like(u)],-1)@np.linalg.inv(K).T
 recon=(rays*z[...,None])@T[:3,:3].T+T[:3,3];mask=valid&np.isfinite(world).all(-1)
 err=np.linalg.norm(recon-world,axis=-1);bound=hypersim_quantization_bound(raw,depth,scale)
 radial=np.abs(np.linalg.norm(world-T[:3,3],axis=-1)-depth.astype(float))
 bad=mask&(err>bound);radialbad=mask&(radial>bound)
 perclass={str(int(c)):dict(pixels=int((mask&(sem==c)).sum()),ray_inconsistent=int((bad&(sem==c)).sum()),radial_inconsistent=int((radialbad&(sem==c)).sum())) for c in np.unique(sem)}
 points=np.argwhere(bad);worst=[]
 if len(points):
  ratio=(err/bound)[bad];order=np.argsort(ratio)[-10:][::-1]
  for i in order:
   y,x=points[i];worst.append(dict(u=int(x),v=int(y),label=int(sem[y,x]),error_m=float(err[y,x]),bound_m=float(bound[y,x]),radial_error_m=float(radial[y,x])))
 result=dict(status='diagnosed_not_requalified',scene=scene,camera=cam,frame=frame,pixels=int(mask.sum()),
  ray_inconsistent_pixels=int(bad.sum()),radial_inconsistent_pixels=int(radialbad.sum()),
  ray_error_p50_p95_m=np.percentile(err[mask],[50,95]).tolist(),raw_orientation_uniform_scale=rs,
  perclass=perclass,worst_pixels=worst,
  conclusion='No tolerance/selection changes. Radial consistency versus ray consistency separates depth-position mismatch from projection/ray geometry; causal attribution requires further publisher evidence.',
  position_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),localization_executed=False)
 out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);s=json.dumps(result,indent=2);digest=hashlib.sha256(s.encode()).hexdigest()
 immutable=out.parent/(out.stem+'.'+digest[:16]+'.json');immutable.write_text(s)
 temp=out.with_suffix('.tmp');temp.write_text(s);temp.replace(out)
 print(json.dumps({k:v for k,v in result.items() if k not in ['perclass','worst_pixels']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('scene');p.add_argument('output');a=p.parse_args();main(a.scene,a.output)
