"""Independent Unity-vs-SPVLoc structural projection check; no fitted transforms."""
import os,sys,json
from pathlib import Path
os.environ['PYOPENGL_PLATFORM']='egl';sys.path.insert(0,'/data/fyp_3d/sources/spvloc')
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from spvloc.config.defaults import get_cfg_defaults
from spvloc.data.dataset import load_scene_data
from spvloc.utils.render import render_scene
from spvloc.utils.projection import projects_onto_floor
from spvloc.utils.transformations import parse_camera_info
ROOT=Path('/data/fyp_3d/datasets/procthor_spvloc');CLASSES={'room':1,'Ceiling_room':2,'wall':3,'door':4,'window':5}
def qualify(scene):
 receipt=json.loads((scene/'capture_receipt.json').read_text());cfg=get_cfg_defaults();cfg.merge_from_file('/data/fyp_3d/sources/spvloc/configs/config_s3d.yaml');cfg.DATASET.PATH=str(ROOT)
 data=load_scene_data([receipt['scene_id']],str(ROOT),'pickle_v3_full160',True,for_visualisation=True,config=cfg)[0];errs=[];agreements=[];counts=np.zeros(6,dtype=np.int64);checks=[]
 for i,rec in enumerate(receipt['records']):
  p=scene/rec['path']
  for f in [p/'rgb_rawlight.png',p/'instance_rgb.png',p/'class_rgb.png']:
   with Image.open(f) as im:im.load();assert im.size==(256,256)
  raw=np.asarray(Image.open(p/'instance_rgb.png'));lab=np.zeros((256,256),np.uint8)
  for name,color in json.loads((p/'instance_colors.json').read_text()).items():
   cls=name.split('|')[0]
   if cls in CLASSES:lab[np.all(raw==color,axis=2)]=CLASSES[cls]
  counts+=np.bincount(lab.ravel(),minlength=6)
  T=np.loadtxt(p/'camera_to_world_mm.txt');R,t,K=parse_camera_info(np.loadtxt(p/'camera_pose.txt'),256,256)
  assert np.allclose(T[:,3],t) and np.allclose(T[:,:3]@np.diag([1,-1,1]),R,atol=1e-8)
  d=np.load(p/'depth_m.npy');assert d.shape==(256,256) and np.isfinite(d).all() and (d>0).mean()>.5
  if i%5:continue
  idx=projects_onto_floor(t,data['floor_planes']);assert idx>=0
  pose=np.eye(4);pose[:3,:3]=R;pose[:3,3]=t
  sem=render_scene(cfg,data['geometry_clipped'],pose,data['materials_clipped'],K,mode='semantic',room_idx=None,clip_holes=True)
  pred=render_scene(cfg,data['geometry_clipped'],pose,data['materials_clipped'],K,mode='depth',room_idx=None,clip_holes=True)
  ys,xs=np.mgrid[:256,:256];radial=d*np.sqrt(1+((xs+.5-128)/128)**2+((ys+.5-128)/128)**2)
  mask=binary_erosion((lab>=1)&(lab<=3),iterations=3)&(d>0);error=np.abs(pred[mask]-radial[mask])
  if len(error)<=100:
   checks.append(dict(query=rec['query'],pixels=int(mask.sum()),status='NO_VISIBLE_ARCHITECTURE_PROBE_NOT_APPLICABLE'))
   continue
  errs.extend(error[::16].tolist());agreement=float((sem[mask]==lab[mask]).mean());agreements.append(agreement)
  checks.append(dict(query=rec['query'],pixels=int(mask.sum()),median_depth_error_m=float(np.median(error)),p95_depth_error_m=float(np.quantile(error,.95)),semantic_agreement=agreement))
 e=np.array(errs);assert len(agreements)>=10 and len(e)*16>=10000,'Insufficient independent architectural coverage'
 passes=np.quantile(e,.95)<=.15 and np.median(agreements)>=.95
 result=dict(status='DATA_READY_PILOT' if passes else 'GEOMETRY_OR_SEMANTICS_REJECTED',scene_id=receipt['scene_id'],queries=len(receipt['records']),source_class_to_layout=CLASSES,unmapped_furniture='excluded from architectural comparison, never relabelled wall',structural_pixel_counts=counts.tolist(),geometry_checks=checks,median_depth_error_m=float(np.median(e)),p95_depth_error_m=float(np.quantile(e,.95)),median_structural_semantic_agreement=float(np.median(agreements)),thresholds=dict(depth_p95_m=.15,semantic_agreement_median=.95,erosion_pixels=3),depth_tolerance_reason='Architecture uses ProcTHOR wall polygon reference surface; 15cm includes finite wall thickness and raster error',pano_status='Cubemap centre consistency pending; only actual perspective mode eligible',geometry_validation='Full multiroom structural mesh with open door/window apertures; physical visible surfaces. Training retains original per-room semantic aperture proxies, an amodal architectural target.',eligible_official_dataset_mode='PERSP_FROM_PANO=False',formal_cross_building_benchmark=False)
 (scene/'qualification.json').write_text(json.dumps(result,indent=2));print('DATA_QUALIFICATION',scene.name,result['status'],result['p95_depth_error_m'],result['median_structural_semantic_agreement'],flush=True)
if __name__=='__main__':
 for s in sorted(ROOT.glob('scene_*')):
  if (s/'capture_receipt.json').exists():
   try:qualify(s)
   except Exception as exc:
    (s/'qualification.json').write_text(json.dumps(dict(status='BLOCKED_INVALID_DATA',error_type=type(exc).__name__,reason=str(exc))))
    print('QUALIFICATION_SCENE_FAILED',s.name,type(exc).__name__,str(exc),flush=True)
