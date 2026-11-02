"""Real AI2-THOR acquisition: canonical poses and cubemap panoramas, no fake imagery."""
import argparse,json,hashlib,math,os,shutil,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from procthor_geometry import A,xyz
ROOT=Path('/data/fyp_3d/datasets/procthor_spvloc')
class DataController(Controller):
 @property
 def base_dir(self):return '/data/fyp_3d/tools/ai2thor'
def rotation(yaw,horizon):
 y,p=np.deg2rad([yaw,horizon]);Ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
 Rx=np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
 return Ry@Rx@np.diag([1,-1,1])
def pano_from_faces(faces,size=(512,1024)):
 H,W=size;v,u=np.mgrid[:H,:W];u=(u-W/2)/W*2*np.pi;v=-(v-H/2)/H*np.pi
 rays=np.stack((np.cos(v)*np.sin(u),np.cos(v)*np.cos(u),np.sin(v)),axis=-1)@A
 best=np.full((H,W),-1.);out=np.zeros((H,W,3),np.uint8)
 for (yaw,horizon),im in faces:
  camera=rays@rotation(yaw,horizon);z=camera[:,:,2];mask=z>best
  x=127.5+128*camera[:,:,0]/np.maximum(z,1e-9);y=127.5+128*camera[:,:,1]/np.maximum(z,1e-9)
  valid=mask&(z>0)&(x>=-.5)&(x<=255.5)&(y>=-.5)&(y<=255.5)
  for c in range(3):out[:,:,c][valid]=map_coordinates(im[:,:,c],[y[valid],x[valid]],order=1,mode='nearest')
  best[valid]=z[valid]
 assert np.all(best>0)
 return out

def capture(c,scene_id,panos=20,queries=160):
 scene=ROOT/f'scene_{scene_id:05d}';receipt=scene/'capture_receipt.json'
 previous=json.loads(receipt.read_text()) if receipt.exists() else None
 augment=previous is not None and previous['query_frames']==100 and queries==160
 if previous and not augment:print('REUSE_CAPTURE',scene_id,flush=True);return
 if augment:
  assert not (scene/'capture_receipt_pilot100.json').exists()
  (scene/'capture_receipt_pilot100.json').write_text(receipt.read_text())
 h=json.loads((scene/'procthor_house.json').read_text());event=c.reset(scene=h)
 assert event.metadata['lastActionSuccess'],event.metadata.get('errorMessage')
 (scene/'reference_object_metadata.json').write_text(json.dumps(event.metadata['objects']))
 event=c.step(action='GetReachablePositions');positions=event.metadata['actionReturn'];assert len(positions)>=panos
 # Deterministic spatial spread; no selection using localization performance.
 pos=np.array([xyz(p) for p in positions]);chosen=[int(np.argmin(pos[:,0]+pos[:,2]))]
 for _ in range(1,panos):chosen.append(int(np.argmax(np.min(np.linalg.norm(pos[:,None,[0,2]]-pos[chosen][None,:,[0,2]],axis=-1),axis=1))))
 # Class dictionary is recorded exactly as returned. Structural layout targets
 # are rendered separately from official house geometry, never from unknown IDs.
 counts=dict(previous['raw_class_pixel_counts']) if augment else {};records=list(previous['records']) if augment else [];start=time.time()
 for j,index in enumerate(chosen):
  assert shutil.disk_usage(ROOT).free>100*1024**3
  room=scene/'2D_rendering'/f'{j:05d}';pdir=room/'panorama'/'full';pdir.mkdir(parents=True,exist_ok=True)
  faces=[];camera_position=None;face_centres=[]
  for yaw,horizon in [(0,0),(90,0),(180,0),(270,0),(0,-90),(0,90)]:
   e=c.step(action='TeleportFull',position=positions[index],rotation=dict(x=0,y=yaw,z=0),horizon=horizon,standing=True,forceAction=True)
   assert e.metadata['lastActionSuccess'],e.metadata.get('errorMessage')
   faces.append(((yaw,horizon),e.frame.copy()));camera_position=xyz(e.metadata['cameraPosition'])
   face_centres.append(camera_position.copy())
  assert np.max(np.linalg.norm(np.array(face_centres)-face_centres[0],axis=1))<1e-5,'Noncentral cubemap rejected'
  if not augment:
   pano=pano_from_faces(faces);Image.fromarray(pano).save(pdir/'rgb_rawlight.png')
   np.savetxt(room/'panorama'/'camera_xyz.txt',camera_position@A.T*1000)
  else:assert np.linalg.norm(np.loadtxt(room/'panorama'/'camera_xyz.txt')-camera_position@A.T*1000)<.01
  # Five genuinely rendered different camera views at each of 20 positions.
  for k in range(3 if augment else queries//panos):
   yaw=(j*137.507764+([36,108,252][k] if augment else k*360/(queries//panos)))%360;horizon=(-10,0,10,0,-5)[k%5]
   e=c.step(action='TeleportFull',position=positions[index],rotation=dict(x=0,y=yaw,z=0),horizon=horizon,standing=True,forceAction=True)
   assert e.metadata['lastActionSuccess'];idx=100+j*3+k if augment else j*(queries//panos)+k
   actual=e.metadata['agent'];assert abs((actual['rotation']['y']-yaw+180)%360-180)<.001 and abs(actual['cameraHorizon']-horizon)<.001
   pd=room/'perspective'/'full'/f'{idx:05d}';pd.mkdir(parents=True,exist_ok=True)
   Image.fromarray(e.frame).save(pd/'rgb_rawlight.png')
   depth=np.asarray(e.depth_frame,np.float32);assert np.isfinite(depth).all() and (depth>0).mean()>.5
   np.save(pd/'depth_m.npy',depth)
   # Raw instance raster retains publisher RGB encoding and a per-frame lookup.
   Image.fromarray(e.instance_segmentation_frame).save(pd/'instance_rgb.png')
   Image.fromarray(e.class_segmentation_frame).save(pd/'class_rgb.png')
   labels={str(key):list(value) for key,value in e.object_id_to_color.items()}
   (pd/'instance_colors.json').write_text(json.dumps(labels))
   for cls,mask in e.class_masks.items():counts[cls]=counts.get(cls,0)+int(mask.sum())
   cp=xyz(e.metadata['cameraPosition']);R=A@rotation(yaw,horizon);T=np.c_[R,cp@A.T*1000]
   assert np.allclose(R.T@R,np.eye(3),atol=1e-8) and np.linalg.det(R)>.99999
   np.savetxt(pd/'camera_to_world_mm.txt',T)
   # The official loader expects camera pose as position, look-at, up, half-FoV.
   camera_info=np.r_[T[:,3],T[:,2],-T[:,1],np.pi/4,np.pi/4,1.0]
   np.savetxt(pd/'camera_info.txt',camera_info)
   np.savetxt(pd/'camera_pose.txt',camera_info)
   records.append(dict(query=idx,path=str(pd.relative_to(scene)),camera_position_unity_m=cp.tolist(),yaw_deg=yaw,horizon_deg=horizon,depth_valid_fraction=float((depth>0).mean())))
  print('PROCTHOR_CAPTURE',scene_id,j+1,panos,len(records),flush=True)
 data=dict(status='CAPTURED_NOT_DATA_READY',scene_id=scene_id,query_frames=len(records),panoramas=panos,raw_class_pixel_counts=counts,records=records,seconds=time.time()-start,ai2thor_commit='f0825767cd50d69f666c7f282e54abfe58f1e917',pose_units='canonical mm, right handed CV camera; Unity metres transformed (x,z,y)',intrinsics=dict(fx=128,fy=128,cx=127.5,cy=127.5,width=256,height=256),pano_rays='Official SPVLoc panorama.py rays, centre +Y, positive image x rotates toward +X',structural_semantics='Not inferred from unknown raster pixels; generated separately from source geometry')
 tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(receipt)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--scenes',nargs='+',type=int,default=[0,3000,3250]);args=parser.parse_args()
 c=DataController(platform=CloudRendering,width=256,height=256,fieldOfView=90,renderDepthImage=True,renderInstanceSegmentation=True,renderSemanticSegmentation=True,quality='Very Low',server_start_timeout=120)
 try:
  for scene_id in args.scenes:
   try:capture(c,scene_id)
   except Exception as exc:
    (ROOT/f'scene_{scene_id:05d}'/'capture_failure.json').write_text(json.dumps(dict(status='FAILED',error_type=type(exc).__name__,reason=str(exc))))
    print('CAPTURE_SCENE_FAILED',scene_id,type(exc).__name__,str(exc),flush=True)
 finally:c.stop()
