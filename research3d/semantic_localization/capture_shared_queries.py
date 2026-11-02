"""Shared unseen-house queries: simultaneous three-camera RGB, calibrated poses.
Software Vulkan rendering keeps the training GPU untouched. No synthetic text labels.
"""
import os,json,argparse,math,shutil,time
from pathlib import Path
os.environ['VK_ICD_FILENAMES']='/usr/share/vulkan/icd.d/lvp_icd.json'
os.environ['VK_DRIVER_FILES']='/usr/share/vulkan/icd.d/lvp_icd.json'
import numpy as np
from PIL import Image
from capture_procthor import DataController,rotation
from procthor_geometry import A,xyz
from ai2thor.platform import CloudRendering
SOURCE=Path('/data/fyp_3d/datasets/procthor_spvloc');OUT=Path('/data/fyp_3d/datasets/procthor_shared_localization');OUT.mkdir(exist_ok=True)
def main(scenes,limit):
 c=DataController(platform=CloudRendering,width=256,height=256,fieldOfView=90,quality='Very Low',server_start_timeout=180,server_timeout=120)
 all_samples=[]
 try:
  for sid in scenes:
   src=SOURCE/f'scene_{sid:05d}';dest=OUT/src.name;dest.mkdir(exist_ok=True)
   h=json.loads((src/'procthor_house.json').read_text());e=c.reset(scene=h);assert e.metadata['lastActionSuccess']
   objects=e.metadata['objects'];(dest/'reference_objects.json').write_text(json.dumps(objects))
   previous=json.loads((src/'capture_receipt.json').read_text())
   candidates=[]
   # Last ten reference capture positions form query-only positions; first ten
   # source positions supply the geometry baseline's known RGB-D reference map.
   for rec in previous['records']:
    pano=int(Path(rec['path']).parts[1])
    if pano>=10 and rec['horizon_deg']==0 and pano not in [x[0] for x in candidates]:candidates.append((pano,rec))
   assert len(candidates)==10
   for i,(pano,rec) in enumerate(candidates[:limit]):
    assert shutil.disk_usage(OUT).free>100*1024**3
    p=dest/f'query_{i:03d}';p.mkdir(exist_ok=True);cp=np.array(rec['camera_position_unity_m']);yaw=rec['yaw_deg']
    for camera_id,angle in enumerate([yaw-120,yaw,yaw+120]):
     args=dict(position=dict(zip(['x','y','z'],map(float,cp))),rotation=dict(x=0.,y=float(angle%360),z=0.),fieldOfView=90)
     if i==0:e=c.step(action='AddThirdPartyCamera',**args)
     else:e=c.step(action='UpdateThirdPartyCamera',thirdPartyCameraId=camera_id,**args)
     assert e.metadata['lastActionSuccess'],e.metadata.get('errorMessage')
    frames=e.third_party_camera_frames;assert len(frames)==3
    camera_meta=e.metadata.get('thirdPartyCameras',[])
    for name,im in zip(['left','center','right'],frames):
     assert im.shape==(256,256,3) and im.dtype==np.uint8;Image.fromarray(im).save(p/(name+'.png'))
    R=A@rotation(yaw,0);T=np.eye(4);T[:3,:3]=R;T[:3,3]=cp@A.T
    record=dict(scene_id=sid,query=i,source_capture_position=pano,camera_to_world_cv_m=T.tolist(),planar_GT_offline=dict(x=float(T[0,3]),y=float(T[1,3]),theta=float((np.pi/2-np.deg2rad(yaw)+np.pi)%(2*np.pi)-np.pi)),intrinsics=dict(fx=128,fy=128,cx=127.5,cy=127.5,width=256,height=256),camera_offsets_deg=[120,0,-120],FoV_deg=90,render_backend='Mesa lavapipe Vulkan CPU; no training GPU context',simultaneous=True,third_party_camera_metadata=camera_meta)
    (p/'GT_AND_CALIBRATION_OFFLINE.json').write_text(json.dumps(record,indent=2));all_samples.append(dict(scene_id=sid,query=i,path=str(p),reference_capture_positions=list(range(10)),query_capture_position=pano));print('SHARED_RGB_TRIPLET',sid,i,len(frames),flush=True)
   # ProcTHOR's exact object categories are map labels. No instance suffixes,
   # invented signs or invisible identifying attributes are given to the VLM.
   landmarks=[];classes={}
   for obj in objects:
    box=obj.get('axisAlignedBoundingBox',{});corners=np.array(box.get('cornerPoints',[]))
    if corners.shape!=(8,3):continue
    lo=corners.min(0);hi=corners.max(0);size=hi-lo
    if size[0]*size[2]<.05 or size[1]<.2 or obj.get('pickupable'):continue
    cls=obj['objectType'];name=cls
    if name not in classes:classes[name]=len(classes)
    x0,y0=lo[[0,2]];x1,y1=hi[[0,2]];v=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
    landmarks.append(dict(id=classes[name],name=name,collider=[dict(s=list(a),e=list(b)) for a,b in zip(v,v[1:]+v[:1])],label_pos=[float((x0+x1)/2),float((y0+y1)/2)],source_instance=obj['objectId']))
   floor=np.array([xyz(v) for r in h['rooms'] for v in r['floorPolygon']]);lo=floor.min(0);hi=floor.max(0)
   mapcfg=dict(map_name='ProcTHOR object-category footprint map',min_x=float(lo[0]),max_x=float(hi[0]),min_y=float(lo[2]),max_y=float(hi[2]),objects=landmarks)
   (dest/'labeled_footprint_map.json').write_text(json.dumps(mapcfg,indent=2));(dest/'map_semantic_policy.json').write_text(json.dumps(dict(type='Object-category semantic map, not a textual-sign landmark map',label_dictionary=classes,source='AI2-THOR full known reference scene metadata',repeated_class_names_share_class_id=True,query_visibility_labels_not_provided_to_API=True)))
   print('SHARED_MAP',sid,len(landmarks),len(classes),flush=True)
 finally:c.stop()
 (OUT/('pilot_manifest.json' if limit<10 else 'shared_30_manifest.json')).write_text(json.dumps(dict(samples=all_samples,reference_query_camera_positions_disjoint=True,scene_ids=scenes,all_methods_use_same_center_RGB=True,VLG_additional_views='left/right 120deg camera offsets, 90deg FoV; report this information advantage',landmark_type='Object categories only; no fabricated text'),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scenes',nargs='+',type=int,default=[3250,3251,3252]);p.add_argument('--limit',type=int,default=10);a=p.parse_args();main(a.scenes,a.limit)
