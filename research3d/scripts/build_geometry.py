"""Offline train-only 7Scenes RGB-D geometry, raw depth sensor coordinates.
Geometry uses depth poses and K_depth, RGB sampled after depth->RGB projection.
Query images and query poses are never used to create this map.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from knownmap.contracts import se3

def main(a):
 root=Path(a.root);src=root/'7scenes_source'/'heads';out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(Path(a.ace)/'datasets'));from setup_7scenes import d_to_rgb
 sequences=['seq-'+x.strip()[8:].zfill(2) for x in (src/'TrainSplit.txt').read_text().splitlines() if x.strip()]
 points=[];colors=[];inverse_errors=[];reprojection_errors=[];ref_ids=[]
 for seq in sequences:
  for p in sorted((src/seq).glob('*.depth.png'))[::a.frame_stride]:
   raw=np.asarray(Image.open(p));y,x=np.mgrid[0:raw.shape[0]:a.pixel_stride,0:raw.shape[1]:a.pixel_stride];z=raw[::a.pixel_stride,::a.pixel_stride].astype(float)/1000
   valid=(z>0)&(z<65.535);x=x[valid];y=y[valid];z=z[valid]
   eye=np.stack([(x-320)*z/585,(y-240)*z/585,z],-1)
   T=se3(np.loadtxt(p.with_name(p.name.replace('.depth.png','.pose.txt'))));world=eye@T[:3,:3].T+T[:3,3]
   recovered=(world-T[:3,3])@T[:3,:3]
   inverse_errors.append(float(np.max(np.linalg.norm(eye-recovered,axis=1))))
   uv=recovered[:,:2]/recovered[:,2:]*585+[320,240];reprojection_errors.append(float(np.max(np.linalg.norm(uv-np.stack([x,y],-1),axis=1))))
   rgb_eye=eye@d_to_rgb[:3,:3].T+d_to_rgb[:3,3];rgb_uv=np.rint(rgb_eye[:,:2]/rgb_eye[:,2:]*525+[320,240]).astype(int)
   mask=(rgb_eye[:,2]>0)&(rgb_uv[:,0]>=0)&(rgb_uv[:,0]<640)&(rgb_uv[:,1]>=0)&(rgb_uv[:,1]<480)
   rgb=np.asarray(Image.open(p.with_name(p.name.replace('.depth.png','.color.png'))))
   points.append(world[mask]);colors.append(rgb[rgb_uv[mask,1],rgb_uv[mask,0]]);ref_ids.append(seq+'/'+p.name)
 points=np.concatenate(points);colors=np.concatenate(colors)
 _,idx=np.unique(np.floor(points/a.voxel).astype(int),axis=0,return_index=True);points=points[idx].astype('float32');colors=colors[idx]
 path=out/'geometry.npz';np.savez_compressed(path,points_world=points,colors_rgb=colors)
 # PLY is a portable real 3D artifact, not a rendered approximation.
 with (out/'geometry.ply').open('w') as f:
  f.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
  for xyz,c in zip(points,colors):f.write(' '.join(map(str,xyz))+' '+' '.join(map(str,c))+'\n')
 manifest=dict(map_id='7scenes_heads_train_depth',geometry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference_ids=ref_ids,train_sequences=sequences,points=len(points),voxel_m=a.voxel,frame_stride=a.frame_stride,pixel_stride=a.pixel_stride,max_camera_world_roundtrip_m=max(inverse_errors),max_self_reprojection_px=max(reprojection_errors),coordinate_checks_scope='algebraic roundtrip only, not independent physical calibration proof',geometry_source='original Kinect depth and TRAIN poses only',color_source='default intrinsics and ACE Kinect extrinsic, approximate RGB/depth registration')
 (out/'map.json').write_text(json.dumps(manifest,indent=2));print(manifest)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('output');p.add_argument('--ace',required=True);p.add_argument('--frame-stride',type=int,default=10);p.add_argument('--pixel-stride',type=int,default=4);p.add_argument('--voxel',type=float,default=.01);main(p.parse_args())
