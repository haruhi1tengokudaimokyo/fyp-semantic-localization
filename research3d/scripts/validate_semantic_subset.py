"""CPU validation of all images and poses, plus deterministic mesh geometry probes.
Thresholds are fixed before inspecting the result; unknown IDs are never relabelled.
"""
import sys,json,argparse,hashlib,concurrent.futures
from pathlib import Path
ROOT=Path('/data/fyp_3d/datasets/structured3d_original_subset')
sys.path.insert(0,str(ROOT/'audit_packages'))
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh
from acquire_semantic_subset import atomic
THRESHOLDS=dict(min_valid_depth_fraction_per_frame=.5,quaternion_norm_atol=1e-4,
 mesh_vertex_distance_p95_m=.10,mesh_vertex_distance_median_m=.03,
 minimum_mesh_coverage_within_10cm=.95,ray_depth_error_median_m=.03,
 ray_depth_error_p95_m=.10,minimum_ray_hit_fraction=.9)
def frame_check(args):
 seq,i=args
 arrays=[]
 for mode in ['rgb','depth','semantic']:
  with Image.open(seq/mode/f'{i:06d}.png') as im:im.load();arrays.append(np.array(im))
 rgb,d,s=arrays
 assert rgb.shape==(480,640,3) and rgb.dtype==np.uint8
 assert d.shape==s.shape==(480,640) and d.dtype==s.dtype==np.uint16
 valid=(d>0)&(d<65535);frac=float(valid.mean())
 assert frac>=THRESHOLDS['min_valid_depth_fraction_per_frame']
 counts=np.bincount(s.ravel(),minlength=65536)
 ys,xs=np.mgrid[40:480:100,40:640:100];v=valid[ys,xs]
 # OpenGL camera coordinates from metric z depth (publisher convention).
 z=d[ys,xs][v]/1000.;x=xs[v];y=ys[v]
 points=np.stack(((x-320)/320*z,-(y-240)/320*z,-z),axis=1)
 return counts,frac,float(d[valid].min()/1000),float(d[valid].max()/1000),points

def validate(name):
 scene=ROOT/name;mesh=trimesh.load(scene/'tsdf_mesh_w0.0.ply',process=False)
 assert isinstance(mesh,trimesh.Trimesh) and len(mesh.faces)>0 and np.isfinite(mesh.vertices).all()
 tree=cKDTree(mesh.vertices);counts=np.zeros(65536,dtype=np.int64);distances=[];seqs=[];probe_points=[];origins=[];ray_directions=[];expected_depth=[]
 failures=[]
 for seq in sorted(p for p in scene.iterdir() if p.is_dir()):
  poses=np.loadtxt(seq/'groundtruth.txt');k=np.loadtxt(seq/'intrinsics.txt');n=len(poses)
  assert poses.shape==(n,8) and np.isfinite(poses).all()
  assert np.allclose(k,[320,320,320,240,640,480],atol=1e-8)
  assert np.allclose(np.linalg.norm(poses[:,4:8],axis=1),1,atol=THRESHOLDS['quaternion_norm_atol'])
  assert np.allclose(np.diff(poses[:,0]),1/30,atol=2e-6)
  expected=[f'{i:06d}.png' for i in range(n)]
  for mode in ['rgb','depth','semantic']:assert sorted(p.name for p in (seq/mode).glob('*.png'))==expected
  rots=Rotation.from_quat(poses[:,4:8]).as_matrix();fracs=[];dmin=1e9;dmax=0
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   for i,result in enumerate(pool.map(frame_check,((seq,i) for i in range(n)))):
    c,f,lo,hi,pts=result;counts+=c;fracs.append(f);dmin=min(dmin,lo);dmax=max(dmax,hi)
    world=pts@rots[i].T+poses[i,1:4];ds=tree.query(world)[0];distances.extend(ds.tolist())
    if i in set(np.linspace(0,n-1,12,dtype=int)):
     # Unnormalised ray z coordinate -1: intersection parameter is z-depth metres.
     rays=pts/(-pts[:,2:3]);ray_directions.extend((rays@rots[i].T).tolist());origins.extend(np.tile(poses[i,1:4],(len(pts),1)).tolist());expected_depth.extend((-pts[:,2]).tolist())
    if i%200==0:print('VALIDATE',name,seq.name,i,n,flush=True)
  seqs.append(dict(agent=seq.name,frames=n,depth_valid_fraction_min=min(fracs),depth_valid_fraction_mean=float(np.mean(fracs)),depth_range_m=[dmin,dmax],pose_format='TUM c2w OpenGL; CV axes conversion diag(1,-1,-1)',timestamp_start=float(poses[0,0]),timestamp_end=float(poses[-1,0]),path_length_m=float(np.linalg.norm(np.diff(poses[:,1:4],axis=0),axis=1).sum())))
 ds=np.array(distances);coverage=float((ds<=.1).mean());geometry=dict(probes=len(ds),nearest_mesh_vertex_distance_median_m=float(np.median(ds)),nearest_mesh_vertex_distance_p95_m=float(np.quantile(ds,.95)),within_10cm_fraction=coverage,mesh_bounds=mesh.bounds.tolist(),vertices=len(mesh.vertices),faces=len(mesh.faces),mesh_semantic_attributes=False)
 atomic(ROOT/(name+'.modality_checkpoint.json'),dict(status='MODALITY_CHECKED_GEOMETRY_NOT_ACCEPTED',sequences=seqs,geometry=geometry,semantic_raw_id_pixels={str(i):int(n) for i,n in enumerate(counts) if n},thresholds=THRESHOLDS))
 print('RAY_CHECK',name,len(origins),flush=True)
 origins=np.array(origins);directions=np.array(ray_directions);expected_depth=np.array(expected_depth)
 # Bound triangle candidate memory. A single vectorised call for 1,440 rays on
 # a million-face mesh can allocate very large intermediate arrays.
 locations=[];ids=[]
 for start in range(0,len(origins),8):
  loc,ind,_=mesh.ray.intersects_location(origins[start:start+8],directions[start:start+8],multiple_hits=False)
  locations.extend(loc.tolist());ids.extend((ind+start).tolist())
 locations=np.asarray(locations);ids=np.asarray(ids,dtype=int)
 t=np.einsum('ij,ij->i',locations-origins[ids],directions[ids])/np.einsum('ij,ij->i',directions[ids],directions[ids])
 errors=np.abs(t-expected_depth[ids]);rayhit=len(ids)/len(origins)
 # Projection is checked against the independent TSDF surface, not a self-roundtrip.
 v1=directions[ids];v2=locations-origins[ids];a=v1/np.linalg.norm(v1,axis=1)[:,None];b=v2/np.linalg.norm(v2,axis=1)[:,None]
 geometry.update(ray_probes=len(origins),ray_hit_fraction=rayhit,ray_depth_error_median_m=float(np.median(errors)),ray_depth_error_p95_m=float(np.quantile(errors,.95)),ray_direction_error_max_rad=float(np.arccos(np.clip((a*b).sum(1),-1,1)).max()))
 if np.median(ds)>.03 or np.quantile(ds,.95)>.1 or coverage<.95:failures.append('DEPTH_POSE_MESH_VERTEX_COVERAGE')
 if rayhit<.9 or np.median(errors)>.03 or np.quantile(errors,.95)>.1:failures.append('INDEPENDENT_MESH_RAY_DEPTH')
 observed={str(i):int(n) for i,n in enumerate(counts) if n}
 # Mapping is separately verified against original Replica per-scene metadata.
 maps=list((ROOT/'audit_sources').glob('*'+name.lower().replace('-','_').replace('apart_','apartment_')+'*info_semantic.json'))
 mapping=None
 for p in maps:
  data=json.loads(p.read_text());mapping=dict(source=str(p),classes=data.get('classes'),objects=data.get('objects'))
 if mapping is None:failures.append('MISSING_VERIFIED_SCENE_LABEL_DICTIONARY')
 result=dict(scene=name,status='VALIDATION_BLOCKED' if failures else 'DATA_READY',dataset='ReplicaMultiAgent Plus',strict_spvloc_training_ready=False,independent_building_id=None,split='PUBLISHER_SPLIT_UNSPECIFIED_NOT_STRUCTURED3D',sequences=seqs,frames=sum(s['frames'] for s in seqs),geometry=geometry,semantic_raw_id_pixels=observed,label_dictionary=mapping,failures=failures,thresholds=THRESHOLDS,modalities=dict(rgb=True,depth=True,semantic_raw_ids=True,pose=True,geometry_mesh=True,structural_annotation=False,semantic_mesh=False),instance_count=None)
 atomic(ROOT/(name+'.validation.json'),result)
 print('VALIDATION_RESULT',name,result['status'],geometry,failures,flush=True)
if __name__=='__main__':
 import resource
 resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
 p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();validate(a.name)
