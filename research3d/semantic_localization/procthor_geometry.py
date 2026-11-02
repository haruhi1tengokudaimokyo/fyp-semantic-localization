"""ProcTHOR official house polygons -> Structured3D topology, metres -> mm.
No detected/predicted semantics are fabricated; door/window holes come from the house.
"""
import json
from pathlib import Path
import numpy as np
ROOT=Path('/data/fyp_3d/datasets/procthor_spvloc')
A=np.array([[1,0,0],[0,0,1],[0,1,0]],float)
def xyz(p):return np.array([p['x'],p['y'],p['z']],float)
def convert(h):
 junctions=[];lines=[];planes=[];semantics=[];jl={};ll={};loops=[]
 def ring(vertices):
  ids=[]
  for v in vertices:
   key=tuple(np.round(v,7))
   if key not in jl:jl[key]=len(junctions);junctions.append(dict(ID=len(junctions),coordinate=list(key)))
   ids.append(jl[key])
  edge=[]
  for a,b in zip(ids,ids[1:]+ids[:1]):
   key=tuple(sorted((a,b)))
   if key not in ll:ll[key]=len(lines);lines.append(dict(ID=len(lines),point=[0,0,0],direction=[0,0,0]))
   edge.append(ll[key])
  return edge
 def plane(vertices,kind,normal=None,holes=()):
  v=np.array(vertices);n=np.cross(v[1]-v[0],v[2]-v[0]);n=n/np.linalg.norm(n) if normal is None else np.array(normal,float)
  idx=len(planes);planes.append(dict(ID=idx,type=kind,normal=n.tolist(),offset=float(-v[0]@n)))
  edges=ring(v.tolist())
  for hole in holes:edges+=ring(hole.tolist())
  loops.append(edges);return idx
 walls={w['id']:w for w in h['walls']}
 parent={r['id']:r['id'] for r in h['rooms']}
 def find(x):
  while parent[x]!=x:x=parent[x]
  return x
 shared={}
 for w in h['walls']:
  if w.get('roomId') not in parent:continue
  key=tuple(sorted(tuple(np.round(xyz(p),6)) for p in w['polygon'][:2]))
  shared.setdefault(key,[]).append(w)
 for members in shared.values():
  if any(w.get('empty',False) for w in members):
   ids=[w['roomId'] for w in members]
   for r in ids[1:]:parent[find(r)]=find(ids[0])
 hole_by_wall={};hole_data=[]
 for kind,items in [('door',h['doors']),('window',h['windows'])]:
  for obj in items:
   w=walls[obj['wall0']];start,end=map(xyz,w['polygon'][:2]);direction=end-start;direction/=np.linalg.norm(direction)
   lo,hi=map(xyz,obj['holePolygon']);vertical=np.array([0.,1,0])
   # Hole polygon x is the local wall distance; y is the height above floor.
   q=np.array([start+direction*x+vertical*y for x,y in [(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])]])
   q=q@A.T*1000
   for wallid in {obj['wall0'],obj['wall1']}:hole_by_wall.setdefault(wallid,[]).append(q)
   p=plane(q,kind);hole_data.append(dict(ID=len(hole_data),type=kind,planeID=[p],source_object_id=obj['id']))
 groups={}
 for room in h['rooms']:
  group=find(room['id']);groups.setdefault(group,dict(ids=[],rooms=[],types=[]))
  ids=[];vs=np.array([xyz(p) for p in room['floorPolygon']])@A.T*1000
  ids.append(plane(vs,'floor',[0,0,1]))
  room_walls=[w for w in h['walls'] if w.get('roomId')==room['id']]
  height=max(p['y'] for w in room_walls for p in w['polygon'])*1000
  ceil=vs.copy();ceil[:,2]=height
  ids.append(plane(ceil,'ceiling',[0,0,-1]))
  for w in room_walls:
   if w.get('empty',False):continue
   p0,p1,p2,p3=[xyz(p)@A.T*1000 for p in w['polygon']]
   poly=np.array([p0,p1,p3,p2]);normal=np.cross(poly[1]-poly[0],poly[2]-poly[0]);normal/=np.linalg.norm(normal)
   center=np.mean(vs,axis=0);mid=np.mean(poly,axis=0)
   if (center-mid)@normal<0:normal=-normal
   ids.append(plane(poly,'wall',normal,hole_by_wall.get(w['id'],[])))
  groups[group]['ids'].extend(ids);groups[group]['rooms'].append(room['id']);groups[group]['types'].append(room['roomType'])
 for group in groups.values():
  semantics.append(dict(ID=len(semantics),type='_'.join(sorted(set(group['types']))).lower(),planeID=group['ids'],source_room_ids=group['rooms']))
 for x in hole_data:x['ID']=len(semantics);semantics.append(x)
 plm=np.zeros((len(planes),len(lines)),dtype=int);ljm=np.zeros((len(lines),len(junctions)),dtype=int)
 for i,edges in enumerate(loops):plm[i,edges]=1
 for edge,idx in ll.items():ljm[idx,list(edge)]=1
 return dict(junctions=junctions,lines=lines,planes=planes,semantics=semantics,planeLineMatrix=plm.tolist(),lineJunctionMatrix=ljm.tolist(),source_coordinates='ProcTHOR Unity metres, canonical (X,Y,Z)=(x,z,y)*1000 mm')
def main():
 for p in sorted(ROOT.glob('scene_*/procthor_house.json')):
  ann=convert(json.loads(p.read_text()));(p.parent/'annotation_3d.json').write_text(json.dumps(ann))
  print('STRUCTURAL_ANNOTATION',p.parent.name,len(ann['planes']),flush=True)
if __name__=='__main__':main()
