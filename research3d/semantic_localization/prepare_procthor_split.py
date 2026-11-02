"""Freeze real ProcTHOR house identities from publisher splits before experiments."""
import gzip,json,hashlib,subprocess
from pathlib import Path
SRC=Path('/data/fyp_3d/sources/procthor-10k');OUT=Path('/data/fyp_3d/datasets/procthor_spvloc');OUT.mkdir(parents=True,exist_ok=True)
def main():
 records=[];seen={}
 for split,count,base in [('train',10,0),('val',3,3000),('test',3,3250)]:
  accepted=0
  for i,line in enumerate(gzip.open(SRC/(split+'.jsonl.gz'),'rt')):
   h=json.loads(line)
   identity={k:h[k] for k in ['rooms','walls','doors','windows','objects']}
   sha=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
   if sha in seen:raise ValueError(f'Duplicate building geometry across/within selected splits {seen[sha]} {(split,i)}')
   if len(h['rooms'])<2 or not h['doors'] or not h['windows']:continue
   seen[sha]=(split,i);scene=f'scene_{base+accepted:05d}';p=OUT/scene;p.mkdir(exist_ok=True)
   payload=json.dumps(h,indent=2).encode();(p/'procthor_house.json').write_bytes(payload)
   records.append(dict(scene_id=base+accepted,split=split,official_house_index=i,source_geometry_sha256=sha,house_file_sha256=hashlib.sha256(payload).hexdigest(),rooms=len(h['rooms']),walls=len(h['walls']),doors=len(h['doors']),windows=len(h['windows']),status='RAW_GEOMETRY_ONLY_NOT_DATA_READY'))
   accepted+=1
   if accepted==count:break
  assert accepted==count
 manifest=dict(dataset='ProcTHOR-10K',source='https://github.com/allenai/procthor-10k',source_commit=subprocess.check_output(['git','-C',str(SRC),'rev-parse','HEAD'],text=True).strip(),license='Apache-2.0',selection='First houses in each official split with >=2 rooms, >=1 door and >=1 window. No localization score inspected.',scene_id_policy='S3D-style scene IDs are adapter IDs; publisher train/val/test is authoritative.',physical_buildings='Synthetic distinct house geometries; not real captured buildings',houses=records,rendering_ready=False)
 (OUT/'building_split_manifest.json').write_text(json.dumps(manifest,indent=2));print([(r['split'],r['official_house_index'],r['rooms'],r['windows']) for r in records])
if __name__=='__main__':main()
