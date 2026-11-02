"""Original complete 411-query dataset: parallel small-image transfer, LFS SHA checked."""
import concurrent.futures,subprocess,json,re,hashlib,shutil,time,os
from pathlib import Path
import requests
from PIL import Image
ROOT=Path('/data/fyp_3d/datasets/vlg_loc_official');OUT=Path('/data/fyp_3d/runs/semantic_localization_20261102/vlg_loc')
def one(q):
 files=[]
 for n in ['left_camera_image.png','center_camera_image.png','right_camera_image.png']:
  p=q/n;rel=p.relative_to(ROOT).as_posix();pointer=subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+rel]);sha=re.search(rb'oid sha256:([0-9a-f]{64})',pointer)[1].decode();size=int(re.search(rb'size (\d+)',pointer)[1])
  old=p.read_bytes()
  if len(old)!=size or hashlib.sha256(old).hexdigest()!=sha:
   assert shutil.disk_usage(ROOT).free-size>100*1024**3
   for attempt in range(2):
    try:
     r=requests.get('https://huggingface.co/datasets/cyberagent/VLG-Loc-Dataset/resolve/main/'+rel,timeout=(20,60));r.raise_for_status();assert len(r.content)==size and hashlib.sha256(r.content).hexdigest()==sha
     t=p.with_suffix('.download.part');t.write_bytes(r.content);t.replace(p);break
    except Exception:
     if attempt:raise
     time.sleep(2)
  with Image.open(p) as im:im.verify()
  files.append(dict(path=rel,sha256=sha,bytes=size))
 print('OFFICIAL_QUERY_COMPLETE',q.parent.name,q.name,flush=True)
 return dict(environment=q.parent.name,query=q.name,images=files)
def main():
 queries=sorted(p for e in ROOT.glob('env_*') for p in e.glob('global_localization_*') if (p/'data.yaml').exists())
 assert len(queries)==411,len(queries)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:samples=list(ex.map(one,queries))
 m=dict(samples=samples,source_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),selection='All 411 queries in the 6 publisher environments, no selection using GT or localization output',license='CC-BY-NC-SA-4.0')
 p=OUT/'fixed_full_manifest.json';t=p.with_suffix('.json.tmp');t.write_text(json.dumps(m,indent=2));t.replace(p);print('FULL_OFFICIAL_DATA_READY',len(samples),flush=True)
if __name__=='__main__':
 env=Path('/home/csy/Documents/Codex/2026-10-31/new-chat/work/fyp/experiment_automation/state/network_environment.json')
 if env.exists():os.environ.update(json.loads(env.read_text()))
 main()
