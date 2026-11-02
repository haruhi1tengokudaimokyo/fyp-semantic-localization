"""Fetch 30 fixed official VLG-Loc queries; verify original Git LFS SHA256s."""
import hashlib,json,os,re,shutil,time
from pathlib import Path
import numpy as np,requests
ROOT=Path('/data/fyp_3d/datasets/vlg_loc_official')
OUT=Path('/data/fyp_3d/runs/semantic_localization_20261102/vlg_loc');OUT.mkdir(parents=True,exist_ok=True)
def main():
 samples=[]
 for env in sorted(ROOT.glob('env_*')):
  queries=sorted(p for p in env.glob('global_localization_*') if (p/'data.yaml').exists())
  for i in np.linspace(0,len(queries)-1,5,dtype=int):
   q=queries[i];files=[]
   for n in ['left_camera_image.png','center_camera_image.png','right_camera_image.png']:
    p=q/n;rel=p.relative_to(ROOT).as_posix();raw=p.read_bytes()
    if raw.startswith(b'version https://git-lfs.github.com/spec/v1'):
     sha=re.search(rb'oid sha256:([0-9a-f]{64})',raw)[1].decode();size=int(re.search(rb'size (\d+)',raw)[1])
     assert shutil.disk_usage(ROOT).free-size>100*1024**3
     u='https://huggingface.co/datasets/cyberagent/VLG-Loc-Dataset/resolve/main/'+rel
     for attempt in range(2):
      try:
       r=requests.get(u,timeout=(20,60));r.raise_for_status();assert len(r.content)==size
       assert hashlib.sha256(r.content).hexdigest()==sha
       temp=p.with_suffix('.download.part');temp.write_bytes(r.content);temp.replace(p);break
      except Exception:
       if attempt:raise
       time.sleep(2)
    else:sha=hashlib.sha256(raw).hexdigest();size=len(raw)
    from PIL import Image
    with Image.open(p) as im:im.verify()
    files.append(dict(path=rel,sha256=sha,bytes=size))
   samples.append(dict(environment=env.name,query=q.name,images=files))
   print('VLG_QUERY_VERIFIED',env.name,q.name,flush=True)
 manifest=dict(samples=samples,selection='5 evenly spaced lexicographically ordered queries per each of 6 publisher environments, fixed before API outputs',source_commit=os.popen('git -C '+str(ROOT)+' rev-parse HEAD').read().strip(),license='CC-BY-NC-SA-4.0',GT_policy='GT remains only in source data.yaml; query images and map labels only sent to VLM')
 p=OUT/'fixed_30_manifest.json';p.write_text(json.dumps(manifest,indent=2));print('READY',len(samples),flush=True)
if __name__=='__main__':
 p=Path('/home/csy/Documents/Codex/2026-10-31/new-chat/work/fyp/experiment_automation/state/network_environment.json')
 if p.exists():os.environ.update(json.loads(p.read_text()))
 main()
