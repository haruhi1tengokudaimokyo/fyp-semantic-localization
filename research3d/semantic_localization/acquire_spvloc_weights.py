import json,os,hashlib,shutil,time
from pathlib import Path
import requests
OUT=Path('/data/fyp_3d/models/spvloc');OUT.mkdir(exist_ok=True)
META=Path('/data/fyp_3d/runs/semantic_localization_20261102/pretrained_source.json');m=json.loads(META.read_text());u=m['url'];size=int(m['headers']['Content-Range'].split('/')[-1]);etag=m['headers']['ETag'];ca='/data/fyp_3d/runs/semantic_localization_20261102/certificates/verified_bundle.pem'
p=OUT/'ckpt_s3d.ckpt';tmp=p.with_suffix('.ckpt.part');pos=tmp.stat().st_size if tmp.exists() else 0
assert shutil.disk_usage(OUT).free-(size-pos)>100*1024**3
if not p.exists():
 while pos<size:
  end=min(size,pos+16*1024**2)-1
  for attempt in range(3):
   try:
    r=requests.get(u+f'?fyp_range={pos}-{end}',headers={'Range':f'bytes={pos}-{end}','If-Range':etag},verify=ca,timeout=(20,60));r.raise_for_status();assert r.status_code==206 and r.headers['Content-Range']==f'bytes {pos}-{end}/{size}' and len(r.content)==end-pos+1 and r.headers['ETag']==etag
    assert shutil.disk_usage(OUT).free-len(r.content)>100*1024**3
    with tmp.open('ab') as f:f.write(r.content)
    pos=end+1;print('OFFICIAL_WEIGHTS',pos,size,flush=True);break
   except Exception:
    if attempt==2:raise
    time.sleep(2)
 assert tmp.stat().st_size==size;tmp.replace(p)
h=hashlib.sha256(p.read_bytes()).hexdigest();(OUT/'source.json').write_text(json.dumps(dict(url=u,bytes=size,etag=etag,sha256=h,TLS_chain_verification=True),indent=2))
print('WEIGHTS_COMPLETE',h,flush=True)
