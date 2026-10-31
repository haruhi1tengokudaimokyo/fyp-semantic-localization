"""Compare downloaded files against revision-pinned HF tree metadata."""
import argparse,hashlib,json
from pathlib import Path
import requests

def audit(root):
 root=Path(root);acq=json.loads((root/'acquisition.json').read_text());repo=acq['repo'];rev=acq['revision'];expected={};dirs={str(Path(r['path']).parent) for r in acq['files']};n=0;bad=[]
 for folder in sorted(dirs):
  url=f'https://huggingface.co/api/datasets/{repo}/tree/{rev}/'+('' if folder=='.' else folder)+'?limit=1000'
  while url:
   r=requests.get(url,timeout=90);r.raise_for_status()
   expected.update({x['path']:x for x in r.json() if x['type']=='file'});url=r.links.get('next',{}).get('url')
 for row in acq['files']:
  p=root/row['path'];x=expected[row['path']];h=hashlib.sha256(p.read_bytes()).hexdigest()
  if p.stat().st_size!=x['size'] or h!=row['sha256']:bad.append(row['path'])
  if x.get('lfs'):
   n+=1
   if h!=x['lfs']['oid']:bad.append(row['path'])
 result=dict(repo=repo,revision=rev,files=len(acq['files']),lfs_sha256_verified=n,bad=bad,total_bytes=sum(x['bytes'] for x in acq['files']))
 (root/'integrity.json').write_text(json.dumps(result,indent=2));print(result)
 if bad:raise RuntimeError('Asset integrity failure')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();audit(a.root)
