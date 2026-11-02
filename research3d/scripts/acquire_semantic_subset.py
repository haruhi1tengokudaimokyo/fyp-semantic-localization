"""CPU-only, bounded, one-public-archive-at-a-time acquisition. No experiment mutation."""
import argparse, hashlib, json, os, shutil, stat, time, zipfile, zlib
from pathlib import Path
import requests
ROOT=Path('/data/fyp_3d/datasets/structured3d_original_subset')
GIB=1024**3

def atomic(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)
def used():return sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
_charged=0
_initial_usage=None
def guard(extra=0,charge=False):
 global _charged,_initial_usage
 if _initial_usage is None:_initial_usage=used()
 assert shutil.disk_usage(ROOT).free-extra >= 100*GIB, '100GiB free space floor'
 assert _initial_usage+_charged+extra <= 60*GIB, '60GiB dataset peak budget'
 if charge:_charged+=extra
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(8*1024**2):h.update(b)
 return h.hexdigest()
def acquire(name):
 receipt_path=ROOT/(name+'.extraction_receipt.json')
 if receipt_path.exists() and (ROOT/name).is_dir():
  verify_extracted(name)
  print('REUSED_VERIFIED_EXTRACTION',name,flush=True);return
 meta=json.loads((ROOT/(name+'_zip_directory.json')).read_text())
 url=meta['url'];size=meta['archive_bytes'];etag=meta['etag']
 archive=ROOT/(name+'.zip');part=ROOT/(name+'.zip.part');binding=ROOT/(name+'.download_identity.json')
 identity=dict(url=url,bytes=size,etag=etag)
 if binding.exists():assert json.loads(binding.read_text())==identity,'Source changed; quarantine old partial'
 else:assert not part.exists();atomic(binding,identity)
 if not archive.exists():
  pos=part.stat().st_size if part.exists() else 0
  assert pos<=size
  guard(size-pos+meta['uncompressed_bytes']+GIB)
  while pos<size:
   end=min(size,pos+64*1024**2)-1
   for attempt in range(3):
    try:
     h={'Range':f'bytes={pos}-{end}','Cache-Control':'no-cache'}
     if etag:h['If-Range']=etag
     with requests.get(url+f'&fyp_range={pos}-{end}',headers=h,stream=True,timeout=(20,60)) as r:
      r.raise_for_status();assert r.status_code==206
      assert r.headers.get('Content-Range')==f'bytes {pos}-{end}/{size}'
      if etag:assert r.headers.get('ETag')==etag
      with part.open('ab') as dst:
       n=0
       for b in r.iter_content(1024**2):guard(len(b),charge=True);dst.write(b);n+=len(b)
       dst.flush();os.fsync(dst.fileno())
      assert n==end-pos+1
     pos=end+1;print('DOWNLOAD',name,pos,size,flush=True);break
    except Exception:
     if part.exists():
      with part.open('r+b') as f:f.truncate(pos)
     if attempt==2:raise
     time.sleep(2)
  assert part.stat().st_size==size
  with zipfile.ZipFile(part) as z:
   assert len(z.infolist())==len(meta['entries'])
   assert z.testzip() is None
  part.replace(archive)
 sha=digest(archive)
 receipt=dict(**identity,sha256=sha,crc_all_members_verified=True,downloaded_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
 atomic(ROOT/(name+'.archive_receipt.json'),receipt)
 stage=ROOT/('.staging_'+name);stage.mkdir(exist_ok=True)
 guard(meta['uncompressed_bytes']+GIB)
 entries=[]
 with zipfile.ZipFile(archive) as z:
  for info in z.infolist():
   rel=Path(info.filename);assert not rel.is_absolute() and '..' not in rel.parts
   assert not stat.S_ISLNK(info.external_attr>>16)
   if info.is_dir():continue
   target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists() and target.stat().st_size==info.file_size:
    old_crc=0;old_h=hashlib.sha256()
    with target.open('rb') as f:
     while b:=f.read(1024**2):old_crc=zlib.crc32(b,old_crc);old_h.update(b)
    if old_crc==info.CRC:
     entries.append(dict(path=str(rel),bytes=info.file_size,crc32=old_crc,sha256=old_h.hexdigest()));continue
   tmp=target.with_name(target.name+'.part');h=hashlib.sha256();crc=n=0
   with z.open(info) as src,tmp.open('wb') as dst:
    while b:=src.read(1024**2):guard(len(b),charge=True);dst.write(b);h.update(b);crc=zlib.crc32(b,crc);n+=len(b)
   assert n==info.file_size and crc==info.CRC
   tmp.replace(target);entries.append(dict(path=str(rel),bytes=n,crc32=crc,sha256=h.hexdigest()))
 atomic(ROOT/(name+'.files.json'),entries)
 final=ROOT/name
 assert not final.exists()
 (stage/name).replace(final);stage.rmdir()
 atomic(ROOT/(name+'.extraction_receipt.json'),dict(archive=receipt,files=len(entries),uncompressed_bytes=sum(e['bytes'] for e in entries),source='https://mags-slam.github.io/dataset/',status='CRC_AND_SHA_VERIFIED_NOT_YET_DATA_READY'))
 print('EXTRACTED',name,len(entries),sha,flush=True)
def verify_extracted(name):
 records=json.loads((ROOT/(name+'.files.json')).read_text())
 receipt=json.loads((ROOT/(name+'.extraction_receipt.json')).read_text())
 assert len(records)==receipt['files']
 for rec in records:
  p=ROOT/rec['path']
  assert p.stat().st_size==rec['bytes'] and digest(p)==rec['sha256'],str(p)
 return records
def release_archive(name):
 assert (ROOT/(name+'.validation.json')).exists(),'Complete data audit before releasing archive'
 verify_extracted(name)
 p=ROOT/(name+'.zip');receipt=json.loads((ROOT/(name+'.archive_receipt.json')).read_text())
 if p.exists():
  assert p.stat().st_size==receipt['bytes'] and digest(p)==receipt['sha256']
  atomic(ROOT/(name+'.archive_release.json'),dict(bytes_released=p.stat().st_size,sha256=receipt['sha256'],reason='All extracted SHA256 values reverified; source receipt retained; dataset readiness recorded separately'))
  p.unlink()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name',choices=['Hotel-0','Office-2','Apart-0']);p.add_argument('--release-archive',action='store_true');a=p.parse_args()
 env=Path('/home/csy/Documents/Codex/2026-10-31/new-chat/work/fyp/experiment_automation/state/network_environment.json')
 if env.exists():os.environ.update(json.loads(env.read_text()))
 if a.release_archive:release_archive(a.name)
 else:acquire(a.name)
