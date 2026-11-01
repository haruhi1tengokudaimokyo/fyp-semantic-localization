"""Finite source-identity checked recovery of an interrupted publisher transfer."""
import argparse,time,json,hashlib,shutil
from pathlib import Path
import requests
from blocked_data import RemoteZip,atomic
def transfer(url,output):
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);part=p.with_suffix(p.suffix+'.part');binding=part.with_suffix(part.suffix+'.source.json')
    if p.exists() and p.with_suffix(p.suffix+'.json').exists():return
    remote=RemoteZip(url);offset=part.stat().st_size if part.exists() else 0
    identity=dict(url=url,total=remote.size,etag=remote.etag)
    if offset:assert json.loads(binding.read_text())==identity,'Do not append another source/version to retained partial'
    else:atomic(binding,identity)
    assert shutil.disk_usage(p.parent).free-(remote.size-offset)>30*1024**3
    remote.seek(offset)
    with part.open('ab') as f:
        while offset<remote.size:
            block=remote.read(min(8*1024**2,remote.size-offset));f.write(block);offset+=len(block)
            assert shutil.disk_usage(p.parent).free>30*1024**3
            if offset%(128*1024**2)==0:print('VERIFIED_MODEL_BYTES',offset,remote.size,flush=True)
    h=hashlib.sha256()
    with part.open('rb') as f:
        while block:=f.read(4*1024**2):h.update(block)
    assert part.stat().st_size==remote.size
    part.replace(p);atomic(p.with_suffix(p.suffix+'.json'),dict(url=url,bytes=remote.size,sha256=h.hexdigest(),zip_crc_verified=False,source_identity=identity,range_recovery=True))
def main(url,output):
    for attempt in range(3):
        try:return transfer(url,output)
        except (requests.RequestException,ConnectionError) as error:
            print('TRANSFER_INTERRUPTION',attempt+1,type(error).__name__,flush=True)
            if attempt==2:raise
            time.sleep(10*(attempt+1))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('output');a=p.parse_args();main(a.url,a.output)
