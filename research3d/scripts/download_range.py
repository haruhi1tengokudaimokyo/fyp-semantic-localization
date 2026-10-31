"""Resumable HTTP range downloader, verifies Content-Range and final length.
No credentials or mirrors; logs final SHA256. ZIP integrity is checked separately.
"""
import argparse, concurrent.futures, hashlib, json, time
from pathlib import Path
import requests

def download(url, output, workers=16, chunk=4*1024**2):
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    r=requests.get(url,headers={'Range':'bytes=0-0'},stream=True,timeout=60)
    r.raise_for_status()
    if r.status_code!=206: raise RuntimeError('Server did not honor Range')
    total=int(r.headers['Content-Range'].split('/')[-1]);r.close()
    parts=output.with_suffix(output.suffix+'.parts');parts.mkdir(exist_ok=True)
    start=time.monotonic()
    def fetch(i):
        lo=i*chunk;hi=min(total,lo+chunk)-1;p=parts/str(i)
        if p.exists() and p.stat().st_size==hi-lo+1:return p
        for retry in range(6):
            try:
                with requests.get(url,headers={'Range':f'bytes={lo}-{hi}'},stream=True,timeout=90) as rr:
                    rr.raise_for_status()
                    assert rr.status_code==206 and rr.headers['Content-Range']==f'bytes {lo}-{hi}/{total}',rr.headers
                    with p.with_suffix('.tmp').open('wb') as f:
                        for data in rr.iter_content(1024*256):f.write(data)
                assert p.with_suffix('.tmp').stat().st_size==hi-lo+1
                p.with_suffix('.tmp').replace(p);print(f'part {i} complete',flush=True);return p
            except Exception:
                if retry==5:raise
                time.sleep(2**retry)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        paths=list(ex.map(fetch,range((total+chunk-1)//chunk)))
    h=hashlib.sha256()
    with output.with_suffix('.tmp').open('wb') as f:
        for p in paths:
            with p.open('rb') as g:
                while data:=g.read(1024**2):f.write(data);h.update(data)
    output.with_suffix('.tmp').replace(output)
    record=dict(url=url,bytes=total,sha256=h.hexdigest(),seconds=time.monotonic()-start)
    output.with_suffix(output.suffix+'.json').write_text(json.dumps(record,indent=2));print(record)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('url');a.add_argument('output');a.add_argument('--workers',type=int,default=16);o=a.parse_args();download(o.url,o.output,o.workers)
