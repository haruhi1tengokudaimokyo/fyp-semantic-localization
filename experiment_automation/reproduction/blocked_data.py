"""Strict public archive acquisition for the previously blocked core runs.

Only selected ZIP members are retained; CRC and source identity are checked.
No authentication workaround or disabling TLS validation is used.
"""
import argparse, io, json, os, shutil, time, zipfile, hashlib
from pathlib import Path


def atomic(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.tmp'); q.write_text(json.dumps(value, indent=2)); q.replace(p)


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        import requests
        self.url, self.pos = url, 0
        self.cache_start,self.cache=0,b''
        self.session = requests.Session()
        r = self.session.get(url, headers={'Range': 'bytes=0-0'}, timeout=(30,120), stream=True)
        r.raise_for_status(); assert r.status_code == 206, 'Server must support byte ranges'
        self.size = int(r.headers['Content-Range'].split('/')[-1]); self.etag = r.headers.get('ETag'); r.close()
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset)
        assert 0 <= self.pos <= self.size
        return self.pos
    def read(self, n=-1):
        n = min(self.size-self.pos, n if n >= 0 else self.size-self.pos)
        if not n: return b''
        assert n <= 32*1024**2
        if self.cache_start<=self.pos and self.pos+n<=self.cache_start+len(self.cache):
            data=self.cache[self.pos-self.cache_start:self.pos-self.cache_start+n];self.pos+=n;return data
        requested=n
        n=min(self.size-self.pos,max(n,8*1024**2))
        start, end = self.pos, self.pos+n-1
        headers={'Range':f'bytes={start}-{end}', 'Cache-Control':'no-cache'}
        if self.etag: headers['If-Range']=self.etag
        # Explicit range query prevents a proxy reusing an unrelated cached 206.
        url=self.url+('&' if '?' in self.url else '?')+f'fyp_range={start}-{end}'
        r=self.session.get(url, headers=headers, timeout=(30,120));r.raise_for_status()
        assert r.status_code==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{self.size}'
        if self.etag: assert r.headers.get('ETag')==self.etag
        data=r.content;assert len(data)==n
        self.cache_start,self.cache=self.pos,data
        self.pos+=requested;return data[:requested]


def selected_zip(url, out, selector):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    remote=RemoteZip(url); receipt=out/'archive_members.json'
    records=[]
    with zipfile.ZipFile(remote) as z:
        selected=[i for i in z.infolist() if not i.is_dir() and selector(i.filename)]
        assert selected, 'No requested scientific members'
        for info in selected:
            rel=Path(info.filename);assert not rel.is_absolute() and '..' not in rel.parts
            p=out/rel;p.parent.mkdir(parents=True,exist_ok=True)
            old=p.with_name(p.name+'.receipt.json')
            if p.exists() and old.exists():
                saved=json.loads(old.read_text());h=hashlib.sha256(p.read_bytes()).hexdigest()
                assert saved['sha256']==h and saved['crc32']==info.CRC and saved['bytes']==info.file_size
                records.append(saved);continue
            assert shutil.disk_usage(out).free-info.file_size>30*1024**3
            tmp=p.with_name(p.name+'.part');h=hashlib.sha256();crc=0;size=0
            import zlib
            with z.open(info) as src, tmp.open('wb') as dst:
                while block:=src.read(8*1024**2):
                    assert shutil.disk_usage(out).free>30*1024**3
                    dst.write(block);h.update(block);crc=zlib.crc32(block,crc);size+=len(block)
            assert size==info.file_size and crc==info.CRC
            tmp.replace(p)
            record=dict(path=str(p),member=info.filename,bytes=size,crc32=crc,sha256=h.hexdigest())
            atomic(old,record);records.append(record);print('ACCEPTED',info.filename,size,flush=True)
    atomic(receipt,dict(status='accepted',url=url,archive_bytes=remote.size,etag=remote.etag,members=records,full_archive_downloaded=False))
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['sg_checkpoint','sg_scannet','euroc','tartan','ace12']);p.add_argument('name');p.add_argument('output');a=p.parse_args()
    if a.kind in ['sg_checkpoint','sg_scannet']:
        ident='1S-lBXJV8oNErp0pf7j4A10VsThC17lhU' if a.kind=='sg_checkpoint' else '1GpJy65Be1HJRImB_GnTZml9x9sce2NcO'
        member='pretrained.pth.tar' if a.kind=='sg_checkpoint' else 'epoch-31.pth.tar'
        records=selected_zip(f'https://drive.usercontent.google.com/download?id={ident}&export=download&confirm=t',a.output,lambda n:n.endswith(member))
        assert len(records)==1
        atomic(Path(a.output)/'checkpoint.json',dict(status='accepted',checkpoint=records[0]['path'],sha256=records[0]['sha256'],source='SG2Loc official provided archive, SceneGraphLoc weights'))
    elif a.kind=='euroc':
        url=f'https://datasets.ifi.uzh.ch/rpg_datasets/EuRoC/{a.name}.zip'
        selected_zip(url,a.output,lambda n:n.startswith('mav0/') and any(x in n for x in ['/cam0/','/cam1/','/state_groundtruth_estimate0/']))
    elif a.kind=='tartan':
        selected_zip(f'https://tartanair.blob.core.windows.net/tartanair-test-release/stereo/{a.name}.zip',a.output,lambda n:('image_left/' in n or 'image_right/' in n or n.endswith(('pose_left.txt','pose_right.txt'))))
    else:
        selected_zip(f'https://graphics.stanford.edu/projects/reloc/data/{a.name}.zip',a.output,lambda n:True)


if __name__=='__main__':main()
