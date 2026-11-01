"""Stream the original release; retain one scene's complete Habitat mesh/labels.

No 34GB compressed archive is staged. HDR Ptex textures are not needed for the
vertex-colored Habitat adapter; they are not represented as downloaded assets.
"""
import argparse,io,tarfile,string,hashlib,shutil,json
from pathlib import Path
from blocked_data import RemoteZip,atomic

class ReleaseStream(io.RawIOBase):
    def __init__(self):self.part=None;self.index=0;self.sources=[]
    def readable(self):return True
    def read(self,n=-1):
        assert 0<n<=32*1024**2
        chunks=[];left=n
        while left:
            if self.part is None or self.part.tell()==self.part.size:
                if self.index>=17:break
                suffix='a'+string.ascii_lowercase[self.index];self.index+=1
                url='https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.part'+suffix
                self.part=RemoteZip(url);self.sources.append(dict(url=url,bytes=self.part.size,etag=self.part.etag))
            b=self.part.read(min(left,self.part.size-self.part.tell()));chunks.append(b);left-=len(b)
        return b''.join(chunks)

def main(scene,out):
    assert scene in ['office_0','room_0']
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    required={'mesh_semantic.ply','info_semantic.json','mesh_semantic.navmesh','replica_stage.stage_config.json'}
    accepted=[];stream=ReleaseStream()
    with tarfile.open(fileobj=stream,mode='r|gz') as archive:
        for info in archive:
            parts=Path(info.name).parts
            if not info.isfile() or scene not in parts:continue
            tail=parts[parts.index(scene)+1:]
            if len(tail)!=2 or tail[0]!='habitat' or tail[1] not in required:continue
            target=out/'habitat'/tail[1];target.parent.mkdir(exist_ok=True)
            assert shutil.disk_usage(out).free-info.size>30*1024**3
            tmp=target.with_suffix(target.suffix+'.part');h=hashlib.sha256();size=0
            with archive.extractfile(info) as src,tmp.open('wb') as dst:
                while block:=src.read(4*1024**2):
                    assert shutil.disk_usage(out).free>30*1024**3
                    h.update(block);dst.write(block);size+=len(block)
            assert size==info.size;tmp.replace(target)
            accepted.append(dict(member=info.name,path=str(target),bytes=size,sha256=h.hexdigest()))
            required.remove(tail[1]);print('REPLICA_SCENE_ASSET',scene,info.name,size,flush=True)
            if not required:break
    assert not required, f'Official scene lacks required assets: {required}'
    json.loads((out/'habitat/info_semantic.json').read_text())
    assert (out/'habitat/mesh_semantic.ply').open('rb').read(3)==b'ply'
    atomic(out/'acquisition.json',dict(status='accepted',dataset='Original Replica release v1.0',scene=scene,members=accepted,source_parts=stream.sources,compressed_archive_staged=False,HDR_Ptex_downloaded=False,scope='Complete original Habitat semantic mesh, labels, navmesh and stage metadata; vertex-color rendering, not HDR Ptex rendering',full_gzip_CRC_verified=False,transport_identity_and_member_SHA256=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('scene');p.add_argument('output');a=p.parse_args();main(a.scene,a.output)
