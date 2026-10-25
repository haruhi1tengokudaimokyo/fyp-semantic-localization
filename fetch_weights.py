"""Download only the two author-published ZInD weights and verify recorded SHA256."""
import os,hashlib
from pathlib import Path
import gdown
ROOT=Path(os.environ.get('FYP_ROOT','/data/fyp-semrayloc'))
WEIGHTS=[('depth','1H9MQoK3aDG6-A62D3cXGRCjwvyUI9E-e','9dd1611aca820dc0aa6d341579dae4e75a74c2870332dade1e23ad7390a95cb9'),('semantic','1sSw6Pa8X14XyHQkBGaXQLgJVCl1U4_29','f1ee439e60c7cbaa9e07db29239e8274699086f7d381d0e974bdf50fa6e8a593')]
for kind,file_id,sha in WEIGHTS:
    out=ROOT/'models/zind'/f'final_{kind}.ckpt';out.parent.mkdir(parents=True,exist_ok=True)
    def digest(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        return h.hexdigest()
    if out.exists() and digest(out)==sha: print(f'{kind}: verified');continue
    partial=out.with_suffix('.download')
    gdown.download(id=file_id,output=str(partial),quiet=False,resume=True)
    if digest(partial)!=sha: raise RuntimeError(f'Checksum mismatch: {partial}')
    partial.replace(out);print(f'{kind}: downloaded and verified')
