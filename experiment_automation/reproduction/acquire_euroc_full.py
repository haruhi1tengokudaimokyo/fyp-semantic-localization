"""Fetch only complete ASL stereo sequences from the migrated ETH publisher archive."""
import argparse,json,zipfile,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from blocked_data import RemoteZip,selected_zip,atomic
URL='https://www.research-collection.ethz.ch/server/api/core/bitstreams/7b2419c1-62b5-4714-b7f8-485e5fe3e5fe/content?download=1'
def main(name,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(RemoteZip(URL)) as z:names=z.namelist()
    nested=[n for n in names if n.endswith('/'+name+'.zip') or n==name+'.zip']
    if nested:
        assert len(nested)==1
        rows=selected_zip(URL,out/'publisher_members',lambda n:n==nested[0])
        archive=Path(rows[0]['path'])
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for info in z.infolist():
                n=Path(info.filename);assert not n.is_absolute() and '..' not in n.parts
                if info.is_dir():continue
                if 'mav0' not in n.parts:continue
                rel=Path(*n.parts[n.parts.index('mav0'):])
                if not any(x in rel.parts for x in ['cam0','cam1','state_groundtruth_estimate0']):continue
                p=out/rel;p.parent.mkdir(parents=True,exist_ok=True)
                assert shutil.disk_usage(out).free-info.file_size>30*1024**3
                with z.open(info) as src,p.with_suffix(p.suffix+'.part').open('wb') as dst:shutil.copyfileobj(src,dst)
                p.with_suffix(p.suffix+'.part').replace(p)
    else:
        rows=selected_zip(URL,out/'publisher_members',lambda n:name+'/' in n and any(x in n for x in ['/cam0/','/cam1/','/state_groundtruth_estimate0/']))
        roots=list((out/'publisher_members').rglob('mav0'));assert len(roots)==1
        if not (out/'mav0').exists():(out/'mav0').symlink_to(roots[0].resolve(),target_is_directory=True)
    root=out/'mav0';left=sorted((root/'cam0/data').glob('*.png'));right=sorted((root/'cam1/data').glob('*.png'))
    L={int(p.stem):p for p in left};R={int(p.stem):p for p in right};common=sorted(set(L)&set(R));assert len(common)>2000
    for p in [*left,*right]:
        with Image.open(p) as im:im.load();assert im.size==(752,480)
    atomic(out/'acceptance.json',dict(status='accepted',dataset='EuRoC',sequence=name,frames=len(common),left_frames=len(left),right_frames=len(right),unpaired_left=len(L)-len(common),unpaired_right=len(R)-len(common),first_timestamp_ns=common[0],last_timestamp_ns=common[-1],duration_s=(common[-1]-common[0])/1e9,full_source_sequence=True,publisher_doi='10.3929/ethz-b-000690084',source=URL,online_inputs='Official rectified stereo only; IMU/GT are not online inputs'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('sequence');p.add_argument('output');a=p.parse_args();main(a.sequence,a.output)
