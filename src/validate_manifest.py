"""Fail-fast schema and building leakage audit for prepared real samples."""
from common import *
import argparse
from PIL import Image

def audit(path):
    value=json.loads(Path(path).read_text()); groups={}
    if isinstance(value,list): groups={'engineering_only':value}
    else: groups={k:v for k,v in value.items() if k in ['train','val','test','engineering_only']}
    buildings={}; n=0
    for split,paths in groups.items():
        buildings[split]=set()
        for p in paths:
            meta,_=read_sample(p); buildings[split].add(meta['building_id']); n+=1
            assert Image.open(Path(p)/'rgb.png').size==(640,360)
            assert len(meta['depth_axial_m'])==len(meta['semantic'])==40
            assert np.all(np.isfinite(meta['depth_axial_m']))
            assert np.all((np.array(meta['depth_axial_m'])>=0)&(np.array(meta['depth_axial_m'])<=15.01))
            assert set(meta['semantic'])<=set(range(4))
            assert len(meta['pose'])==3 and np.all(np.isfinite(meta['pose']))
            assert meta['map_m_per_pixel']==.01,'Atlas is defined at 1cm/pixel'
            h,w=np.asarray(Image.open(meta['map'])).shape[:2]
            x,y,_=meta['pose']; assert 0<=x<w*.01 and 0<=y<h*.01
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not buildings.get(a,set())&buildings.get(b,set()),f'Building leakage: {a}/{b}'
    print(json.dumps({'samples':n,'buildings':{k:sorted(v) for k,v in buildings.items()},'schema_valid':True}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');a=p.parse_args();audit(a.manifest)
