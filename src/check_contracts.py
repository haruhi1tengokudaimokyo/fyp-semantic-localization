from common import *
from matching import atlas,match
from utils.raycast_utils import ray_cast
from PIL import Image

def main():
    manifest=json.loads((ROOT/'data/zind_public_sample/manifest.json').read_text()); meta,_=read_sample(manifest[0]); a=atlas(meta)
    img=np.asarray(Image.open(meta['map']).convert('RGB'))/255.
    rng=np.random.default_rng(42); coords=np.argwhere(a['valid']); idx=rng.choice(len(coords),size=100,replace=False)
    max_error=0
    for n in idx:
        y,x=coords[n]; o=int(rng.integers(36)); dist,sem,_=ray_cast(img,[x*10,y*10],o*np.pi/18)
        error=abs(float(a['depth'][y,x,o])-dist*.01); max_error=max(max_error,error)
        assert error<2e-6 and sem==a['semantic'][y,x,o],(y,x,o,error,sem)
    assert np.all(np.isfinite(meta['depth_axial_m'])) and len(meta['semantic'])==40
    assert set(meta['semantic'])<=set(range(4))
    try: match(a,np.array(meta['depth_axial_m']),np.array(meta['semantic']),np.zeros(40),np.eye(4)[meta['semantic']],'weighted_jsd')
    except ValueError: pass
    else: raise AssertionError('Must abstain on fully masked observation')
    candidates,_=match(a,np.array(meta['depth_axial_m']),np.array(meta['semantic']))
    result={'atlas_random_rays_checked':100,'atlas_max_error_m':max_error,'all_masked_abstention':True,'oracle_gt_rays_pose':candidates[0],'gt_pose':meta['pose'],'note':'Map-only oracle check; not RGB accuracy and not held-out evaluation.'}
    write_json(ROOT/'runs/contracts.json',result); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
