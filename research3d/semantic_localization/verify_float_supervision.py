"""Compare original renderer targets with post-transform tensors on real houses."""
import json,os,sys
from pathlib import Path
os.environ['PYOPENGL_PLATFORM']='egl'
sys.path.insert(0,'/data/fyp_3d/sources/spvloc')
os.chdir('/data/fyp_3d/sources/spvloc')
import numpy as np,torch
from spvloc.config.defaults import get_cfg_defaults
import spvloc.data.dataset as ds
from float_targets import install

def main():
    np.random.seed(20261102);torch.set_num_threads(1)
    cls=install();original=cls._get_room_data;raw={}
    def capture(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        raw.clear();raw.update({k:result[index].copy() for k,index in [('layout',2),('normal',3),('semantics',4)]})
        return result
    cls._get_room_data=capture
    cfg=get_cfg_defaults();cfg.merge_from_file('configs/config_s3d.yaml')
    cfg.DATASET.PATH='/data/fyp_3d/datasets/procthor_spvloc'
    cfg.DATASET.PREPARED_DATA_PATH='pickle_v4_paper_pano'
    cfg.DATASET.TRAIN_FURNITURE=['full'];cfg.DATASET.TEST_SET_FURNITURE=['full']
    cfg.DATASET.PERSP_FROM_PANO_CORRECT_ROLL=False
    scenes=[0,1,3000,3250];rows=[];valid=0
    for sid in scenes:
        ds.scenes_split=lambda *a,**kw:[sid]
        dataset=cls(cfg,'train',visualise=True)
        for index in [0,20]:
            item=dataset[index]
            expected={'persp_semantics':raw['semantics'][None],
                      'persp_layout':raw['layout'][None],
                      'persp_normal':raw['normal'].transpose(2,0,1)}
            errors={}
            for key,value in expected.items():
                observed=item[key].numpy()
                assert observed.shape==value.shape,(key,observed.shape,value.shape)
                errors[key]=float(np.max(np.abs(observed-value)))
                assert errors[key]<1e-6,(sid,index,key,errors[key])
                assert np.isfinite(observed).all()
            classes=np.unique(item['persp_semantics'].numpy())
            assert set(classes)<=set(range(6)) and item['reference_histogram'].sum().item()>.9999
            hist=item['reference_histogram'].numpy()
            passed=bool(hist[0]<.05 and (hist[1:]>.002).sum()>2 and hist[4]<.7 and hist[5]<.7)
            valid+=int(passed)
            rows.append(dict(scene_id=sid,index=index,max_absolute_transform_errors=errors,
                             classes=classes.tolist(),histogram=hist.tolist(),passes_original_filter=passed,
                             depth_min_m=float(raw['layout'].min()),depth_max_m=float(raw['layout'].max())))
    assert valid>0,'All original validity filters still fail'
    out=dict(status='PASSED',scenes=scenes,real_queries=len(rows),samples_passing_original_filter=valid,
             class_depth_normal_values_preserved=True,RGB_transform_unchanged=True,rows=rows)
    path=Path('/data/fyp_3d/runs/semantic_localization_20261103/float_target_real_data_checks.json')
    path.write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
