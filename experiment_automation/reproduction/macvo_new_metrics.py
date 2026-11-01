"""Original MAC-VO metrics plus independent SE3 ATE check, full new sequences only."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from blocked_data import atomic
def main(data,run):
    sys.path[:0]=['/data/fyp_3d/envs/macvo_metrics_overlay','/data/fyp_3d/sources/MAC-VO']
    from evo.core import lie_algebra
    lie_algebra.is_so3=lambda r:bool(np.allclose(r.T@r,np.eye(3),atol=1e-4,rtol=0) and np.isclose(np.linalg.det(r),1,atol=1e-4))
    from evo.core.trajectory import PosePath3D
    from Evaluation.MetricsSeq import evaluateATE,evaluateRTE,evaluateROE,evaluateRPE
    acceptance=json.loads((Path(data)/'acceptance.json').read_text());run=Path(run)
    z=np.load(run/'trajectory.npz');p=z['T_world_camera_raw'];g=z['T_world_camera_gt'];n=len(p)
    assert n==acceptance['frames'] and n>200
    gt_valid=np.isfinite(g).all((1,2));p=p[gt_valid];g=g[gt_valid]
    assert len(p)>200
    assert all(lie_algebra.is_so3(r) for r in p[:,:3,:3])
    values={}
    for name,fun in [('ATE_m',evaluateATE),('translation_relative_m_per_frame',evaluateRTE),('rotation_relative_deg_per_frame',evaluateROE),('RPE_full_transform',evaluateRPE)]:
        result=fun(PosePath3D(poses_se3=g.copy()),PosePath3D(poses_se3=p.copy()),correct_scale=False)
        values[name]={k:float(v) for k,v in result.stats.items()}
    # Independent Umeyama rigid alignment, no scale and no interpolation.
    x=p[:,:3,3];y=g[:,:3,3];xc=x-x.mean(0);yc=y-y.mean(0)
    u,_,vt=np.linalg.svd(xc.T@yc);d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt);R=u@d@vt
    independent=float(np.sqrt(np.mean(np.sum((xc@R-yc)**2,axis=1))))
    assert np.isclose(independent,values['ATE_m']['rmse'],atol=1e-5)
    atomic(run/'official_metrics.json',dict(status='accepted',frames=n,evaluated_GT_frames=int(gt_valid.sum()),GT_association_mask=gt_valid.tolist(),full_source_sequence=True,GT_online=False,original_metric_functions=True,independent_ATE_m=independent,metrics=values,lost_frames=int(z['lost_before_interpolation'][1:].sum()),scale_correction=False,interpolation=False,dataset=acceptance))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('run');a=p.parse_args();main(a.data,a.run)
