"""Independent pose-array acceptance; failures remain in recall denominators."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from blocked_data import atomic

def evaluate(data,run):
    data,run=Path(data),Path(run)
    truth=np.load(data/'ground_truth_conditioned_motion_and_eval.npz')['T_world_camera']
    source=run/'trajectory.npz';poses=np.load(source)['T_world_camera']
    assert poses.shape==truth.shape==(600,4,4)
    valid=np.isfinite(poses).all((1,2));assert valid.any(),'No valid localization at any frame; cannot accept core completion'
    p=poses[valid];assert np.allclose(p[:,3,:],[0,0,0,1],atol=1e-6)
    assert np.allclose(np.swapaxes(p[:,:3,:3],1,2)@p[:,:3,:3],np.eye(3),atol=1e-4)
    assert np.allclose(np.linalg.det(p[:,:3,:3]),1,atol=1e-4)
    trans=np.full(len(poses),np.inf);angle=trans.copy()
    trans[valid]=np.linalg.norm(p[:,:3,3]-truth[valid,:3,3],axis=1)
    angle[valid]=np.degrees(Rotation.from_matrix(np.swapaxes(p[:,:3,:3],1,2)@truth[valid,:3,:3]).magnitude())
    np.savez_compressed(run/'independent_errors.npz',translation_m=trans,rotation_deg=angle,valid=valid)
    execution=json.loads((run/'execution.json').read_text());assert execution['localization_executed'] is True
    result=dict(status='accepted',localization_executed=True,frames=600,valid_frames=int(valid.sum()),valid_fraction=float(valid.mean()),map_coordinate_translation_rmse_valid_m=float(np.sqrt(np.mean(trans[valid]**2))),median_rotation_valid_deg=float(np.median(angle[valid])),recall_25cm_10deg=float(np.mean((trans<=.25)&(angle<=10))),recall_1m_30deg=float(np.mean((trans<=1)&(angle<=30))),unavailable_frames=np.flatnonzero(~valid).tolist(),GT_online=execution['GT_online'],trajectory_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),alignment='None: map coordinate error, not SE3 fitted trajectory',independently_recomputed=True)
    atomic(run/'metrics.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data');p.add_argument('run');a=p.parse_args();evaluate(a.data,a.run)
