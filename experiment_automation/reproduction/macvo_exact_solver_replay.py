"""Verify captured original optimizer inputs against actual spawned-worker outputs.
Unlike the historical reconstruction, retains original live motion guesses and
world points/covariances. This is a solver equivalence check, not a VO ablation.
"""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np

def main(run,source):
    import torch
    from types import SimpleNamespace
    sys.path.insert(0,str(source))
    from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO
    from blocked_data import atomic
    torch.set_num_threads(8)
    protocol=json.loads((run/'protocol.json').read_text())
    cfg=SimpleNamespace(**protocol['config']['Odometry']['optimizer']['args'])
    rows=[]
    for log in run.glob('solver_audit.*.jsonl'):rows.extend(json.loads(x) for x in log.read_text().splitlines())
    assert rows and protocol['backend_covariance']=='original'
    errors=[]
    for index,row in enumerate(sorted(rows,key=lambda v:v['frame'])):
        receipt=row['graph_capture'];path=Path(receipt['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256']
        data=torch.load(path,map_location='cpu',weights_only=False)
        assert np.array_equal(data.init_motion.tensor().numpy(),np.asarray(row['init_motion']))
        context=TwoFrame_PGO.init_context(cfg);_,answer=TwoFrame_PGO._optimize(context,data)
        error=float(np.max(np.abs(answer.motion.detach().tensor().numpy()-np.asarray(row['optimized_motion']))))
        errors.append(error)
        if index%100==0:print('EXACT_SOLVER_REPLAY',index,'max_error',max(errors),flush=True)
    maximum=max(errors);assert maximum<1e-5,maximum
    atomic(run/'exact_solver_replay.json',dict(status='accepted',captured_real_pipeline_inputs=True,
        full_saved_solver_job_coverage=True,solver_jobs=len(rows),max_pose_parameter_difference=maximum,
        motion_initialization_preserved=True,world_points_and_covariances_preserved=True,GT_read=False,
        scope='Exact captured optimizer input/output equivalence, not a new full VO trajectory or ablation'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('source',type=Path);a=p.parse_args();main(a.run,a.source)
