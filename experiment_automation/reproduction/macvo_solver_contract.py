"""Actual original ICP solve in spawned processes; no GT read or GPU frontend."""
import argparse,copy,json,os,sys,multiprocessing as mp
from pathlib import Path
from macvo_instrumentation import install_from_environment

def solve(data,connection):
    import torch
    from types import SimpleNamespace
    from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO
    torch.set_num_threads(8)
    context=TwoFrame_PGO.init_context(SimpleNamespace(autodiff=True,graph_type='icp',device='cpu',vectorize=True))
    _,result=TwoFrame_PGO._optimize(context,data)
    connection.send(result.motion.detach().tensor().tolist());connection.close()

def main(source,run,out):
    import numpy as np,torch,pypose as pp
    from types import SimpleNamespace
    sys.path.insert(0,str(source))
    from Module.Map import MatchObs,PointNode
    from Module.Optimization.TwoFramePGO.Graphs import GraphInput
    from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO
    from Utility.Point import pixel2point_NED
    from blocked_data import atomic
    torch.set_num_threads(8);out.mkdir(parents=True,exist_ok=True)
    z=np.load(run/'observations.npz');j=5;indices=np.flatnonzero(z['edge/match2frame2/mapping']==j)
    assert len(indices)>6
    fields={k.removeprefix('match//'):torch.from_numpy(z[k][indices].copy()) for k in z.files if k.startswith('match//')}
    previous=pp.SE3(torch.from_numpy(z['frames//pose'][j-1].copy()))
    K=torch.from_numpy(z['frames//K'][j].copy());R=previous.rotation().matrix().double().expand(len(indices),3,3)
    points=PointNode.init(dict(pos_Tw=previous.Act(pixel2point_NED(fields['pixel1_uv'],fields['pixel1_d'][:,0],K)),cov_Tw=R@fields['obs1_covTc']@R.transpose(-1,-2),color=torch.zeros(len(indices),3,dtype=torch.uint8)))
    data=GraphInput(torch.tensor([j]),torch.tensor([j-1]),previous.unsqueeze(0),torch.from_numpy(z['frames//baseline'][j:j+1].copy()),MatchObs.init(fields),points,K,torch.zeros(len(indices),dtype=torch.long),'cpu')
    context=TwoFrame_PGO.init_context(SimpleNamespace(autodiff=True,graph_type='icp',device='cpu',vectorize=True))
    _,ref=TwoFrame_PGO._optimize(context,copy.deepcopy(data));reference=ref.motion.detach().tensor().numpy()
    results={}
    for mode in ['original','identity']:
        os.environ['FYP_MACVO_AUDIT']=json.dumps(dict(source=str(source),output=str(out/mode),mode=mode,capture_graphs=True))
        ctx=mp.get_context('spawn');parent,child=ctx.Pipe();worker=ctx.Process(target=solve,args=(copy.deepcopy(data),child));worker.start();child.close()
        assert parent.poll(90),'Spawned real solver timed out'
        results[mode]=np.array(parent.recv());worker.join(15);assert worker.exitcode==0
        logs=list((out/mode).glob('solver_audit.*.jsonl'));assert logs
        audit=json.loads(logs[0].read_text().splitlines()[0]);assert audit['covariance_evaluations']>0 and audit['pid']!=os.getpid()
    difference=float(np.max(np.abs(reference-results['original'])));assert difference<1e-5,difference
    changed=float(np.max(np.abs(results['identity']-results['original'])));assert changed>1e-8,changed
    atomic(out/'acceptance.json',dict(status='accepted',spawn_verified=True,actual_original_ICP=True,GT_read=False,observations=len(indices),weighted_passthrough_max_difference=difference,identity_result_difference=changed))
    print((out/'acceptance.json').read_text(),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('run',type=Path);parser.add_argument('output',type=Path);a=parser.parse_args();main(a.source,a.run,a.output)
