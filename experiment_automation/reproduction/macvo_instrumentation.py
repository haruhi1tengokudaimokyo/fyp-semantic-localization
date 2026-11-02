"""Spawn-safe auditing of the released MAC-VO ICP solver; no GT dependencies."""
import functools,json,os,sys,time,hashlib
from pathlib import Path

_OPTIONS=None
_CALL_COUNT=0
_ORIGINAL_OPTIMIZE=None
_ORIGINAL_COVARIANCE=None


def _options():
    raw=os.environ.get('FYP_MACVO_AUDIT')
    return json.loads(raw) if raw else None


def audited_covariance(graph):
    global _CALL_COUNT
    import torch
    _CALL_COUNT+=1
    original=_ORIGINAL_COVARIANCE(graph)
    graph._fyp_cov_calls=getattr(graph,'_fyp_cov_calls',0)+1
    if _OPTIONS['mode']=='identity':
        result=torch.eye(3,dtype=original.dtype,device=original.device).expand_as(original)
        assert torch.equal(result,torch.eye(3,dtype=result.dtype,device=result.device).expand_as(result))
        return result
    return original


def audited_optimize(context,data):
    import torch
    options=_OPTIONS;root=Path(options['output']);root.mkdir(parents=True,exist_ok=True)
    frame=int(data.frame_idx.flatten()[0]);capture=None
    if options.get('capture_graphs'):
        import shutil
        assert shutil.disk_usage(root).free>30*1024**3
        folder=root/'solver_inputs';folder.mkdir(exist_ok=True)
        path=folder/f'{frame:06d}.pt';tmp=path.with_suffix('.part')
        torch.save(data,tmp);tmp.replace(path)
        capture=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    initial=data.init_motion.detach().tensor().clone().tolist()
    before=_CALL_COUNT
    start=time.perf_counter();ctx,result=_ORIGINAL_OPTIMIZE(context,data);elapsed=time.perf_counter()-start
    calls=_CALL_COUNT-before;assert calls>0,'No actual ICP covariance evaluations'

    row=dict(frame=frame,pid=os.getpid(),mode=options['mode'],wall_seconds=elapsed,
        initialization='Actual TartanMotionNet guess from live original pipeline',
        covariance_evaluations=calls,init_motion=initial,optimized_motion=result.motion.detach().tensor().tolist(),
        observations=len(data.observations),graph_capture=capture,GT_read=False)
    with (root/f'solver_audit.{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    return ctx,result


def install_from_environment():
    global _OPTIONS,_ORIGINAL_OPTIMIZE,_ORIGINAL_COVARIANCE
    options=_options()
    if options is None or _OPTIONS is not None:return
    assert options['mode'] in ['original','identity']
    sys.path.insert(0,options['source'])
    from Module.Optimization.TwoFramePGO.Graphs import ICP_TwoframePGO
    from Module.Optimization.TwoFramePGO.Optimizer import TwoFrame_PGO
    _OPTIONS=options
    _ORIGINAL_COVARIANCE=ICP_TwoframePGO.covariance_array
    _ORIGINAL_OPTIMIZE=TwoFrame_PGO._optimize
    ICP_TwoframePGO.covariance_array=audited_covariance
    TwoFrame_PGO._optimize=staticmethod(audited_optimize)


class StageProfiler:
    """CUDA events are resolved only after the complete pipeline drains.

    CPU wall spans and CUDA event spans can overlap across stages/processes;
    they must not be summed or presented as exclusive kernel execution time.
    """
    def __init__(self):self.rows={}
    def wrap(self,obj,name,label,gpu=False):
        import torch
        original=getattr(obj,name)
        @functools.wraps(original)
        def call(*args,**kwargs):
            events=None
            if gpu:
                events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True));events[0].record()
            start=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:
                wall=(time.perf_counter()-start)*1000
                if events:events[1].record()
                self.rows.setdefault(label,[]).append((wall,events))
        setattr(obj,name,call)
    def finish(self):
        import numpy as np,torch
        torch.cuda.synchronize();result={}
        for name,items in self.rows.items():
            cpu=np.array([w for w,_ in items]);cuda=np.array([a.elapsed_time(b) for _,ev in items if ev for a,b in [ev]])
            result[name]=dict(calls=len(items),cpu_wall_mean_ms=float(cpu.mean()),cpu_wall_median_ms=float(np.median(cpu)),
                cuda_span_mean_ms=float(cuda.mean()) if len(cuda) else None,
                steady_cpu_wall_mean_ms=float(cpu[2:].mean()) if len(cpu)>2 else None,
                steady_cuda_span_mean_ms=float(cuda[2:].mean()) if len(cuda)>2 else None)
        return result


# Executed when multiprocessing spawn imports this module through the wrapper.
install_from_environment()
