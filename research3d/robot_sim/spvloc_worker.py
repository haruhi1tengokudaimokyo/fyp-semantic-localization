"""Persistent RGB-only official SPVLoc RPC, with Top-3 uncalibrated scores."""
from __future__ import annotations
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

from .ace_worker import DatasetReadGuard
from .run_spvloc import SpvlocRuntime


class Worker:
    def __init__(self,runtime,online_root,dataset_root=None,install_guard=True):
        self.runtime=runtime
        self.root=Path(online_root).resolve(strict=True)
        self.guard=DatasetReadGuard([self.root]+([dataset_root] if dataset_root else []))
        self.last_stamp=-1
        if install_guard:sys.addaudithook(self.guard)

    def localize(self,request):
        allowed={"op","request_id","query_id","map_id","map_sha256","timestamp_ns","image_path","camera","metadata"}
        if set(request)-allowed or request.get("op")!="localize":raise ValueError("unsupported input fields; no query GT or labels")
        meta=request.get("metadata",{})
        if not isinstance(meta,dict) or set(meta)-{"sequence_id","sensor_frame"}:raise ValueError("unsupported metadata")
        stamp=request["timestamp_ns"]
        if type(stamp) is not int or stamp<0 or stamp<=self.last_stamp:raise ValueError("timestamps must increase")
        path=Path(request["image_path"]).resolve(strict=True)
        if not path.is_relative_to(self.root) or path.suffix.lower() not in {".png",".jpg",".jpeg"}:raise ValueError("RGB outside online root")
        if path.stat().st_size>16*1024**2:raise ValueError("oversized RGB")
        self.guard.allowed=path
        try:
            result=self.runtime.predict(path,request)
            self.last_stamp=stamp
            return {**result,"request_id":request["request_id"]}
        finally:self.guard.allowed=None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--references",type=Path,required=True)
    p.add_argument("--online-root",type=Path,required=True)
    p.add_argument("--dataset-root",type=Path)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--source",type=Path,default=Path("/data/fyp_3d/sources/spvloc"))
    args=p.parse_args(argv)
    os.environ.setdefault("PYOPENGL_PLATFORM","egl")
    os.environ.setdefault("TORCH_HOME","/data/fyp_3d/models/torchvision")
    rpc=os.fdopen(os.dup(sys.stdout.fileno()),"w",buffering=1)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno())
    def emit(row):rpc.write(json.dumps(row,allow_nan=False)+"\n");rpc.flush()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            runtime=SpvlocRuntime(args.references,args.checkpoint,args.source)
            worker=Worker(runtime,args.online_root,args.dataset_root)
        emit({"event":"ready","protocol":"robot_spvloc_rpc_v1",**runtime.signature})
    except Exception as exc:
        emit({"event":"fatal","error_type":type(exc).__name__,"error":str(exc)})
        return 2
    for line in sys.stdin:
        row={}
        try:
            if len(line)>65536:raise ValueError("request too long")
            row=json.loads(line)
            if not isinstance(row,dict):raise ValueError("request must be dict")
            if row.get("op")=="shutdown":emit({"event":"bye"});return 0
            with contextlib.redirect_stdout(sys.stderr):result=worker.localize(row)
            emit(result)
        except Exception as exc:
            emit({"status":"error","source":"official_spvloc","request_id":row.get("request_id"),
                  "query_id":row.get("query_id"),"map_id":row.get("map_id"),"map_sha256":row.get("map_sha256"),
                  "stamp_sec":row.get("timestamp_ns",0)/1e9,"error_type":type(exc).__name__,"error":str(exc)})
    return 0


if __name__=="__main__":raise SystemExit(main())
