"""Run the actual persistent ACE worker on a sealed, GT-free RGB sequence."""
import argparse
import json
from pathlib import Path
import time

from research3d.fusion.contracts import read_jsonl, write_json, sha256_file
from .scheduler import AceRpcProcess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--binding",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--sample-period-s",type=float,default=1.)
    p.add_argument("--python",default="/data/fyp_3d/envs/ace/bin/python")
    p.add_argument("--source",type=Path,default=Path("/data/fyp_3d/sources/ace"))
    a=p.parse_args()
    if a.sample_period_s<0:
        p.error("nonnegative sample period required")
    dataset=a.dataset.resolve(strict=True)
    manifest=json.loads((dataset/"map_manifest.json").read_text())
    rows=read_jsonl(dataset/"query/frames.jsonl")
    a.output.mkdir(parents=True,exist_ok=False)
    rpc=AceRpcProcess([a.python,"-m","research3d.robot_sim.ace_worker","--binding",str(a.binding.resolve()),
                       "--online-root",str(dataset/"query"),"--source",str(a.source.resolve(strict=True))],a.output/"worker.log")
    selected=[];last=-float("inf")
    for r in rows:
        if r["timestamp_ns"]/1e9-last>=a.sample_period_s-1e-6:
            selected.append(r);last=r["timestamp_ns"]/1e9
    write_json(a.output/"protocol.json",dict(method="official ACE DSACStar",map_id=manifest["map_id"],
         binding_sha256=sha256_file(a.binding),input_manifest_sha256=sha256_file(dataset/"query/frames.jsonl"),
         sample_period_s=a.sample_period_s,selected_query_ids=[r["query_id"] for r in selected],
         selection="timestamps only, declared before inference, no GT read",scope="Gazebo development map, no original benchmark claim"))
    started=time.monotonic();latencies=[];failed=0
    temporary=a.output/"predictions.jsonl.partial"
    try:
        with temporary.open("w",buffering=1) as stream:
            for i,r in enumerate(selected):
                request=dict(op="localize",request_id=r["query_id"],query_id=r["query_id"],
                             map_id=manifest["map_id"],map_sha256=manifest["map_sha256"],timestamp_ns=r["timestamp_ns"],
                             image_path=str(dataset/r["image_path"]),camera=r["camera"])
                t=time.perf_counter();reply=rpc(request);elapsed=time.perf_counter()-t
                reply["rpc_wall_seconds"]=elapsed
                stream.write(json.dumps(reply,allow_nan=False)+"\n")
                latencies.append(elapsed)
                failed+=int(reply.get("status")!="ok")
                if i%10==0:
                    print(f"ACE {i+1}/{len(selected)} status={reply.get('status')} inliers={reply.get('inliers')}",flush=True)
        temporary.replace(a.output/"predictions.jsonl")
        write_json(a.output/"summary.json",dict(status="INFERENCE_COMPLETE_NOT_EVALUATED",frames=len(selected),failed=failed,
                     wall_seconds=time.monotonic()-started,GT_online=False,worker_persistent=True))
    finally:
        rpc.close()


if __name__=="__main__":main()
