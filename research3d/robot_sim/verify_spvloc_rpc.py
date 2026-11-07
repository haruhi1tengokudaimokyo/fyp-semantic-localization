"""Exercise one real RGB request through the deployed official SPVLoc RPC."""
import argparse
import json
from pathlib import Path

import numpy as np

from .scheduler import AceRpcProcess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--references",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--python",default="/data/fyp_3d/envs/semantic_loc/bin/python")
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    root=a.dataset.resolve()
    row=json.loads((root/"query/frames.jsonl").read_text().splitlines()[0])
    manifest=json.loads((root/"map_manifest.json").read_text())
    command=[a.python,"-m","research3d.robot_sim.spvloc_worker","--references",str(a.references.resolve()),
             "--online-root",str(root/"query/rgb"),"--dataset-root",str(root),"--checkpoint",str(a.checkpoint.resolve())]
    rpc=AceRpcProcess(command,a.output/"worker.log",timeout_s=90.)
    request={"op":"localize","request_id":"rpc_real_query_0","query_id":row["query_id"],
             "map_id":manifest["map_id"],"map_sha256":manifest["map_sha256"],
             "timestamp_ns":row["timestamp_ns"],"image_path":str(root/row["image_path"]),"camera":row["camera"]}
    try:
        result=rpc(request)
        (a.output/"response.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
        assert result["status"]=="ok",result
        assert result["source"]=="official_spvloc" and len(result["candidates"])==3
        assert result["image_sha256"]==row["image_sha256"]
        for candidate in result["candidates"]:
            assert np.isclose(np.linalg.det(np.asarray(candidate["T_map_camera_optical"])[:3,:3]),1.,atol=1e-4)
        cached=json.loads((a.references/"query_000000.json").read_text())
        difference=float(np.max(np.abs(np.asarray(result["refined_top1_T_map_camera_optical"])-
                                        np.asarray(cached["refined_top1_T_map_camera_optical"]))))
        assert difference<1e-3,f"RPC and offline official inference disagree: {difference}"
        receipt={"status":"PASSED_REAL_OFFICIAL_SPVLOC_RPC","actual_rgb_queries":1,"top_k":3,
                 "map_sha256":result["map_sha256"],"checkpoint_sha256":result["checkpoint_sha256"],
                 "query_gt_passed":False,"fake_prediction_used":False,"latency_s":result["latency_s"],
                 "max_pose_matrix_difference_vs_offline":difference,
                 "scope":"RPC/coordinate parity; not additional independent localization benchmark"}
        (a.output/"verification.json").write_text(json.dumps(receipt,indent=2)+"\n")
        print(json.dumps(receipt),flush=True)
    finally:rpc.close()


if __name__=="__main__":main()
