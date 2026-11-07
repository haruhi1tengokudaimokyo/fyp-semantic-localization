"""Official SPVLoc on Gazebo queries using the same known ProcTHOR structure.

This is cross-renderer deployment evaluation, not a fresh building benchmark:
scene_03000 was a validation building used for selecting the trained checkpoint.
The RGB-only query contract does not expose pose, room or simulator world state.
References are re-rendered at robot-camera height, not reused at human height.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    tmp.replace(path)


def adapt_annotation_to_actual_world(annotation,manifest):
    """Remove source ceiling planes absent from the actual open-roof SDF.

    Door/window labels describe genuine declared openings; they remain semantic
    opening surfaces as in official SPVLoc, not claims of solid door/window mesh.
    Reindex every surviving plane and both topology/semantic references.
    """
    world=ET.parse(manifest["world_path"]).getroot().find("world")
    model_names={m.attrib["name"] for m in world.findall("model")}
    declared={o["name"] for o in manifest["objects"]}
    if model_names != declared|{"robot"}:raise ValueError("SDF visual model inventory differs from declared map")
    has_ceiling=any(o["category"]=="ceiling" for o in manifest["objects"])
    result=copy.deepcopy(annotation)
    removed=[i for i,p in enumerate(result["planes"]) if p["type"]=="ceiling" and not has_ceiling]
    keep=[i for i in range(len(result["planes"])) if i not in removed]
    remap={old:new for new,old in enumerate(keep)}
    result["planes"]=[{**result["planes"][old],"ID":new} for new,old in enumerate(keep)]
    result["planeLineMatrix"]=[result["planeLineMatrix"][old] for old in keep]
    for semantic in result["semantics"]:
        semantic["planeID"]=[remap[p] for p in semantic["planeID"] if p in remap]
    if len(result["planeLineMatrix"])!=len(result["planes"]):raise ValueError("plane topology length mismatch")
    if any(not 0<=p<len(keep) for s in result["semantics"] for p in s["planeID"]):raise ValueError("dangling semantic plane ID")
    audit={"actual_world_has_ceiling_geometry":has_ceiling,"source_ceiling_plane_ids_removed":removed,
           "retained_planes":len(keep),"retained_plane_ids_contiguous":True,
           "actual_sdf_model_inventory_verified":True,
           "door_window_semantics":"source-declared openings; no solid door/window mesh is asserted",
           "world_light_named_ceiling_is_not_a_geometry_plane":True}
    result["gazebo_map_adapter"]=audit
    return result,audit


def reflection_to_optical(value):
    """Official SPV [right,up,forward] reflection -> ROS [right,down,forward]."""
    matrix=np.asarray(value,dtype=float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all():
        raise ValueError("SPV pose must be finite 4x4")
    if not np.allclose(matrix[3],[0,0,0,1],atol=1e-6):raise ValueError("invalid homogeneous row")
    rot=matrix[:3,:3]
    if not np.allclose(rot.T@rot,np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(rot),-1.,atol=1e-4):
        raise ValueError("expected original SPVLoc reflected camera frame")
    out=matrix.copy()
    out[:3,:3]=rot@np.diag([1.,-1.,1.])
    out[:3,3]/=1000.
    if not np.isclose(np.linalg.det(out[:3,:3]),1.,atol=1e-4):raise ValueError("conversion did not produce SO(3)")
    return out


def crop_camera(image,camera):
    """Centered square crop preserves rays before official 256x256 resize."""
    width,height=image.size
    if (width,height)!=(camera["width"],camera["height"]):raise ValueError("image/calibration mismatch")
    k=np.asarray(camera["K"],dtype=float)
    if k.shape!=(3,3) or not np.isfinite(k).all():raise ValueError("invalid calibration")
    if not np.allclose(k[:2,2],[width/2,height/2],atol=.51):raise ValueError("noncentral calibration requires explicit rectification")
    if not np.isclose(k[0,0],k[1,1],rtol=1e-5):raise ValueError("non-square pixels unsupported by current adapter")
    side=min(width,height)
    left,top=(width-side)//2,(height-side)//2
    crop=image.crop((left,top,left+side,top+side))
    cam=np.eye(3)
    cam[0,0]=2*k[0,0]/side
    cam[1,1]=2*k[1,1]/side
    return crop,cam,{"left":left,"top":top,"size":side,
                     "horizontal_fov_deg":float(np.rad2deg(2*np.arctan(side/(2*k[0,0])))),
                     "original_width":width,"original_height":height}


def online_frames(path,root,map_id):
    from research3d.fusion.contracts import no_truth
    rows=[]
    seen=set()
    previous=-1
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        row=json.loads(line)
        no_truth(row)
        if any("pose" in k.lower() or "room_id" in k.lower() for k in row):
            raise ValueError("online frames may not contain query pose or room labels")
        if row["map_id"]!=map_id:raise ValueError("cross-map query")
        if row["query_id"] in seen or type(row["timestamp_ns"]) is not int or row["timestamp_ns"]<=previous:
            raise ValueError("queries must be unique with increasing capture time")
        seen.add(row["query_id"])
        previous=row["timestamp_ns"]
        image=(root/row["image_path"]).resolve()
        if not image.is_relative_to(root.resolve()):raise ValueError("RGB path leaves dataset root")
        if not image.is_file() or sha256(image)!=row["image_sha256"]:raise ValueError("RGB missing/hash mismatch")
        rows.append((row,image))
    if not rows:raise ValueError("empty query sequence")
    return rows


class SpvlocRuntime:
    """Reusable official inference; constructor reads only map and model artifacts."""
    def __init__(self, reference_root, checkpoint, source, device="cuda"):
        self.root=Path(reference_root).resolve()
        self.signature=json.loads((self.root/"reference_binding.json").read_text())
        self.protocol=json.loads((self.root/"protocol.json").read_text())
        self.checkpoint_sha256=sha256(checkpoint)
        if self.checkpoint_sha256!=self.protocol["checkpoint_sha256"]:raise ValueError("checkpoint binding mismatch")
        if sha256(self.root/"semantic_references.pt")!=self.protocol["reference_sha256"]:raise ValueError("reference checksum mismatch")
        sys.path.insert(0,str(Path(source).resolve()))
        os.chdir(source)
        import torch
        from spvloc.config.defaults import get_cfg_defaults
        from spvloc.data.data_inference import scene_from_json
        from spvloc.utils.pytorch3d.meshes import Meshes
        from spvloc.model.spvloc import PerspectiveImageFromLayout
        self.torch=torch
        self.device=device
        torch.set_num_threads(2)
        self.config=get_cfg_defaults()
        self.config.merge_from_file(str(self.root/"official_config.yaml"))
        annopath=self.root/"annotation_3d.json"
        if sha256(annopath)!=self.signature["annotation_sha256"]:raise ValueError("annotation binding mismatch")
        self.scene,_=scene_from_json(json.loads(annopath.read_text()),tmp_path=str(self.root/"geometry_cache"),
                                    name="structure_"+sha256(annopath)[:16])
        floor=self.scene["floor_planes"]
        self.scene["floor_planes"]=Meshes(floor["verts"],floor["faces"])
        self.refs=torch.load(self.root/"semantic_references.pt",map_location="cpu",weights_only=False)
        self.model=PerspectiveImageFromLayout(self.config)
        saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
        self.model.load_state_dict(saved["state_dict"],strict=True)
        del saved
        gc.collect()
        self.model=self.model.to(device).eval()

    def predict(self,path,row):
        from spvloc.data.data_inference import prepare_image
        from research3d.fusion.contracts import no_truth
        no_truth(row)
        if any("pose" in key.lower() or "room_id" in key.lower() for key in row):
            raise ValueError("query pose/room fields forbidden")
        if row["map_id"]!=self.signature["map_id"]:raise ValueError("query map mismatch")
        if row.get("map_sha256",self.signature["map_sha256"])!=self.signature["map_sha256"]:raise ValueError("query map checksum mismatch")
        image=Image.open(path).convert("RGB")
        image,camera,crop=crop_camera(image,row["camera"])
        online={**self.refs,"image":prepare_image(self.config,image).unsqueeze(0),
                "cams":self.torch.tensor(camera).unsqueeze(0)}
        if self.device=="cuda":self.torch.cuda.synchronize()
        start=time.monotonic()
        with self.torch.inference_mode():
            prediction,_=self.model.test_step_full_epoch(online,refine_iterations=1,top_n=3,prepare_plot=False,scene=self.scene)
        if self.device=="cuda":self.torch.cuda.synchronize()
        scores=prediction["max_scores"][0].tolist()
        candidates=[]
        for rank,estimate in enumerate(prediction["initial_estimates"][0],1):
            candidates.append({"rank":rank,"T_map_camera_optical":reflection_to_optical(estimate.cpu().numpy()).tolist(),
                "viewport_score":float(scores[rank-1]),"score_is_probability":False,
                "reference_index":int(prediction["max_score_idx"][0,rank-1])})
        return {"query_id":row["query_id"],"timestamp_ns":row["timestamp_ns"],"stamp_sec":row["timestamp_ns"]/1e9,
                "map_id":self.signature["map_id"],"map_sha256":self.signature["map_sha256"],"source":"official_spvloc",
                "status":"ok","candidates":candidates,
                "refined_top1_T_map_camera_optical":reflection_to_optical(prediction["refined_estimates"][0,0].cpu().numpy()).tolist(),
                "image_sha256":sha256(path),"checkpoint_sha256":self.checkpoint_sha256,
                "latency_s":time.monotonic()-start,"camera_crop":crop,"online_gt_used":False,
                "pose_frame_conversion":"SPV reflection R @ diag(1,-1,1); translation mm / 1000"}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--map-manifest",type=Path,help="prepare-only map manifest before capture finishes")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,default=Path("/data/fyp_3d/runs/semantic_localization_20261103/spvloc_100_20_20_float_targets_v2/checkpoints/epochepoch=18-stepstep=6517.ckpt"))
    p.add_argument("--source",type=Path,default=Path("/data/fyp_3d/sources/spvloc"))
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--camera-height-m",type=float)
    args=p.parse_args(argv)
    root=args.dataset_root.resolve()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if args.map_manifest and not args.prepare_only:p.error("explicit map manifest is allowed only for reference preparation")
    manifest=json.loads((args.map_manifest or root/"map_manifest.json").read_text())
    source=args.source.resolve()
    checkpoint=args.checkpoint.resolve()
    house_path=Path(manifest["source"]["path"])
    if sha256(house_path)!=manifest["source"]["sha256"]:raise ValueError("known-map source hash mismatch")
    if sha256(manifest["world_path"])!=manifest["map_sha256"]:raise ValueError("Gazebo world hash mismatch")
    height=args.camera_height_m
    if height is None:
        height=manifest["sensors"]["wheel_radius_m"]+manifest["T_base_camera_optical"][2][3]
    if not .1<height<3.:raise ValueError("invalid calibrated camera height")
    # This path is built from known map geometry only, not query rooms or GT.
    from research3d.semantic_localization.procthor_geometry import convert
    annotation,map_audit=adapt_annotation_to_actual_world(convert(json.loads(house_path.read_text())),manifest)
    annotation_path=output/"annotation_3d.json"
    atomic(annotation_path,annotation)
    atomic(output/"map_geometry_audit.json",map_audit)
    os.environ.setdefault("PYOPENGL_PLATFORM","egl")
    os.environ.setdefault("TORCH_HOME","/data/fyp_3d/models/torchvision")
    sys.path.insert(0,str(source))
    os.chdir(source)  # official custom shader resource paths are relative
    import torch
    from spvloc.config.defaults import get_cfg_defaults
    from spvloc.data.data_inference import scene_from_json,prepare_pano_renderings,prepare_image
    from spvloc.utils.pose_sampling import _sample_test_poses
    from spvloc.utils.pose_utils import process_poses_spvloc
    from spvloc.utils.pytorch3d.meshes import Meshes
    from spvloc.model.spvloc import PerspectiveImageFromLayout
    torch.set_num_threads(2)
    np.random.seed(20261107)
    torch.manual_seed(20261107)
    if args.device=="cuda":torch.cuda.manual_seed_all(20261107)
    config=get_cfg_defaults()
    config.merge_from_file(str(source/"configs/config_s3d.yaml"))
    config.TEST.IGNORE_PRETRAINED=True
    config.TEST.EVAL_GT_MAP_POINT=False
    config.TEST.PLOT_OUTPUT=False
    config.TEST.SAVE_PLOT_OUTPUT=False
    config.TEST.CAMERA_HEIGHT=int(round(height*1000))
    config.RENDER.FIX_SEED=True
    config.SYSTEM.NUM_WORKERS=0
    config.OUT_DIR=str(output)
    config.POSE_REFINE.MAX_ITERS=1
    scene,_=scene_from_json(annotation,tmp_path=str(output/"geometry_cache"),name="structure_"+sha256(annotation_path)[:16])
    # Same representation conversion as official load_scene_data (dataset.py).
    floor=scene["floor_planes"]
    scene["floor_planes"]=Meshes(floor["verts"],floor["faces"])
    signature={"map_id":manifest["map_id"],"map_sha256":manifest["map_sha256"],
               "annotation_sha256":sha256(annotation_path),"camera_height_m":height,
               "pose_grid_step_mm":config.TEST.POSE_SAMPLE_STEP,"pano_size":list(config.RENDER.PANO_SIZE)}
    reference_file=output/"semantic_references.pt"
    signature_file=output/"reference_binding.json"
    if reference_file.exists():
        if not signature_file.exists() or json.loads(signature_file.read_text())!=signature:
            raise ValueError("stale or mismatched semantic references")
        refs=torch.load(reference_file,map_location="cpu",weights_only=False)
    else:
        start=time.monotonic()
        samples=_sample_test_poses(scene["limits"],config.TEST.CAMERA_HEIGHT,scene["floor_planes"],step=config.TEST.POSE_SAMPLE_STEP)
        if not samples:raise ValueError("no official uniform-grid map references")
        normals,semantics,depth,samples_m,positions=prepare_pano_renderings(config,samples,scene)
        combined=torch.stack([torch.tensor(process_poses_spvloc(np.eye(4),np.array([position])),dtype=torch.float32) for position,_ in samples_m])
        refs={"sampled_normals":normals,"sampled_semantics":semantics,"sampled_depth":depth,
              "sampled_poses":positions,"combined_poses":combined,
              "render_time":torch.tensor(time.monotonic()-start),"sampling_time":torch.tensor(0.)}
        labels=semantics.unique().tolist()
        if any(x not in [0,1,2,3,4,5] for x in labels):raise ValueError("invalid architectural semantic labels")
        torch.save(refs,reference_file.with_suffix(".tmp"))
        reference_file.with_suffix(".tmp").replace(reference_file)
        atomic(signature_file,signature)
    summary={**signature,"reference_count":len(refs["sampled_semantics"]),
             "semantic_labels":refs["sampled_semantics"].unique().tolist(),
             "reference_sha256":sha256(reference_file),"official_core":"PerspectiveImageFromLayout.test_step_full_epoch",
             "top_k":3,"refinement":"official_one_iteration_for_top1_only",
             "query_gt_read":False,"checkpoint_sha256":sha256(checkpoint),
             "map_geometry_audit":map_audit,
             "known_domain_difference":"Gazebo structure-only RGB, 0.60m camera; training AI2-THOR furnished RGB with higher camera",
             "building_split_caveat":"scene_03000 belongs to prior validation set used for checkpoint selection; not heldout building evidence"}
    atomic(output/"protocol.json",summary)
    (output/"official_config.yaml").write_text(config.dump())
    if args.prepare_only:
        atomic(output/"status.json",{"status":"REFERENCES_PREPARED","inference_run":False})
        print(json.dumps(summary),flush=True)
        return
    rows=online_frames(root/"query/frames.jsonl",root,manifest["map_id"])
    runtime=SpvlocRuntime(output,checkpoint,source,args.device)
    output_rows=[]
    began=time.monotonic()
    for index,(row,path) in enumerate(rows):
        per_query=output/(f"query_{index:06d}.json")
        if per_query.exists():
            previous=json.loads(per_query.read_text())
            if previous["image_sha256"]!=row["image_sha256"] or previous["checkpoint_sha256"]!=summary["checkpoint_sha256"]:
                raise ValueError("previous prediction binding mismatch")
            output_rows.append(previous)
            continue
        result=runtime.predict(path,row)
        latency=result["latency_s"]
        atomic(per_query,result)
        output_rows.append(result)
        print(f"SPVLOC_GAZEBO_QUERY {index+1}/{len(rows)} {latency:.3f}s",flush=True)
    temporary=output/"predictions.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row,allow_nan=False)+"\n" for row in output_rows))
    temporary.replace(output/"predictions.jsonl")
    atomic(output/"status.json",{"status":"INFERENCE_COMPLETE_EVALUATION_PENDING","query_count":len(rows),
        "elapsed_wall_s":time.monotonic()-began,"median_query_latency_s":float(np.median([x["latency_s"] for x in output_rows])),
        "real_time_claim":False,"online_gt_used":False})


if __name__=="__main__":main()
