"""P0/P1 offline diagnostics. GT data are confined to this evaluation entry point."""
import argparse,json,csv,time,hashlib,subprocess,sys
from pathlib import Path
import logging
import numpy as np
import torch
from common import ROOT,REPO,read_sample,write_json
from localization.models.observation import Observation,ObservationModel
from localization.geometry import axial_to_radial,pose_error,COLUMN_ANGLES
from localization.floorplan.representation import Floorplan
from localization.datasets.zind_audit import audit_annotations
from localization.matching.engine import Matcher
from localization.evaluation.metrics import localization_metrics,classification_metrics,reliability

BASE=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',default=str(ROOT/'data/zind_public_sample/manifest.json'));parser.add_argument('--out',required=True);parser.add_argument('--cached-perception');args=parser.parse_args()
    logging.getLogger().setLevel(logging.WARNING)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(Path(args.manifest).read_text());profiles=[json.loads((BASE/f'configs/{p}/zind.json').read_text()) for p in ['official_reproduction','integration']]
    write_json(out/'profiles.json',profiles)
    annotation,room_masks,room_bbox=audit_annotations(manifest);write_json(out/'annotation_audit.json',annotation)
    meta,_=read_sample(manifest[0]);floor=Floorplan(meta['map'],meta['free_mask']);ratio=10
    # The official room-aware code uses annotation-derived room bounding boxes. The fixed map is unchanged.
    bbox={i:m[::ratio,::ratio][:floor.rays['valid'].shape[0],:floor.rays['valid'].shape[1]].astype(bool) for i,m in room_bbox.items()}
    matchers={p['name']:Matcher(floor.rays,floor.image,p,bbox) for p in profiles}
    if args.cached_perception:
        records=json.loads(Path(args.cached_perception).read_text());observations=[Observation(np.array(q['pred_depth']),np.array(q['semantic_prob']),np.array(q['depth_prob']),np.array(q['room_prob']),np.ones(40)) for q in records]
    else:
        model=ObservationModel();observations=[];records=[]
        for path in manifest:
            ob=model.predict(Path(path)/'rgb.png');observations.append(ob)
            records.append({'sample':path,'pred_depth':ob.depth.tolist(),'semantic_prob':ob.semantic_probability.tolist(),'depth_prob':ob.depth_probability.tolist(),'room_prob':ob.room_probability.tolist(),'network_ms':ob.latency_ms,'peak_allocated_mib':ob.peak_allocated_mib,'peak_reserved_mib':ob.peak_reserved_mib})
        write_json(out/'perception.json',records)
    rows=[];gt_all=[];pred_all=[];prob_all=[];perception=[]
    for index,(path,obs) in enumerate(zip(manifest,observations)):
        meta,_=read_sample(path);gd=np.array(meta['depth_axial_m']);gs=np.array(meta['semantic']);gt=np.array(meta['pose']);details={'sample':path,'ground_truth_pose':gt.tolist(),'ground_truth_depth_axial':gd.tolist(),'ground_truth_depth_radial':axial_to_radial(gd).tolist(),'ground_truth_semantic':gs.tolist(),'predicted_depth':obs.depth.tolist(),'predicted_semantic_probability':obs.semantic_probability.tolist(),'profiles':{}};heats={}
        labels=obs.labels();gt_all.extend(gs);pred_all.extend(labels);prob_all.extend(obs.semantic_probability)
        perception.append({'sample':path,'all_wall':bool(np.all(labels==0)),'gt_door_or_window_count':int(((gs==1)|(gs==2)).sum()),'mean_wall_probability':float(obs.semantic_probability[:,0].mean()),'depth_mae_if_axial_m':float(abs(obs.depth-gd).mean()),'depth_mae_if_radial_m':float(abs(obs.depth-axial_to_radial(gd)).mean())})
        for profile in profiles:
            matcher=matchers[profile['name']];prediction=obs.labels(profile['semantic_prediction'],profile['seed']+index);details['profiles'][profile['name']]={}
            for condition,d,s in [('A_gt_gt',gd,gs),('B_pred_gt',obs.depth,gs),('C_gt_pred',gd,prediction),('D_pred_pred',obs.depth,prediction)]+([('D_pred_pred_room_aware',obs.depth,prediction)] if profile['room_aware'] else []):
                pd,ps,valid=matcher.volumes(d,s,room_probability=obs.room_probability if condition.endswith('_room_aware') else None)
                details['profiles'][profile['name']][condition]={}
                for fusion in ['depth','semantic','combined']:
                    start=time.perf_counter();result,score=matcher.run_from_volumes(d,s,pd,ps,valid,fusion)
                    cxy,cyaw=pose_error(result['coarse_pose'],gt);fxy,fyaw=pose_error(result['estimated_pose'],gt)
                    row={'sample_index':index,'sample':path,'profile':profile['name'],'condition':condition,'fusion':fusion,'coarse_xy_m':cxy,'coarse_yaw_deg':cyaw,'fine_xy_m':fxy,'fine_yaw_deg':fyaw,'raw_score':result['raw_score'],'fine_cost':result['fine_cost'],'grid_entropy':result['normalized_grid_entropy'],'max_grid_mass':result['normalized_grid_mass_max'],'candidate_and_fine_ms':1000*(time.perf_counter()-start)};rows.append(row)
                    details['profiles'][profile['name']][condition][fusion]=result
                    # Retain both max score and marginal discrete mass, which are different quantities.
                    heats[f"{profile['name']}_{condition}_{fusion}_maxscore"]=score.max(-1).astype('float32')
                    heats[f"{profile['name']}_{condition}_{fusion}_marginal"]=(score/score.sum()).sum(-1).astype('float32')
        write_json(out/'per_image'/f'{index:03d}.json',details);np.savez_compressed(out/'per_image'/f'{index:03d}_heatmaps.npz',**heats)
        print(f'completed {index+1}/{len(manifest)}',flush=True)
    with (out/'per_image_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'scope':'engineering diagnostic only; one public building; no independent test or claim of paper reproduction','n_images':len(manifest),'building_count':1,'cases':{},'perception':classification_metrics(gt_all,pred_all,prob_all),'all_wall_images':sum(q['all_wall'] for q in perception),'mean_depth_mae_axial_m':float(np.mean([q['depth_mae_if_axial_m'] for q in perception])),'mean_depth_mae_radial_m':float(np.mean([q['depth_mae_if_radial_m'] for q in perception])),'training_performed':False,'diagnostic_room_prior':'disabled for A-D to isolate ray perception; separate D_pred_pred_room_aware restores full configured prior'}
    for profile in profiles:
        for condition in ['A_gt_gt','B_pred_gt','C_gt_pred','D_pred_pred']+(['D_pred_pred_room_aware'] if profile['room_aware'] else []):
            for fusion in ['depth','semantic','combined']:
                v=[r for r in rows if r['profile']==profile['name'] and r['condition']==condition and r['fusion']==fusion]
                summary['cases'][f"{profile['name']}/{condition}/{fusion}"]={'coarse':localization_metrics(v,'coarse'),'fine':localization_metrics(v,'fine')}
        v=[r for r in rows if r['profile']==profile['name'] and r['condition']=='D_pred_pred' and r['fusion']=='combined']
        summary.setdefault('raw_score_misuse_audit',{})[profile['name']]=reliability([r['raw_score'] for r in v],[r['coarse_xy_m']<1 and r['coarse_yaw_deg']<30 for r in v])
    summary['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU','seed':42,'code_commit_at_launch':subprocess.check_output(['git','-C',str(BASE),'rev-parse','HEAD'],text=True).strip(),'working_tree_dirty_at_launch':bool(subprocess.check_output(['git','-C',str(BASE),'status','--porcelain'],text=True).strip()),'upstream_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),'map_sha256':floor.hash,'dataset_manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()}
    write_json(out/'summary.json',summary);write_json(out/'perception_errors.json',perception)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['cases','raw_score_misuse_audit']},indent=2))
if __name__=='__main__':main()
