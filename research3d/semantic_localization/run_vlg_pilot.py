"""30 fixed real queries, original million-particle VLG-Loc + Qwen API.
No scan, GT, true pose, or query visibility labels enter the online localizer.
"""
import os,sys,json,time,dataclasses,subprocess
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_PLATFORM_NAME']='cpu';os.environ['MPLBACKEND']='Agg'
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['XLA_FLAGS']='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
SRC=Path('/data/fyp_3d/sources/VLG-Loc');os.environ['ROS_WORKSPACE']=str(SRC);sys.path.insert(0,str(SRC/'lib'))
HERE=Path('/home/csy/Documents/Codex/2026-10-31/new-chat/work/fyp/research3d/semantic_localization');sys.path.insert(0,str(HERE))
import jax,numpy as np,yaml,hydra
from PIL import Image
from hydra import initialize_config_dir,compose
import importlib,bailian_adapter
importlib.reload(bailian_adapter)
from bailian_adapter import BailianVLM,OUT,atomic
import vlg_loc.recognition.vlm_recognizer as recognizer
recognizer.AzureOpenAIVLM=BailianVLM
from vlg_loc.common.io_format import CameraImages,Pose
from vlg_loc.map.labeled_footprint_map import LabeledFootprintMap
from vlg_loc.localization.vision_localizer import VisionLocalizer
from vlg_loc.loc_evaluator import LocEvaluator
from vlg_loc.common import util
DATA=Path('/data/fyp_3d/datasets/vlg_loc_official')
def main():
 full=os.environ.get('FYP_VLG_SCOPE')=='official_full_v1'
 manifest=json.loads((OUT/('fixed_full_manifest.json' if full else 'fixed_30_manifest.json')).read_text());results=[];localizers={};samplings={}
 protocol=dict(method='Original VLG-Loc vision localizer, Qwen cross-model adaptation',particles=1000000,seed=util.DEFAULT_RANDOM_SEED,source_commit=subprocess.check_output(['git','-C',str(SRC),'rev-parse','HEAD'],text=True).strip(),changes=['Only VLM client replaced','Official prompt placeholder typo corrected (#OBJECT_LIST# vs replace_word)','jax.vmap -> exact chunked jax.lax.map, 1024 particles per chunk; all million retained'],GT='offline only after estimate; no pointcloud input',API_budget_rmb=10,model='qwen3-vl-plus',flash='Unavailable in authenticated models list; no verified current price, not called',formal_benchmark=False)
 protocol['scope']='official_complete_411' if full else 'fixed_30_pilot'
 protocol['formal_benchmark']=full
 atomic(OUT/('protocol_official_full_v1.json' if full else 'protocol_schema_normalized.json'),protocol)
 for sample in manifest['samples']:
  env=sample['environment'];qid=sample['query'];p=OUT/('results_official_full_v1' if full else 'results_schema_normalized')/env/(qid+'.json');p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():results.append(json.loads(p.read_text()));continue
  previous=OUT/('results_schema_normalized' if full else 'results')/env/(qid+'.json')
  if previous.exists():
   saved=json.loads(previous.read_text())
   if 'translation_error_m' in saved:
    atomic(p,saved);results.append(saved);continue
  if env not in localizers:
   with initialize_config_dir(config_dir=str(SRC/'config'/env),version_base=None):cfg=compose(config_name='loc_eval_'+env)
   cfg.vlm_config.model_name='qwen3-vl-plus';cfg.recognizer_config.prompt=cfg.recognizer_config.prompt.replace('#OBJECT_LIST#',cfg.recognizer_config.replace_word)
   robot=hydra.utils.instantiate(cfg.robot_config);vc=hydra.utils.instantiate(cfg.vision_localizer_config);vm=hydra.utils.instantiate(cfg.vlm_config);rc=hydra.utils.instantiate(cfg.recognizer_config);mc=hydra.utils.instantiate(cfg.labeled_footprint_map_config)
   lm=LabeledFootprintMap.from_config(mc)
   localizers[env]=VisionLocalizer(robot,vc,vm,rc,mc,lm)
   samplings[env]=LocEvaluator.sample_poses(None,num_of_poses=vc.num_of_particles,random_seed=util.DEFAULT_RANDOM_SEED,min_x=mc.min_x,max_x=mc.max_x,min_y=mc.min_y,max_y=mc.max_y,min_theta=-np.pi,max_theta=np.pi)
   atomic(OUT/(env+'.map_protocol.json'),dict(map_name=mc.map_name,bounds=[mc.min_x,mc.max_x,mc.min_y,mc.max_y],landmark_names=list(lm.lookup_table_name2id),particles=vc.num_of_particles))
  query=DATA/env/qid
  images=CameraImages(*[Image.open(DATA/f['path']).convert('RGB') for f in sample['images']])
  if not (OUT/'batch_equivalence.json').exists():
   import vlg_loc.localization.vision_localizer as vm_module
   namespace=dict(vm_module.__dict__)
   source=subprocess.check_output(['git','-C',str(SRC),'show','HEAD:lib/vlg_loc/localization/vision_localizer.py'],text=True)
   exec(compile(source,'official_unmodified_vision_localizer.py','exec'),namespace)
   poses=jax.tree_util.tree_map(lambda x:x[:2048],samplings[env])
   original=namespace['VisionLocalizer'].estimate_global_pose(localizers[env],images,poses)
   bounded=localizers[env].estimate_global_pose(images,poses)
   difference=float(np.max(np.abs(np.array(original[1])-np.array(bounded[1]))))
   assert difference<1e-6
   assert np.allclose(np.array(original[0]),np.array(bounded[0]),atol=1e-6)
   atomic(OUT/'batch_equivalence.json',dict(actual_query=qid,particles=2048,maximum_log_likelihood_difference=difference,original_pose=list(map(float,original[0])),bounded_pose=list(map(float,bounded[0])),status='PASS'))
  start=time.perf_counter()
  try:
   est,ll,report=localizers[env].estimate_global_pose(images,samplings[env]);jax.block_until_ready(ll);elapsed=time.perf_counter()-start
  except Exception as exc:
   from pydantic import ValidationError
   import requests
   if isinstance(exc,bailian_adapter.BudgetExhausted):
    atomic(OUT/'API_BUDGET_PAUSED.json',dict(status='PAUSED_BUDGET',completed_trials=len(results),planned_trials=len(manifest['samples']),new_requests_sent_after_pause=False));break
   if not isinstance(exc,(ValidationError,requests.RequestException)) and not (isinstance(exc,RuntimeError) and str(exc).startswith('Bailian HTTP')):raise
   rec=dict(environment=env,query=qid,status='FAILED_VLM_SCHEMA' if isinstance(exc,ValidationError) else 'FAILED_API_REQUEST',error_type=type(exc).__name__,seconds=time.perf_counter()-start,success_1m30deg=False,pose_emitted=False)
   atomic(p,rec);results.append(rec);print('VLG_QUERY_FAILURE',env,qid,rec['status'],flush=True);continue
  # GT accessed only after the online algorithm has returned.
  gt=yaml.safe_load((query/'data.yaml').read_text())['ground_truth_pose'];xyz=[float(est.x),float(est.y),float(est.theta)]
  te=util.calc_abs_trans_error(xyz[0],xyz[1],gt['x'],gt['y']);re=util.calc_abs_rot_error(xyz[2],gt['theta'])
  independent_te=float(np.hypot(xyz[0]-gt['x'],xyz[1]-gt['y']));independent_re=float(abs((xyz[2]-gt['theta']+np.pi)%(2*np.pi)-np.pi))
  assert abs(te-independent_te)<1e-9 and abs(re-independent_re)<1e-9
  response=dataclasses.asdict(report) if dataclasses.is_dataclass(report) else report._asdict()
  rec=dict(environment=env,query=qid,estimated_pose=xyz,ground_truth_offline=gt,translation_error_m=te,rotation_error_deg=float(np.rad2deg(re)),success_1m30deg=bool(te<1 and re<np.pi/6),seconds=elapsed,report=response,particles=1000000,landmark_hallucination_rate=None,hallucination_note='No independent per-image landmark annotations released; human audit required, do not invent hallucination rates')
  atomic(p,rec);results.append(rec);print('VLG_ACTUAL_LOCALIZATION',env,qid,'t_m',te,'r_deg',rec['rotation_error_deg'],'seconds',elapsed,flush=True)
  jax.clear_caches() if len(results)%5==0 else None
 valid=[r for r in results if 'translation_error_m' in r]
 atomic(OUT/('metrics_official_full_v1.json' if full else 'metrics_30_schema_normalized.json'),dict(status='OFFICIAL_DATASET_QWEN_ADAPTATION_COMPLETE' if full else 'PILOT_COMPLETE_NOT_FULL_BENCHMARK',parser_change='Only exact valid label lists under properties unwrapped; raw outputs and original failures preserved',trials=len(results),poses=len(valid),failures=len(results)-len(valid),translation_median_m=float(np.median([r['translation_error_m'] for r in valid])),translation_mean_m=float(np.mean([r['translation_error_m'] for r in valid])),translation_std_m=float(np.std([r['translation_error_m'] for r in valid])),translation_rmse_m=float(np.sqrt(np.mean([r['translation_error_m']**2 for r in valid]))),rotation_median_deg=float(np.median([r['rotation_error_deg'] for r in valid])),rotation_mean_rad=float(np.mean([np.deg2rad(r['rotation_error_deg']) for r in valid])),rotation_std_rad=float(np.std([np.deg2rad(r['rotation_error_deg']) for r in valid])),success_1m30deg=float(np.mean([r['success_1m30deg'] for r in results])),error_statistics_condition='Only emitted poses; all failures counted unsuccessful in success denominator',results=results))
if __name__=='__main__':main()
