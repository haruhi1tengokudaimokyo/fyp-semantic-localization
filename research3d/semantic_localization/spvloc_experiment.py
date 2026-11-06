"""Run unmodified SPVLoc network/loss/localization on qualified ProcTHOR houses.
Only split selection, hardware batch size, and data/record adapters are changed.
"""
import argparse,os,sys,json,time,hashlib,gc,random,subprocess
from pathlib import Path
os.environ['PYOPENGL_PLATFORM']='egl';os.environ['TORCH_HOME']='/data/fyp_3d/models/torchvision'
sys.path.insert(0,'/data/fyp_3d/sources/spvloc')
os.chdir('/data/fyp_3d/sources/spvloc')  # Upstream custom shader paths are relative.
import numpy as np,torch,pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback,ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from spvloc.config.defaults import get_cfg_defaults
import spvloc.data.dataset as ds
import spvloc.data.data_inference as di
from spvloc.model.spvloc import PerspectiveImageFromLayout
from spvloc.data.dataset import load_scene_data
ROOT=Path('/data/fyp_3d/datasets/procthor_spvloc');OUT=Path('/data/fyp_3d/runs/semantic_localization_20261102/spvloc_pilot');OUT.mkdir(parents=True,exist_ok=True)
SPLIT={'train':[0],'val':[3000],'test':[3250]}
def atomic(p,d):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def config():
 c=get_cfg_defaults();c.merge_from_file('/data/fyp_3d/sources/spvloc/configs/config_s3d.yaml')
 c.DATASET.PATH=str(ROOT);c.DATASET.PREPARED_DATA_PATH='pickle_v2_openplan';c.DATASET.PREPARED_DATA_PATH_TEST='pickle_test_v2_openplan';c.DATASET.PERSP_FROM_PANO=False;c.DATASET.TRAIN_FURNITURE=['full'];c.DATASET.TEST_SET_FURNITURE=['full'];c.DATASET.CACHE_TEST_BATCHES=True
 if ARGS.full_split:c.DATASET.PREPARED_DATA_PATH='pickle_v3_full160';c.DATASET.PREPARED_DATA_PATH_TEST='pickle_test_v3_full160'
 if ARGS.paper_pano:
  c.DATASET.PERSP_FROM_PANO=True
  # Upstream projection.py explicitly documents False as exact paper roll behavior.
  c.DATASET.PERSP_FROM_PANO_CORRECT_ROLL=False
  c.DATASET.PREPARED_DATA_PATH='pickle_v4_paper_pano';c.DATASET.PREPARED_DATA_PATH_TEST='pickle_test_v4_paper_pano'
 c.SYSTEM.NUM_WORKERS=0;c.SYSTEM.NUM_GPUS=1;c.TRAIN.BATCH_SIZE=ARGS.batch;c.TEST.IGNORE_PRETRAINED=True;c.TEST.EVAL_GT_MAP_POINT=False;c.TEST.PLOT_OUTPUT=False;c.TEST.SAVE_PLOT_OUTPUT=False;c.TEST.SAVE_PLOT_DETAILS=False;c.POSE_REFINE.MAX_ITERS=1;c.OUT_DIR=str(OUT)
 c.TRAIN.NUM_EPOCHS=25
 ds.scenes_split=lambda split,*a,**kw:SPLIT[split]
 di.scenes_split=lambda split,*a,**kw:SPLIT[split]
 for ids in SPLIT.values():
  for sid in ids:assert json.loads((ROOT/f'scene_{sid:05d}'/'qualification.json').read_text())['status']=='DATA_READY_PILOT'
 if ARGS.paper_pano:
  for ids in SPLIT.values():
   for sid in ids:assert json.loads((ROOT/f'scene_{sid:05d}'/'panorama_qualification.json').read_text())['status']=='PANORAMA_READY'
 return c
def load_model(c,path):
 m=PerspectiveImageFromLayout(c);saved=torch.load(path,map_location='cpu');m.load_state_dict(saved['state_dict'],strict=True);del saved;gc.collect();return m
class Evidence(Callback):
 def __init__(self):self.started=None;self.feasibility_saved=False;self.last_progress_step=0;self.valid_samples=0;self.total_samples=0
 def on_train_batch_start(self,trainer,module,batch,batch_idx):
  if self.started is None:self.started=time.time();torch.cuda.reset_peak_memory_stats()
  if ARGS.float_targets:
   h=batch['reference_histogram'];valid=(h[:,0]<.05)&((h[:,1:]>.002).sum(-1)>2)&(h[:,4]<.7)&(h[:,5]<.7)
   self.valid_samples+=int(valid.sum());self.total_samples+=len(h)
   if not (OUT/'real_batch_target_contract.json').exists():
    sem=batch['persp_semantics'];depth=batch['persp_layout'];normal=batch['persp_normal']
    assert torch.isfinite(sem).all() and torch.equal(sem,sem.round()) and sem.min()>=0 and sem.max()<=5
    assert torch.isfinite(depth).all() and depth.min()>=0 and depth.max()>0
    assert torch.isfinite(normal).all() and normal.abs().max()<=1.00001
    atomic(OUT/'real_batch_target_contract.json',dict(status='PASSED',semantic_classes=sem.unique().tolist(),depth_range_m=[float(depth.min()),float(depth.max())],normal_range=[float(normal.min()),float(normal.max())],histogram_sums=h.sum(-1).tolist()))
 def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
  if trainer.global_step==1 and not self.feasibility_saved:
   self.feasibility_saved=True
   torch.cuda.synchronize();atomic(OUT/'batch_feasibility.json',dict(real_optimizer_step=1,precision='FP32',query_batch_size=module.config.TRAIN.BATCH_SIZE,gradient_accumulation=ARGS.accumulate,nominal_effective_batch=module.config.TRAIN.BATCH_SIZE*ARGS.accumulate,near_references=4,far_references=1,query_resolution=[256,256],pano_resolution=[128,256],cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),first_update_seconds=time.time()-self.started,loss=float(outputs['loss'])))
  step=trainer.global_step
  if step>self.last_progress_step:
   self.last_progress_step=step
   elapsed=time.time()-self.started
   atomic(OUT/'training_progress.json',dict(status='TRAINING',optimizer_step=step,epoch=trainer.current_epoch,total_optimizer_steps=int(trainer.estimated_stepping_batches),elapsed_training_seconds=elapsed,mean_seconds_per_optimizer_step=elapsed/step,estimated_remaining_train_seconds=elapsed/step*(trainer.estimated_stepping_batches-step),validation_timing_included=trainer.current_epoch>0,original_filter_valid_fraction=self.valid_samples/self.total_samples if self.total_samples else None,unix_time=time.time()))
   if step%10==0:print('REAL_SPVLOC_UPDATE',step,float(outputs['loss']),flush=True)

def train(c,steps):
 pl.seed_everything(20261102,workers=True);m=load_model(c,'/data/fyp_3d/models/spvloc/ckpt_s3d.ckpt')
 before={n:p.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad}
 atomic(OUT/'training_protocol.json',dict(status='RUNNING',initialization='Official ckpt_s3d.ckpt, strict=True all state keys; not the previous pilot checkpoint',source_weight_sha256=json.loads(Path('/data/fyp_3d/models/spvloc/source.json').read_text())['sha256'],split=SPLIT,config=c.dump(),requested_steps=steps,precision='32-true',actual_algorithm='Original PerspectiveImageFromLayout.common_step and configure_optimizers: all original losses, heads, Adam, scheduler',dataset_difference='ProcTHOR furnished houses; original architectural target classes; open-plan connected rooms merged from source topology',query_sampling='Original random panorama crops, FoV 45-135, yaw/pitch/roll and paper CORRECT_ROLL=False' if ARGS.paper_pano else 'Genuine fixed native perspective queries',eligible_scope='100/20/20 ProcTHOR adaptation, not original Structured3D benchmark' if ARGS.dataset_manifest else 'Small-scale ProcTHOR adaptation'))
 checkpoint=ModelCheckpoint(dirpath=OUT/'checkpoints',monitor='validation/loss',mode='min',save_top_k=1,save_last=True,filename='epoch{epoch}-step{step}')
 recovery=ModelCheckpoint(dirpath=OUT/'recovery',monitor=None,save_top_k=1,every_n_train_steps=100,filename='step{step}')
 trainer=pl.Trainer(accelerator='gpu',devices=1,precision='32-true',max_steps=steps,max_epochs=25,accumulate_grad_batches=ARGS.accumulate,num_sanity_val_steps=0,logger=CSVLogger(str(OUT),name='loss_logs'),callbacks=[checkpoint,recovery,Evidence()],enable_progress_bar=False,log_every_n_steps=1,check_val_every_n_epoch=1)
 trainer.fit(m)
 if not checkpoint.best_model_path:
  trainer.validate(m);trainer.save_checkpoint(str(OUT/'checkpoints'/'last.ckpt'))
 changed=[]
 for n,p in m.named_parameters():
  if n in before:
   delta=float((p.detach().cpu()-before[n]).double().norm())
   if delta:changed.append(dict(name=n,numel=p.numel(),l2_update=delta))
 assert trainer.global_step>0 and changed
 best=checkpoint.best_model_path or str(OUT/'checkpoints'/'last.ckpt')
 atomic(OUT/'training_evidence.json',dict(status='TRAINED_EVALUATION_PENDING',optimizer_steps=trainer.global_step,epochs_completed=trainer.current_epoch,trainable_parameter_count=sum(v.numel() for v in before.values()),updated_parameter_count=sum(x['numel'] for x in changed),updated_tensors=changed,best_checkpoint_by_validation_loss=best,best_validation_loss=float(checkpoint.best_model_score) if checkpoint.best_model_score is not None else None,test_used_for_checkpoint_selection=False))
 print('TRAINING_COMPLETE',trainer.global_step,best,flush=True)

def evaluate(c,path,label):
 pl.seed_everything(20261102,workers=True);c.TRAIN.NUM_NEAR_SAMPLES=1;c.TRAIN.NUM_FAR_SAMPLES=0;c.TRAIN.FILTER_PANO_SAMPLES=False;c.RENDER.FIX_SEED=True
 model=load_model(c,path).cuda().eval();data=di.Structured3DPlans_Inference(c,'test',visualise=True);results=[];expected={}
 for sceneindex in range(len(data)):
  batch=data[sceneindex];sid=int(batch['scene_id']);source=load_scene_data([sid],str(ROOT),c.DATASET.PREPARED_DATA_PATH,True,for_visualisation=True,config=c)[0]
  expected[sid]=len(batch['image']);assert expected[sid]>0
  atomic(OUT/f'offline_evaluation_gt_{sid:05d}.json',dict(scene_id=sid,poses=batch['gt_poses'].numpy().tolist(),units='mm',online_input=False))
  clean_map={k:source[k] for k in ['geometry','geometry_clipped','materials','materials_clipped','floor_planes','limits']}
  refs={k:v for k,v in batch.items() if k not in ['gt_poses','gt_map_points','image','cams','encoded_data','encoding_length']}
  # Every query retained; batching alone changes to fit 6GB. All reference poses retained.
  for i in range(len(batch['image'])):
   p=OUT/f'eval_{label}_{sid:05d}_{i:05d}.json'
   if p.exists():results.append(json.loads(p.read_text()));continue
   online={**refs,'image':batch['image'][i:i+1],'cams':batch['cams'][i:i+1]}
   with torch.inference_mode():
    start=time.time();pred,_=model.test_step_full_epoch(online,refine_iterations=1,top_n=1,prepare_plot=False,scene=clean_map);torch.cuda.synchronize();seconds=time.time()-start
   gt=batch['gt_poses'][i].numpy();r={}
   for key in ['initial_estimates','refined_estimates']:
    est=pred[key][0,0].numpy();te=float(np.linalg.norm(est[:3,3]-gt[:3,3])/1000)
    # SPVLoc stores camera reflections. Relative orientation is a proper rotation.
    cos=(np.trace(est[:3,:3]@gt[:3,:3].T)-1)/2;re=float(np.rad2deg(np.arccos(np.clip(cos,-1,1))))
    r[key]=dict(translation_error_m=te,rotation_error_deg=re,pose=est.tolist(),success_1m30deg=bool(te<1 and re<30))
   # Depth is a length-one placeholder when PANO_ENCODE_DEPTH=False.
   # The actual reference population is the semantic panorama batch.
   rec=dict(scene_id=sid,query_index=i,seconds=seconds,estimates=r,GT_offline_only=True,reference_count=len(batch['sampled_semantics']),checkpoint=path);atomic(p,rec);results.append(rec)
   print('REAL_SPVLOC_GLOBAL_POSE',label,sid,i,r['initial_estimates']['translation_error_m'],r['initial_estimates']['rotation_error_deg'],flush=True)
 metrics={}
 for key in ['initial_estimates','refined_estimates']:
  rows=[r['estimates'][key] for r in results]
  metrics[key]=dict(samples=len(rows),translation_median_m=float(np.median([r['translation_error_m'] for r in rows])),rotation_median_deg=float(np.median([r['rotation_error_deg'] for r in rows])),success_1m30deg=float(np.mean([r['success_1m30deg'] for r in rows])))
 assert set(expected)==set(SPLIT['test'])
 assert all(sum(r['scene_id']==sid for r in results)==n for sid,n in expected.items())
 atomic(OUT/f'metrics_{label}.json',dict(status='EVALUATED_PROCTHOR_ADAPTATION_NOT_ORIGINAL_BENCHMARK',split=SPLIT,metrics=metrics,checkpoint=path,queries=len(results),eligible_query_counts_by_scene=expected))
 if ARGS.dataset_manifest:
  subprocess.run([sys.executable,Path(__file__).with_name('verify_spvloc_metrics.py'),OUT,label],check=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','evaluate']);p.add_argument('--steps',type=int,default=100);p.add_argument('--checkpoint',default='/data/fyp_3d/models/spvloc/ckpt_s3d.ckpt');p.add_argument('--label',default='official_zero_shot');p.add_argument('--batch',type=int,default=1);p.add_argument('--accumulate',type=int,default=1);p.add_argument('--run-name',default='spvloc_pilot');p.add_argument('--full-split',action='store_true');p.add_argument('--dataset-manifest',type=Path);p.add_argument('--paper-pano',action='store_true');p.add_argument('--output-root',type=Path);p.add_argument('--float-targets',action='store_true');a=p.parse_args();ARGS=a
 if a.float_targets:
  from float_targets import install
  install()
 if a.mode=='train':assert a.float_targets,'Uncorrected float supervision training has been retired'
 assert a.run_name.replace('_','').isalnum()
 OUT=(a.output_root or OUT.parent)/a.run_name;OUT.mkdir(parents=True,exist_ok=True)
 if a.dataset_manifest:
  manifest=json.loads(a.dataset_manifest.read_text());assert manifest['status']=='DATA_READY' and manifest['accepted_counts']=={'train':100,'val':20,'test':20}
  SPLIT={s:[h['scene_id'] for h in manifest['houses'] if h['split']==s] for s in ['train','val','test']}
  snapshot=dict(accepted_split=SPLIT,manifest_sha256=hashlib.sha256(a.dataset_manifest.read_bytes()).hexdigest(),source_manifest=str(a.dataset_manifest))
  path=OUT/'dataset_subset.json'
  if path.exists():assert json.loads(path.read_text())==snapshot,'Immutable run split mismatch'
  else:atomic(path,snapshot)
 elif a.full_split:
  snapshot=OUT/'dataset_subset.json'
  if snapshot.exists():
   # A scene qualified after training started must not change the recorded
   # training population when the later evaluation process starts.
   SPLIT=json.loads(snapshot.read_text())['accepted_split']
  else:
   manifest=json.loads((ROOT/'building_split_manifest.json').read_text());SPLIT={s:[] for s in ['train','val','test']};excluded=[]
   for h in manifest['houses']:
    q=ROOT/f"scene_{h['scene_id']:05d}"/'qualification.json'
    if q.exists() and json.loads(q.read_text()).get('status')=='DATA_READY_PILOT':SPLIT[h['split']].append(h['scene_id'])
    else:excluded.append(h['scene_id'])
   assert len(SPLIT['train'])>=3 and len(SPLIT['val'])>=1 and len(SPLIT['test'])>=1
   atomic(snapshot,dict(accepted_split=SPLIT,excluded_for_data_quality=excluded,requested_split_counts=[10,3,3],complete_10_3_3=not excluded,selection='Source split and fixed data acceptance only; no localization score used'))
 c=config()
 if a.float_targets:atomic(OUT/'float_target_adapter.json',dict(enabled=True,PIL_mode='F',preserved=['semantic class IDs 0..5','depth metres','signed normal components'],RGB_pipeline_changed=False,original_network_and_loss_unchanged=True))
 if a.mode=='train':train(c,a.steps)
 else:evaluate(c,a.checkpoint,a.label)
