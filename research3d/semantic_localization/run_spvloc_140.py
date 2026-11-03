"""One authorized full training run followed by frozen held-out evaluation."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/data/fyp_3d/runs/semantic_localization_20261103')
RUN=ROOT/'spvloc_100_20_20_paper_pano'
MANIFEST=Path('/data/fyp_3d/datasets/procthor_spvloc/building_split_manifest_100_20_20_accepted.json')
PY='/data/fyp_3d/envs/semantic_loc/bin/python'

def atomic(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def invoke(name,args):
 suffix='_float_v2' if RUN.name.endswith('_v2') else ''
 with (ROOT/(name+suffix+'.log')).open('a') as log:
  subprocess.run([PY,HERE/'spvloc_experiment.py',*map(str,args)],env=os.environ,stdout=log,stderr=subprocess.STDOUT,check=True)
def main():
 global RUN
 parser=argparse.ArgumentParser();parser.add_argument('--float-targets',action='store_true');options=parser.parse_args()
 assert options.float_targets,'Uncorrected supervision path retired; use --float-targets and the v2 service'
 if options.float_targets:RUN=ROOT/'spvloc_100_20_20_float_targets_v2'
 ROOT.mkdir(exist_ok=True)
 verified=json.loads((ROOT/'pretraining_data_verification.json').read_text())
 assert verified['status']=='PASSED' and verified['manifest_sha256']==hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
 # No resource eviction and no concurrent GPU job launch.
 occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 assert not occupied, 'GPU occupied; protected experiment will not be stopped'
 common=['--dataset-manifest',str(MANIFEST),'--paper-pano','--output-root',str(ROOT),'--run-name',RUN.name,'--batch','4','--accumulate','10']
 if options.float_targets:
  checks=json.loads((ROOT/'float_target_real_data_checks.json').read_text());assert checks['status']=='PASSED' and checks['scenes']==[0,1,3000,3250]
  common+=['--float-targets']
 assert not (RUN/'training_protocol.json').exists(),'Existing training run must not be silently retrained'
 atomic(ROOT/'pipeline_status.json',dict(stage='TRAINING',started_unix=time.time(),run=str(RUN)))
 invoke('train_140',['train','--steps','-1',*common])
 evidence=json.loads((RUN/'training_evidence.json').read_text())
 assert evidence['epochs_completed']==25 and evidence['updated_parameter_count']>0
 best=evidence['best_checkpoint_by_validation_loss'];assert Path(best).is_file()
 atomic(ROOT/'pipeline_status.json',dict(stage='HELD_OUT_EVALUATION',checkpoint=best))
 invoke('evaluate_140_finetuned',['evaluate','--checkpoint',best,'--label','finetuned',*common])
 # Original pretrained baseline uses the identical held-out query population;
 # neither test result can change training or the validation-selected checkpoint.
 invoke('evaluate_140_official',['evaluate','--label','official_zero_shot',*common])
 for label in ['finetuned','official_zero_shot']:
  metrics=json.loads((RUN/f'metrics_{label}.json').read_text())
  assert len(metrics['split']['test'])==20 and metrics['queries']>0
 atomic(ROOT/'pipeline_status.json',dict(stage='COMPLETE',run=str(RUN),finished_unix=time.time(),test_used_for_checkpoint_selection=False))
if __name__=='__main__':
 os.environ.update(LD_LIBRARY_PATH='/data/fyp_3d/sources/spvloc_redner/redner-dependencies/embree/lib-linux',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYOPENGL_PLATFORM='egl')
 try:main()
 except BaseException as exc:
  atomic(ROOT/'pipeline_status.json',dict(stage='FAILED',error=repr(exc)));raise
