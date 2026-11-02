"""One existing-computer experiment pipeline; not a scheduler or monitor."""
import json,os,subprocess,time,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;PY='/data/fyp_3d/envs/semantic_loc/bin/python';OUT=Path('/data/fyp_3d/runs/semantic_localization_20261102')
os.environ.update(LD_LIBRARY_PATH='/data/fyp_3d/sources/spvloc_redner/redner-dependencies/embree/lib-linux',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
def run(name,args):
 with (OUT/(name+'.log')).open('a') as log:
  return subprocess.run([PY,*map(str,args)],env=os.environ,stdout=log,stderr=subprocess.STDOUT).returncode
def main():
 deadline=time.monotonic()+3600
 while True:
  states=[subprocess.check_output(['systemctl','--user','show',u,'-p','ActiveState','--value'],text=True).strip() for u in ['fyp-procthor-capture-16.service','fyp-procthor-augment-pilot.service']]
  if not any(s in ['active','activating'] for s in states):break
  if time.monotonic()>deadline:raise RuntimeError('Finite capture dependency deadline reached')
  time.sleep(30)
 assert run('procthor_qualification_16',[HERE/'qualify_procthor.py'])==0
 args=[HERE/'spvloc_experiment.py','evaluate','--full-split','--run-name','spvloc_10_3_3','--label','official_zero_shot']
 assert run('spvloc_16_zero_shot',args)==0
 train=[HERE/'spvloc_experiment.py','train','--steps','-1','--full-split','--run-name','spvloc_10_3_3','--batch','8','--accumulate','5']
 code=run('spvloc_16_train',train)
 if code:
  text=(OUT/'spvloc_16_train.log').read_text()
  if 'CUDA out of memory' not in text:raise RuntimeError('Training failed; scientific code/data errors are not blindly retried')
  # One declared hardware fallback only, never a data/accuracy threshold change.
  train=[HERE/'spvloc_experiment.py','train','--steps','-1','--full-split','--run-name','spvloc_10_3_3_batch4','--batch','4','--accumulate','10']
  assert run('spvloc_16_train_batch4',train)==0;run_name='spvloc_10_3_3_batch4'
 else:run_name='spvloc_10_3_3'
 evidence=json.loads((OUT/run_name/'training_evidence.json').read_text());best=evidence['best_checkpoint_by_validation_loss']
 assert evidence['optimizer_steps']>0 and evidence['updated_parameter_count']>0
 assert run('spvloc_16_finetuned_eval',[HERE/'spvloc_experiment.py','evaluate','--full-split','--run-name',run_name,'--checkpoint',best,'--label','finetuned'])==0
 print('FULL_SPVLOC_PIPELINE_COMPLETE',run_name,flush=True)
if __name__=='__main__':main()
