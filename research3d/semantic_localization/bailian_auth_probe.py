"""Read a secret from environment/hidden terminal input, never save it."""
import getpass,json,os
from pathlib import Path
import requests
p=Path('/home/csy/Documents/Codex/2026-10-31/new-chat/work/fyp/experiment_automation/state/network_environment.json')
if p.exists():os.environ.update(json.loads(p.read_text()))
key=os.environ.get('DASHSCOPE_API_KEY') or getpass.getpass('DASHSCOPE_API_KEY (hidden): ')
records=[]
for base in ['https://dashscope.aliyuncs.com/compatible-mode/v1','https://dashscope-intl.aliyuncs.com/compatible-mode/v1']:
 try:
  r=requests.get(base+'/models',headers={'Authorization':'Bearer '+key},timeout=(15,30))
  try:d=r.json()
  except ValueError:d={}
  rec=dict(base_url=base,status=r.status_code,error_code=d.get('error',{}).get('code') if isinstance(d.get('error'),dict) else d.get('code'),models=[v['id'] for v in d.get('data',[]) if 'qwen3-vl' in v.get('id','')])
 except requests.RequestException as e:rec=dict(base_url=base,error_type=type(e).__name__)
 records.append(rec);print(json.dumps(rec),flush=True)
 if rec.get('status')==200:break
out=Path('/data/fyp_3d/runs/semantic_localization_20261102/vlg_loc/auth_probe.json');out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(records,indent=2))
# Secret only remains in process memory for an explicitly requested runner command.
if any(r.get('status')==200 for r in records):
 os.environ['DASHSCOPE_API_KEY']=key
 os.environ['DASHSCOPE_BASE_URL']=records[-1]['base_url']
 print('AUTH_OK_READY_FOR_COMMAND',flush=True)
 import runpy
 while True:
  command=input('Runner path (empty exits): ').strip()
  if not command:break
  try:runpy.run_path(command,run_name='__main__')
  except Exception as exc:print('RUNNER_FAILED',type(exc).__name__,flush=True)
