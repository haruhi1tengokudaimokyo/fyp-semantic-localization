"""Receive a key through a private kernel FIFO, then keep it in process memory only."""
import argparse,json,os,runpy,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--key-fifo',required=True);a=p.parse_args();fifo=Path(a.key_fifo)
assert fifo.parent==Path(f'/run/user/{os.getuid()}') and fifo.is_fifo()
with fifo.open() as f:key=f.readline().strip()
fifo.unlink();assert key
net=Path('/home/csy/Documents/Codex/2026-10-31/new-chat/work/fyp/experiment_automation/state/network_environment.json')
if net.exists():os.environ.update(json.loads(net.read_text()))
os.environ['DASHSCOPE_API_KEY']=key;os.environ['DASHSCOPE_BASE_URL']='https://dashscope.aliyuncs.com/compatible-mode/v1';os.environ['FYP_VLG_SCOPE']='official_full_v1'
out=Path('/data/fyp_3d/runs/semantic_localization_20261102/vlg_loc');deadline=time.monotonic()+3600
while not (out/'fixed_full_manifest.json').exists():
 state=subprocess.check_output(['systemctl','--user','show','fyp-vlg-official-data.service','-p','ActiveState','--value'],text=True).strip()
 if state in ['failed','inactive'] or time.monotonic()>deadline:raise RuntimeError('Official complete input data not ready; no new API requests sent')
 time.sleep(30)
runpy.run_path(str(Path(__file__).with_name('run_vlg_pilot.py')),run_name='__main__')
