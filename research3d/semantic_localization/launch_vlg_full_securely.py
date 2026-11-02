"""Call from the already authenticated in-memory session; no secret files or CLI args."""
import errno,json,os,subprocess,time,uuid
from pathlib import Path
key=os.environ['DASHSCOPE_API_KEY'];root=Path('/data/fyp_3d/runs/semantic_localization_20261102')
pilot=json.loads((root/'vlg_loc/metrics_30_schema_normalized.json').read_text());assert pilot['trials']==pilot['poses']==30
fifo=Path(f'/run/user/{os.getuid()}')/('fyp-vlg-key-'+uuid.uuid4().hex+'.fifo');os.mkfifo(fifo,0o600)
args=['systemd-run','--user','--unit=fyp-vlg-plus-official-full','--property=MemoryMax=4G','--property=MemorySwapMax=512M','--property=CPUQuota=200%','--property=RuntimeMaxSec=14400','--property=StandardOutput=append:'+str(root/'vlg_official_full.log'),'--property=StandardError=append:'+str(root/'vlg_official_full.log'),'/data/fyp_3d/envs/semantic_loc/bin/python',str(Path(__file__).with_name('run_vlg_full_worker.py')),'--key-fifo',str(fifo)]
env={k:v for k,v in os.environ.items() if k!='DASHSCOPE_API_KEY'}
try:
 subprocess.run(args,check=True,env=env)
 deadline=time.monotonic()+15
 while True:
  try:fd=os.open(fifo,os.O_WRONLY|os.O_NONBLOCK);break
  except OSError as exc:
   if exc.errno!=errno.ENXIO or time.monotonic()>deadline:raise
   time.sleep(.1)
 with os.fdopen(fd,'w') as f:f.write(key+'\n')
 print('SECURE_FORMAL_WORKER_STARTED; key transmitted via private kernel FIFO only',flush=True)
except Exception:
 if fifo.exists():fifo.unlink()
 raise
