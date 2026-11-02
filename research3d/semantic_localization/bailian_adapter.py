"""OpenAI-compatible Qwen adapter for the original VLG-Loc recognizer.
Hard RMB 10 shared budget; uncertain paid requests retain their full reservation.
"""
import fcntl,hashlib,json,os,time,uuid
from pathlib import Path
from decimal import Decimal
import requests
OUT=Path('/data/fyp_3d/runs/semantic_localization_20261102/vlg_loc');OUT.mkdir(parents=True,exist_ok=True)
PRICE=dict(source='https://help.aliyun.com/zh/model-studio/model-pricing',region='cn-beijing',currency='CNY',model='qwen3-vl-plus',input_per_million=['1','1.5','3'],output_per_million=['10','15','30'],input_thresholds=[32000,128000,256000],verified_on='2026-11-02')
MAX_OUTPUT=1024
# Entire model context at its highest tier, even though 480px images use far less.
RESERVE=Decimal(256000)*Decimal(3)/1000000+Decimal(MAX_OUTPUT)*Decimal(30)/1000000
LIMIT=Decimal('10')
class BudgetExhausted(RuntimeError):pass
def parse_response(response,response_format):
 data=json.loads(response)
 try:return response_format.model_validate(data),None
 except Exception:
  # Observed Qwen output put the actual list of recognized labels under
  # "properties". Unwrap only if it already validates as the required data.
  # A JSON schema object with {type:array,items:...} still fails validation.
  if isinstance(data,dict) and isinstance(data.get('properties'),dict):
   return response_format.model_validate(data['properties']),'properties_object_unwrapped_no_label_changes'
  raise
def atomic(p,v):
 t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False));t.replace(p)
def ledger_update(request_id,record=None):
 with (OUT/'budget.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  p=OUT/'api_budget.json';d=json.loads(p.read_text()) if p.exists() else dict(limit_rmb=str(LIMIT),price=PRICE,requests={})
  if record is None:
   used=sum(Decimal(r.get('charged_or_reserved_rmb','0')) for r in d['requests'].values())
   if used+RESERVE>LIMIT:raise BudgetExhausted('RMB10 hard budget exhausted; no request sent')
   d['requests'][request_id]=dict(status='RESERVED',charged_or_reserved_rmb=str(RESERVE))
  else:d['requests'][request_id]=record
  atomic(p,d)
class BailianVLM:
 def __init__(self,config):
  assert config.model_name=='qwen3-vl-plus','Unverified model availability/pricing'
  self.model=config.model_name;self.base=os.environ['DASHSCOPE_BASE_URL'];self.key=os.environ['DASHSCOPE_API_KEY']
  assert self.base=='https://dashscope.aliyuncs.com/compatible-mode/v1','Price region mismatch'
  self.cache=OUT/'api_cache';self.cache.mkdir(exist_ok=True)
 def get_formatted_response_with_multiple_images(self,query,data_urls,response_format):
  schema=response_format.model_json_schema()
  format_instruction='\nReturn a JSON object conforming to this schema: '+json.dumps(schema,sort_keys=True)
  content=[dict(type='text',text=query+format_instruction)]+[dict(type='image_url',image_url=dict(url=u)) for u in data_urls]
  payload=dict(model=self.model,temperature=0.,seed=12345,max_tokens=MAX_OUTPUT,enable_thinking=False,messages=[dict(role='user',content=content)],response_format=dict(type='json_object'))
  identity=dict(endpoint=self.base,version_scope='alias_snapshot_2026-11-02',payload=payload)
  key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();p=self.cache/(key+'.json')
  if p.exists():return response_format.model_validate(json.loads(p.read_text())['parsed'])
  raw_cache=self.cache/(key+'.raw.json')
  if raw_cache.exists():
   raw=json.loads(raw_cache.read_text());parsed,normalization=parse_response(raw['response'],response_format)
   raw['parsed']=parsed.model_dump();raw['format_normalization']=normalization;atomic(p,raw);return parsed
  req=uuid.uuid4().hex;ledger_update(req);start=time.time()
  try:
   r=requests.post(self.base+'/chat/completions',headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'},json=payload,timeout=(20,120))
   if r.status_code>=400:
    if r.status_code<500:ledger_update(req,dict(status='HTTP_REJECTED',http_status=r.status_code,charged_or_reserved_rmb='0'))
    raise RuntimeError(f'Bailian HTTP {r.status_code}; no automatic retry')
   d=r.json();usage=d.get('usage');assert usage is not None,'Usage missing; reservation retained'
   n=int(usage['prompt_tokens']);out=int(usage['completion_tokens']);tier=0 if n<=32000 else 1 if n<=128000 else 2
   assert n<=256000 and out<=MAX_OUTPUT
   cost=(Decimal(n)*Decimal(PRICE['input_per_million'][tier])+Decimal(out)*Decimal(PRICE['output_per_million'][tier]))/1000000
   assert cost<=RESERVE
   elapsed=time.time()-start;response=d['choices'][0]['message']['content']
   ledger_update(req,dict(status='BILLED_RESPONSE',charged_or_reserved_rmb=str(cost),usage=usage,seconds=elapsed,cache_hash=key,reported_model=d.get('model'),system_fingerprint=d.get('system_fingerprint')))
   # Cache raw paid response before parsing, so invalid JSON is never re-purchased.
   rec=dict(request_id=req,model=d.get('model'),system_fingerprint=d.get('system_fingerprint'),image_sha256=[hashlib.sha256(u.encode()).hexdigest() for u in data_urls],original_prompt=query,response=response,usage=usage,cost_rmb=str(cost),seconds=elapsed,price=PRICE)
   atomic(self.cache/(key+'.raw.json'),rec)
   parsed,normalization=parse_response(response,response_format);rec['parsed']=parsed.model_dump();rec['format_normalization']=normalization;atomic(p,rec)
   return parsed
  except Exception:
   # No secret or HTTP body in exception/logs. Uncertain reservations remain.
   raise
