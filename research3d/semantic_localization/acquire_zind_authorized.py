"""Authorized ZInD acquisition. Bridge credential is used in memory only.

Uses the publisher's endpoint, annotation conversion and MD5s. Licensed data
and signed download URLs stay outside Git in a private dataset directory.
No GPU operations, training commands, or recurring monitoring are involved.
"""
import argparse,concurrent.futures,copy,getpass,hashlib,importlib.util,json,os,shutil,sys,time
from pathlib import Path
import requests
import ijson
from PIL import Image

ROOT=Path('/data/fyp_3d/datasets/zind_original')
SOURCE=Path('/data/fyp_3d/sources/zind')
API='https://api.bridgedataoutput.com/api/v2/OData/zgindoor/Indoor/replication'
GiB=1024**3
MIN_FREE=103*GiB  # additional 3GiB protects the existing trainer's checkpoints
MAX_DATA=45*GiB

def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def init():
    ROOT.mkdir(parents=True,exist_ok=True);ROOT.chmod(0o700)
    env=Path('/home/csy/Documents/Codex/2026-10-31/new-chat/work/fyp/experiment_automation/state/network_environment.json')
    if env.exists():os.environ.update(json.loads(env.read_text()))
    os.umask(0o077)
def check_space():
    if shutil.disk_usage(ROOT).free<MIN_FREE:raise RuntimeError('STORAGE_PAUSED: protect >=100GiB plus training reserve')
def iter_houses(path):
    with path.open('rb') as stream:yield from ijson.items(stream,'value.item',use_float=True)
def metadata():
    init();check_space();key=getpass.getpass('Bridge Server Token (hidden): ')
    target=ROOT/'.bridge_metadata.json';last=None
    for attempt in range(3):
        part=target.with_suffix('.part')
        try:
            # Standard OData size parameter, same authenticated publisher endpoint.
            with requests.get(API,headers={'Authorization':'Bearer '+key},params={'$top':2000},stream=True,timeout=(20,120)) as response:
                print('BRIDGE_HTTP',response.status_code,flush=True)
                if response.status_code in [401,403]:raise PermissionError('Bridge denied supplied credential or dataset access')
                if response.status_code!=200:raise RuntimeError('HTTP_'+str(response.status_code))
                size=0
                with part.open('wb') as stream:
                    for chunk in response.iter_content(1024*1024):
                        check_space();size+=len(chunk)
                        if size>2*GiB:raise RuntimeError('Metadata exceeded finite 2GiB cap')
                        stream.write(chunk)
                data=part.read_bytes()
                if key.encode() in data:raise RuntimeError('Credential unexpectedly embedded in response; refusing persistence')
                result=json.loads(data);houses=result.get('value')
                if not isinstance(houses,list) or len(houses)!=1575:raise RuntimeError('Expected complete official 1575-house list; got '+str(len(houses) if isinstance(houses,list) else 'invalid schema'))
                assert len({str(h['home_id']) for h in houses})==len(houses)
                part.replace(target)
                atomic(ROOT/'metadata_receipt.json',dict(status='AUTHORIZED_METADATA_RECEIVED',house_count=len(houses),bytes=size,sha256=hashlib.sha256(data).hexdigest(),endpoint=API,credential_saved=False,source='https://github.com/zillow/zind',license='User-approved Bridge/Zillow data agreement; source-code Apache license is separate',unix_time=time.time()))
                print('AUTHORIZED_FULL_METADATA',len(houses),'houses',size,'bytes',flush=True);return
        except PermissionError:part.unlink(missing_ok=True);raise
        except Exception as exc:
            last=type(exc).__name__+(': '+str(exc) if not isinstance(exc,requests.RequestException) else '')
            print('METADATA_ATTEMPT_FAILED',attempt+1,last,flush=True)
            part.unlink(missing_ok=True)
            if attempt<2:time.sleep(3)
    raise RuntimeError('Finite metadata attempts exhausted: '+str(last))

def digest(path,algorithm):
    result=hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()
def validate_image(path,md5):
    if digest(path,'md5').lower()!=md5.lower():raise ValueError('Publisher MD5 mismatch')
    with Image.open(path) as image:
        image.load();width,height=image.size
        if min(width,height)<=0:raise ValueError('Empty image')
    return dict(md5=md5,sha256=digest(path,'sha256'),bytes=path.stat().st_size,width=width,height=height)
def fetch(job):
    url,name,md5=job;path=Path(name);assert path.resolve().is_relative_to(ROOT.resolve())
    if path.exists():
        try:return path,validate_image(path,md5)
        except (OSError,ValueError):pass
    part=path.with_suffix(path.suffix+'.part');last=None
    for attempt in range(3):
        check_space()
        try:
            offset=part.stat().st_size if part.exists() else 0
            headers={'Range':f'bytes={offset}-'} if offset else {}
            with requests.get(url,headers=headers,stream=True,timeout=(20,60)) as response:
                if response.status_code not in [200,206]:raise ValueError('IMAGE_HTTP_'+str(response.status_code))
                append=bool(offset and response.status_code==206)
                if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('Invalid Range response')
                with part.open('ab' if append else 'wb') as stream:
                    for chunk in response.iter_content(1024*1024):check_space();stream.write(chunk)
            info=validate_image(part,md5);part.replace(path);return path,info
        except RuntimeError:raise
        except Exception as exc:
            last=type(exc).__name__+(': '+str(exc) if isinstance(exc,ValueError) else '')
            if isinstance(exc,ValueError):part.unlink(missing_ok=True)
            if attempt<2:time.sleep(attempt+1)
    raise RuntimeError('File acquisition failed after 3 attempts: '+path.name+' ('+str(last)+')')

def download():
    init();check_space()
    meta=ROOT/'.bridge_metadata.json';receipt=json.loads((ROOT/'metadata_receipt.json').read_text())
    assert digest(meta,'sha256')==receipt['sha256']
    house_count=receipt['house_count'];assert house_count==1575
    partition=json.loads((SOURCE/'zind_partition.json').read_text())
    split_of={str(h):split for split,values in partition.items() for h in values}
    assert len(split_of)==sum(map(len,partition.values()))
    spec=importlib.util.spec_from_file_location('zind_publisher_download',SOURCE/'download_data.py')
    publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)
    # Pilot spans the three publisher splits, then the remainder in source order.
    pilot=[str(partition[split][0]) for split in ['train','val','test']]
    def ordered_houses():
        found={}
        for house in iter_houses(meta):
            home=str(house['home_id'])
            if home in pilot:found[home]=house
            if len(found)==3:break
        assert len(found)==3
        for home in pilot:yield found[home]
        for house in iter_houses(meta):
            if str(house['home_id']) not in pilot:yield house
    houses=ordered_houses()
    accepted=[];failures=[];verified_bytes=meta.stat().st_size;images=0
    status=dict(stage='DOWNLOADING',total_houses=house_count,complete_houses=0,verified_images=0,raw_dataset_only=True)
    atomic(ROOT/'download_status.json',status)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool, (ROOT/'FILES_VERIFIED.jsonl').open('a') as audit:
        for house in houses:
            check_space();home=str(house['home_id']);assert home.isdigit() and len(home)<8 and home in split_of
            used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
            if used>MAX_DATA:raise RuntimeError('STORAGE_PAUSED: 45GiB dataset budget reached')
            try:
                assert 'merger' in house and 'scale_meters_per_coordinate' in house
                jobs=publisher.process_house(copy.deepcopy(house),str(ROOT));assert jobs
            except Exception as exc:
                failures.append(dict(home_id=home,error='Metadata '+type(exc).__name__))
                atomic(ROOT/'metadata_failures.json',failures)
                continue
            results=[];error=None
            futures={pool.submit(fetch,job):job for job in jobs}
            for future in concurrent.futures.as_completed(futures):
                try:
                    path,info=future.result();results.append(dict(path=str(path.relative_to(ROOT)),**info));audit.write(json.dumps(dict(home_id=home,**results[-1]))+'\n')
                except Exception as exc:error=type(exc).__name__+': '+str(exc)
            audit.flush()
            if error:
                failures.append(dict(home_id=home,error=error));atomic(ROOT/home/'download_failure.json',failures[-1])
                if 'STORAGE_PAUSED' in error:raise RuntimeError(error)
            else:
                home_bytes=sum(x['bytes'] for x in results);verified_bytes+=home_bytes;images+=len(results)
                entry=dict(status='DOWNLOAD_VERIFIED_NOT_SPVLOC_CONVERTED',home_id=home,split=split_of[home],image_count=len(results),bytes=home_bytes,files=results)
                atomic(ROOT/home/'download_receipt.json',entry);accepted.append(dict(home_id=home,split=split_of[home],image_count=len(results),bytes=home_bytes))
            status.update(complete_houses=len(accepted),verified_images=images,verified_bytes=verified_bytes,failed_houses=failures,current_home=home,free_GiB=shutil.disk_usage(ROOT).free/GiB,unix_time=time.time())
            status['three_split_pilot_verified']=all(h in {x['home_id'] for x in accepted} for h in pilot)
            atomic(ROOT/'download_status.json',status)
            print('ZIND_DOWNLOAD_PROGRESS',len(accepted),'/1575',images,'verified images',len(failures),'failed houses',flush=True)
    complete=len(accepted)==1575 and not failures
    status['stage']='DOWNLOAD_VERIFIED' if complete else 'INCOMPLETE'
    atomic(ROOT/'DATASET_DOWNLOAD_MANIFEST.json',dict(**status,houses=accepted,source_api=API,official_partition=partition,source_license='Bridge/Zillow user-approved agreement; no licensed raw data redistribution',SPVLOC_DATA_READY=False,semantic_map_conversion_pending=True))
    atomic(ROOT/'download_status.json',status)
    if not complete:raise RuntimeError('Incomplete houses; preserved all verified data for bounded retry')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['metadata','download']);a=p.parse_args()
    try:metadata() if a.mode=='metadata' else download()
    except BaseException as exc:
        print('ZIND_WORKER_STOPPED',type(exc).__name__,flush=True)
        if a.mode=='download':atomic(ROOT/'worker_failure.json',dict(error_type=type(exc).__name__,reason=str(exc),unix_time=time.time()))
        sys.exit(1)
