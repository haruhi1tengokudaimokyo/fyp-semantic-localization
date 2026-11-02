"""One-shot, data-only ProcTHOR 100/20/20 preparation, independent of Codex.

Never launches a training command. Existing SPVLoc work owns the GPU until its
whole train/evaluate process exits. All acceptance uses source data, not scores.
Restarting reuses per-house atomic capture/acceptance receipts.
"""
import argparse, collections, csv, hashlib, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path
from prepare_large_split import MANIFEST, ROOT, TARGET, atomic, main as freeze

HERE = Path(__file__).resolve().parent
OUT = Path('/data/fyp_3d/runs/semantic_localization_20261102/data_100_20_20')
PYTHON = '/data/fyp_3d/envs/semantic_loc/bin/python'
DEPENDENCY = 'fyp-spvloc-16-pipeline.service'
GiB = 1024**3

def state(stage, **kwargs):
    atomic(OUT / 'preparation_status.json', dict(stage=stage, unix_time=time.time(),
           target=TARGET, automatic_full_training=False, **kwargs))
    print('DATA_PREPARATION', stage, json.dumps(kwargs), flush=True)

def environment():
    env = os.environ.copy()
    env.update(LD_LIBRARY_PATH='/data/fyp_3d/sources/spvloc_redner/redner-dependencies/embree/lib-linux',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYOPENGL_PLATFORM='egl')
    return env

def command(scene, name, args, timeout):
    log = OUT / f'{scene.name}_{name}.log'
    with log.open('a') as stream:
        process = subprocess.Popen([PYTHON, *map(str,args)], cwd=HERE,
                                   env=environment(), stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            raise
    if code:
        raise RuntimeError(f'{name} exited {code}; see {log}')

def storage(initial_bytes):
    free = shutil.disk_usage(ROOT).free
    used = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    assert free > 101*GiB, 'Stop before crossing 100GiB free-space floor'
    assert used - initial_bytes < 59*GiB, 'Stop before crossing 60GiB additional-data ceiling'
    return used, free

def wait_for_gpu():
    deadline = time.monotonic() + 24*3600
    while True:
        info = subprocess.check_output(['systemctl','--user','show',DEPENDENCY,
                                       '-p','ActiveState','-p','MainPID'], text=True).strip()
        # Ignore the graphics/display engine; only wait for actual compute users.
        cuda = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                       '--format=csv,noheader,nounits'], text=True).strip()
        if 'ActiveState=active' not in info and 'ActiveState=activating' not in info and not cuda:
            state('GPU_AVAILABLE', previous_worker=info)
            return
        if time.monotonic() > deadline:
            raise TimeoutError('Protected GPU dependency still occupied after 24 hours; no task stopped')
        if not (OUT / 'preparation_status.json').exists() or json.loads((OUT/'preparation_status.json').read_text())['stage'] != 'WAITING_FOR_EXISTING_SPVLOC':
            state('WAITING_FOR_EXISTING_SPVLOC', protected_unit=DEPENDENCY,
                  worker=info, compute_pids=cuda.splitlines())
        time.sleep(30)

def qualify(scene):
    receipt = scene / 'capture_receipt.json'
    assert receipt.exists(), 'No complete acquisition receipt'
    capture = json.loads(receipt.read_text())
    assert capture['query_frames'] >= 160 and capture['panoramas'] >= 20
    # Existing independent geometric checks are reusable because the source and
    # capture are immutable; new houses run exactly that same CPU renderer check.
    qpath = scene / 'qualification.json'
    if not qpath.exists():
        code = ('import sys;from pathlib import Path;from qualify_procthor import qualify;'
                'qualify(Path(sys.argv[1]))')
        command(scene, 'geometry', ['-c',code,scene], 1800)
    quality = json.loads(qpath.read_text())
    assert quality['status'] == 'DATA_READY_PILOT', 'Independent geometric acceptance failed'
    assert quality['queries'] == capture['query_frames']
    command(scene, 'panorama', [HERE/'qualify_panorama.py',scene], 300)
    pquality = json.loads((scene/'panorama_qualification.json').read_text())
    assert pquality['status'] == 'PANORAMA_READY'
    checksums = []
    paths = [scene/'procthor_house.json', scene/'annotation_3d.json', receipt, qpath,
             scene/'panorama_qualification.json']
    paths.extend(sorted((scene/'2D_rendering').rglob('*')))
    for path in paths:
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
            checksums.append(f'{digest.hexdigest()}  {path.relative_to(scene)}\n')
    tmp = scene / 'accepted_files.sha256.tmp'
    tmp.write_text(''.join(checksums)); tmp.replace(scene/'accepted_files.sha256')
    return dict(query_frames=capture['query_frames'], panoramas=capture['panoramas'],
                structural_pixel_counts=quality['structural_pixel_counts'],
                raw_class_pixel_counts=capture['raw_class_pixel_counts'],
                depth_p95_m=quality['p95_depth_error_m'],
                semantic_agreement=quality['median_structural_semantic_agreement'],
                checksum_list_sha256=hashlib.sha256((scene/'accepted_files.sha256').read_bytes()).hexdigest())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = freeze()
    assert len({x['source_geometry_sha256'] for x in manifest['houses']}) == len(manifest['houses'])
    for house in manifest['houses']:
        scene = ROOT / f"scene_{house['scene_id']:05d}"
        assert (scene/'annotation_3d.json').is_file()
        original = json.loads((scene/'procthor_house.json').read_text())
        identity = {k:original[k] for k in ['rooms','walls','doors','windows','objects']}
        sha = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert sha == house['source_geometry_sha256']
    initial_path = OUT/'storage_baseline.json'
    if not initial_path.exists():
        initial_bytes = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
        atomic(initial_path, dict(dataset_bytes=initial_bytes, free_bytes=shutil.disk_usage(ROOT).free))
    initial_bytes = json.loads(initial_path.read_text())['dataset_bytes']
    storage(initial_bytes)
    if args.preflight_only:
        print('PREFLIGHT_PASSED', len(manifest['houses']), TARGET, flush=True)
        return
    wait_for_gpu()
    accepted, failures, counts = [], [], collections.Counter()
    for house in manifest['houses']:
        split = house['split']
        if counts[split] == TARGET[split]: continue
        used, free = storage(initial_bytes)  # capacity failures stop acquisition, not scientific rejects
        scene = ROOT / f"scene_{house['scene_id']:05d}"
        state('GENERATING_AND_ACCEPTING', scene_id=house['scene_id'], accepted=dict(counts), free_GiB=free/GiB)
        try:
            if not (scene/'capture_receipt.json').exists():
                command(scene,'capture',[HERE/'capture_procthor.py','--scenes',house['scene_id']],900)
            evidence = qualify(scene)
        except Exception as exc:
            failure = dict(**house, error=repr(exc))
            failures.append(failure)
            atomic(OUT/f"rejected_{house['scene_id']:05d}.json", failure)
            # Fixed scientific failures are isolated. No test-score-based retries
            # or threshold changes; try the next frozen same-split candidate.
            continue
        accepted.append(dict(**house, **evidence)); counts[split] += 1
        atomic(OUT/'accepted_progress.json', dict(accepted=accepted, rejected=failures, counts=dict(counts)))
    complete = dict(counts) == TARGET
    used, free = storage(initial_bytes)
    result = dict(status='DATA_READY' if complete else 'BLOCKED_INSUFFICIENT_QUALIFIED_BUILDINGS',
                  target=TARGET, accepted_counts=dict(counts), houses=accepted, rejected=failures,
                  queries=sum(h['query_frames'] for h in accepted),
                  panoramas=sum(h['panoramas'] for h in accepted),
                  source_manifest=str(MANIFEST), independent_synthetic_buildings=len(accepted),
                  official_publisher_split_preserved=True, trained=False, automatic_full_training=False,
                  available_bytes=free, dataset_bytes=used, new_bytes=used-initial_bytes,
                  method_scope='ProcTHOR adaptation of SPVLoc, not original Structured3D benchmark')
    atomic(OUT/'DATASET_SCENE_MANIFEST.json', result)
    # Freeze the exact accepted split for tomorrow, separate from today's training.
    atomic(ROOT/'building_split_manifest_100_20_20_accepted.json', result)
    with (OUT/'SEMANTIC_CLASS_DISTRIBUTION.csv').open('w') as stream:
        writer = csv.writer(stream); writer.writerow(['scene_id','split','class','pixels'])
        for house in accepted:
            for name, value in sorted(house['raw_class_pixel_counts'].items()):
                writer.writerow([house['scene_id'],house['split'],name,value])
    text = (f"# Overnight semantic dataset\n\nStatus: {result['status']}\n\n"
            f"Accepted train/val/test: {dict(counts)}; raw queries: {result['queries']}; panoramas: {result['panoramas']}.\n\n"
            f"Independent synthetic house geometries: {len(accepted)}. Rejected candidates: {len(failures)}.\n\n"
            f"Disk available: {free/GiB:.2f} GiB; dataset: {used/GiB:.2f} GiB.\n\n"
            "All accepted houses passed native RGB/depth/pose decoding, fixed structural ray depth/semantic thresholds, and upstream panorama crossprojection. Raw class dictionaries, source geometry, capture receipts and per-file SHA256 are retained.\n\n"
            "No training is started by this tool. The existing small-scale model is separate.\n")
    (OUT/'DATASET_DOWNLOAD_AUDIT.md').write_text(text)
    (OUT/'DATASET_STORAGE_REPORT.md').write_text(text)
    state(result['status'], counts=dict(counts), queries=result['queries'], free_GiB=free/GiB)
    if not complete: raise RuntimeError('Frozen reserves exhausted; inspect rejected scenes without relaxing thresholds')

if __name__ == '__main__':
    try: main()
    except BaseException as exc:
        if OUT.exists(): state('FAILED', error=repr(exc))
        raise
