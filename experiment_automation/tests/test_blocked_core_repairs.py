"""Scientific boundary and archive-identity regressions for concrete core adapters."""
import importlib.util,sys,json,io,zipfile
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproduction'))
from blocked_data import RemoteZip
from localization_metrics import evaluate


def test_archive_ranges_not_concatenated_or_proxy_cached(monkeypatch):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('camera/frame.png',bytes(range(256))*12345)
    binary=stream.getvalue();requests_seen=[]
    class Response:
        status_code=206
        def __init__(self,lo,hi):
            self.content=binary[lo:hi+1]
            self.headers={'Content-Range':f'bytes {lo}-{hi}/{len(binary)}','ETag':'immutable-source'}
        def raise_for_status(self):pass
        def close(self):pass
    class Session:
        def get(self,url,headers,**kwargs):
            lo,hi=map(int,headers['Range'][6:].split('-'));requests_seen.append((url,lo,hi));return Response(lo,hi)
    import requests
    monkeypatch.setattr(requests,'Session',Session)
    with zipfile.ZipFile(RemoteZip('https://publisher.invalid/raw.zip')) as z:
        assert z.read('camera/frame.png')==bytes(range(256))*12345
    assert requests_seen[0][1:]==(0,0)
    assert all('fyp_range=' in url for url,_,_ in requests_seen[1:])


def test_pose_acceptance_keeps_unavailable_queries_in_denominator(tmp_path):
    data=tmp_path/'data';run=tmp_path/'run';data.mkdir();run.mkdir()
    truth=np.repeat(np.eye(4)[None],600,axis=0)
    estimate=truth.copy();estimate[300:]=np.nan
    np.savez(data/'ground_truth_conditioned_motion_and_eval.npz',T_world_camera=truth)
    np.savez(run/'trajectory.npz',T_world_camera=estimate)
    (run/'execution.json').write_text(json.dumps({'localization_executed':True,'GT_online':False}))
    evaluate(data,run)
    v=json.loads((run/'metrics.json').read_text())
    assert v['recall_25cm_10deg']==.5 and v['valid_frames']==300
    assert v['map_coordinate_translation_rmse_valid_m']==0


def test_no_valid_pose_cannot_accept_core(tmp_path):
    data=tmp_path/'data';run=tmp_path/'run';data.mkdir();run.mkdir()
    np.savez(data/'ground_truth_conditioned_motion_and_eval.npz',T_world_camera=np.repeat(np.eye(4)[None],600,axis=0))
    np.savez(run/'trajectory.npz',T_world_camera=np.full((600,4,4),np.nan))
    with pytest.raises(AssertionError,match='No valid localization'):evaluate(data,run)
    assert not (run/'metrics.json').exists()


def test_core_has_real_metrics_dependencies_and_explicit_dataset():
    specs={t['task_id']:t for t in json.loads((ROOT/'manifests/tasks.json').read_text())}
    for paper in ['SG2Loc','OMCL']:
        for suffix in ['official_benchmark','replica_core']:
            task=specs[f'R.{paper}.{suffix}']
            assert task['command'] and task['depends_on']
            if suffix=='official_benchmark':
                assert len(task['depends_on'])==2 and all(d.endswith('.metrics') for d in task['depends_on'])
                assert task['dataset_id']=='ReplicaCAD_shared_FRL_layouts'
            else:
                assert task['command'][2] in ['global_accept','refine_accept']
                assert task['validation'][0]['path'].endswith('/metrics.json')
                assert task['dataset_id']=='ReplicaCAD_v3_sc0_staging_00'
            assert task['reproduction_level']=='Original Method on Alternative Data'
    for task in specs.values():
        if task['task_id'].startswith('R.MACVO.') and task['task_id'].endswith('.paper') and 'P000' in task['task_id']:
            assert '--frames' not in task['command']


def test_pixel_center_convention_conversion():
    # Habitat integer-index K versus SG2Loc's explicitly +0.5 pixel-center rays.
    x=np.arange(480);k_int=239.5;k_half=k_int+.5
    np.testing.assert_allclose((x-k_int)/240,(x+.5-k_half)/240,rtol=0,atol=0)


def test_sg_complete_window_resume_keeps_rng_and_rejects_other_source(tmp_path):
    from sg2loc_replicacad import save_window_checkpoint,load_window_checkpoint
    path=tmp_path/'checkpoint.npz';poses=np.full((600,4,4),np.nan)
    poses[:5]=np.eye(4);coarse=[{'last_frame':4,'pose_zup':np.eye(4).tolist()}]
    np.random.seed(2089);np.random.normal(size=13)
    save_window_checkpoint(path,'scene-and-config-A',5,poses,coarse)
    expected=np.random.normal(size=50)
    cursor,restored,rows=load_window_checkpoint(path,'scene-and-config-A',5)
    np.testing.assert_array_equal(np.random.normal(size=50),expected)
    np.testing.assert_array_equal(restored,poses);assert cursor==5 and rows==coarse
    with pytest.raises(AssertionError,match='mismatch'):
        load_window_checkpoint(path,'another-scene-or-code',5)
