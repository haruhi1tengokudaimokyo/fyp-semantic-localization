import copy,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reproduction'))
from macvo_followup_acceptance import main

def trial(tmp_path):
    for mode in ['original_fp32','identity_fp32','original_fp16']:
        p=tmp_path/mode;p.mkdir()
        precision='fp16' if mode.endswith('fp16') else 'fp32'
        protocol=dict(seed=1305,data={'args':{'root':'hospital','gtPose':False}},source_commit='fixed',experiment_config='Paper_Reproduce.yaml',backend_covariance='identity' if mode.startswith('identity') else 'original',config={'Odometry':{'name':mode,'frontend':{'args':{'enc_dtype':precision,'dec_dtype':precision,'decoder_depth':12}},'motion':{'type':'TartanMotionNet'}}})
        values={'protocol.json':protocol,'official_metrics.json':dict(frames=563,full_source_sequence=True,GT_online=False,independent_ATE_m=.1,lost_frames=0),'stage_profile.json':dict(solver_jobs=562,inference_pipeline_mean_ms=500 if precision=='fp16' else 1000)}
        for name,v in values.items():(p/name).write_text(json.dumps(v))

def test_comparison_does_not_hide_failed_precision_gate(tmp_path):
    trial(tmp_path);p=tmp_path/'original_fp16/official_metrics.json';v=json.loads(p.read_text());v['independent_ATE_m']=1.;p.write_text(json.dumps(v))
    main(tmp_path);result=json.loads((tmp_path/'comparison.json').read_text())
    assert not result['precision_gate_passed'] and result['full_pipeline_FP16_speedup']==2

def test_comparison_rejects_different_scene_or_fast_iterations(tmp_path):
    trial(tmp_path);p=tmp_path/'original_fp16/protocol.json';v=json.loads(p.read_text());v['config']['Odometry']['frontend']['args']['decoder_depth']=4;p.write_text(json.dumps(v))
    with pytest.raises(AssertionError,match='iterations'):main(tmp_path)
    v['config']['Odometry']['frontend']['args']['decoder_depth']=12;v['data']['args']['root']='other_scene';p.write_text(json.dumps(v))
    with pytest.raises(AssertionError,match='data'):main(tmp_path)

def test_followups_keep_original_baselines_and_live_motion():
    specs={t['task_id']:t for t in json.loads((ROOT/'manifests/tasks.json').read_text())}
    for scene in ['hospital','abandonedfactory']:
        assert specs[f'R.MACVO.{scene}.P000.paper']['command'][1].endswith('/macvo_full_sequence.py')
        for mode in ['original_fp32','identity_fp32','original_fp16']:
            task=specs[f'R.MACVO.{scene}.P000.audit.{mode}.v1'];cmd=task['command']
            assert '--frames' not in cmd and '--motion-weights' in cmd and '--capture-graphs' in cmd
            assert task['resources']['gpu'] and 'R.MACVO.solver_spawn_contract.v1' in task['depends_on']

def test_precision_failure_does_not_block_real_ablation_report(tmp_path):
    import shutil
    trial(tmp_path);shutil.rmtree(tmp_path/'original_fp16')
    main(tmp_path,'ablation')
    result=json.loads((tmp_path/'comparison.json').read_text())
    assert result['full_pipeline_ablation_executed'] and result['precision_gate_passed'] is None
    specs={t['task_id']:t for t in json.loads((ROOT/'manifests/tasks.json').read_text())}
    for scene in ['hospital','abandonedfactory']:
        deps=specs[f'R.MACVO.{scene}.P000.audit.comparison.v1']['depends_on']
        assert len(deps)==2 and not any('fp16' in dep for dep in deps)
