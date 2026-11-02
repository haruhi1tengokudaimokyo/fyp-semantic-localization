"""Full-pipeline covariance ablation and preregistered precision/speed checks."""
import argparse,json
from pathlib import Path
from blocked_data import atomic

def main(root,section="both"):
    data={}
    modes=['original_fp32']+(['identity_fp32'] if section in ['both','ablation'] else [])+(['original_fp16'] if section in ['both','precision'] else [])
    for mode in modes:
        d=root/mode
        data[mode]={k:json.loads((d/name).read_text()) for k,name in [('protocol','protocol.json'),('metrics','official_metrics.json'),('profile','stage_profile.json')]}
    base=data['original_fp32'];identity=data.get('identity_fp32');half=data.get('original_fp16')
    assert len({v['metrics']['frames'] for v in data.values()})==1
    assert all(v['metrics']['full_source_sequence'] and not v['metrics']['GT_online'] for v in data.values())
    if identity is not None:
        a,b=base['protocol']['config']['Odometry'],identity['protocol']['config']['Odometry']
        a=json.loads(json.dumps(a));b=json.loads(json.dumps(b));a.pop('name');b.pop('name');assert a==b
        assert identity['protocol']['backend_covariance']=='identity'
    for key in ['seed','data','source_commit','experiment_config']:
        assert all(v['protocol'][key]==base['protocol'][key] for v in data.values()),key
    if half is not None:
        original_config=json.loads(json.dumps(base['protocol']['config']['Odometry']))
        half_config=json.loads(json.dumps(half['protocol']['config']['Odometry']))
        original_config.pop('name');half_config.pop('name')
        for key in ['enc_dtype','dec_dtype']:
            assert original_config['frontend']['args'].pop(key)=='fp32'
            assert half_config['frontend']['args'].pop(key)=='fp16'
        assert original_config==half_config,'Precision trial must not also change decoder iterations/algorithm'
        assert original_config['frontend']['args']['decoder_depth']==12
    assert all(v['profile']['solver_jobs']>0 for v in data.values())
    precision_gate=None;speedup=None
    if half is not None:
        ate=base['metrics']['independent_ATE_m'];candidate=half['metrics']['independent_ATE_m']
        precision_gate=candidate<=max(1.05*ate,ate+.01) and half['metrics']['lost_frames']<=base['metrics']['lost_frames']+1
        speedup=base['profile']['inference_pipeline_mean_ms']/half['profile']['inference_pipeline_mean_ms']
    atomic(root/('comparison.json' if section in ['both','ablation'] else 'precision_comparison.json'),dict(status='accepted',full_pipeline_ablation_executed=identity is not None,
        live_original_TartanMotionNet=True,all_metrics_independent=True,
        frames=base['metrics']['frames'],dataset_scope='Public TartanAir v1, not the paper v2 test suite',
        exploratory_precision_validation=True,preregistered_precision_gate='ATE <= max(1.05 * FP32 ATE, FP32 ATE + .01m); lost frames <= FP32 + 1',
        precision_gate_passed=precision_gate,full_pipeline_FP16_speedup=speedup,
        speed_gain_observed=speedup>1 if speedup is not None else None,results=data,
        limitation='Identity run is real full pipeline w/o CovOpt; frontend covariance-based keypoint selection remains enabled. FP16 is a separate hardware-adapted variant; not the BF16 4-iteration Fast configuration.'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--section',choices=['both','ablation','precision'],default='both');a=p.parse_args();main(a.root,a.section)
