"""Bind real Gazebo frontend outputs to the existing A--E fusion protocol.

This adapter never runs a substitute frontend and never fits estimates to GT.
The default conversion reads no GT. --ground-truth explicitly enables separate
offline diagnostics after the GT-free replay outputs have been written.

Two distinct timing modes are retained: offline_accuracy gives each cached
prediction ideal availability, while measured_service_fifo replays measured
service times on one serial resource, including accumulated backlog. The latter
is a lower bound excluding model swaps/I/O overhead, NOT a live speed result.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

from research3d.fusion.contracts import (digest, no_truth, pose, read_jsonl,
    sha256_file, validate_motion, validate_observation, write_json, write_jsonl)
from research3d.fusion.replay import run_replay
from research3d.fusion.protocol import validate_config


def finite_seconds(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError('runtime must be a measured finite nonnegative number')
    return float(value)


def index_rows(rows):
    result={}
    for row in rows:
        qid=row['query_id']
        if qid in result: raise ValueError('duplicate query id')
        result[qid]=row
    return result


def match_input(raw, inputs, manifest, require_image=True):
    no_truth(raw)
    if raw['query_id'] not in inputs: raise ValueError('prediction has unknown query id')
    frame=inputs[raw['query_id']]
    if raw['map_id']!=manifest['map_id'] or raw['map_sha256']!=manifest['map_sha256']:
        raise ValueError('prediction is bound to a different map or sensor rig')
    if raw['timestamp_ns']!=frame['timestamp_ns']:
        raise ValueError('prediction timestamp differs from its query image')
    if require_image and raw['image_sha256']!=frame['image_sha256']:
        raise ValueError('prediction did not consume the exact recorded image')
    return frame


def identity(manifest):
    source=manifest.get('source') or {}
    return dict(source_id='gazebo_procthor_structure',building_id=source.get('scene','known_structure'),
                map_id=manifest['map_id'],camera_id='robot_left_optical')


def base_row(frame,manifest,model_sha):
    return dict(schema_version=1,**identity(manifest),query_id=frame['query_id'],
                capture_time_s=frame['timestamp_ns']/1e9,
                provenance=dict(image_sha256=frame['image_sha256'],model_sha256=model_sha,
                    calibration_sha256=digest(dict(camera=frame['camera'],rig=manifest['calibration'])),
                    map_sha256=manifest['map_sha256'],online_gt_used=False))


def adapt_motion(raw_rows,inputs,manifest,protocol,timing):
    by_id=index_rows(raw_rows);timing=index_rows(timing)
    if set(by_id)!=set(inputs) or set(timing)!=set(inputs):
        raise ValueError('MAC-VO must cover the full frozen trajectory, including explicit failures')
    model_sha=digest(dict(frontend=protocol['checkpoint_sha256'],motion=protocol['motion_checkpoint_sha256']))
    rows=[];previous_valid=False;segment=0
    for qid,frame in inputs.items():
        raw=by_id[qid];match_input(raw,inputs,manifest,False)
        if raw['status'] not in {'estimated','lost'}: raise ValueError('unknown MAC-VO status')
        t=timing[qid]
        if t['timestamp_ns']!=frame['timestamp_ns']: raise ValueError('runtime/image timestamp mismatch')
        elapsed=finite_seconds(t['inference_ms']/1000)
        valid=raw['status']=='estimated'
        if valid and not previous_valid: segment+=1
        row=base_row(frame,manifest,model_sha)
        row.update(source='macvo',valid=valid,segment_id=f'continuous_segment_{segment}',
                   T_odom_camera=pose(raw['T_odom_camera']).tolist() if valid else None,
                   measured_service_s=elapsed,completed_time_s=row['capture_time_s']+elapsed)
        validate_motion(row);rows.append(row);previous_valid=valid
    return rows


def adapt_absolute(raw_rows,inputs,manifest,source):
    index_rows(raw_rows);rows=[]
    for raw in sorted(raw_rows,key=lambda r:r['timestamp_ns']):
        frame=match_input(raw,inputs,manifest)
        if raw['status'] not in {'ok','failed'}: raise ValueError('unhandled frontend failure status; cannot invent a candidate')
        model=raw['head_sha256'] if source=='ace' else raw['checkpoint_sha256']
        row=base_row(frame,manifest,model)
        elapsed=finite_seconds(raw.get('rpc_wall_seconds',raw['latency_ms']/1000) if source=='ace' else raw['latency_s'])
        candidates=[]
        if raw['status']=='ok':
            if source=='ace':
                candidates=[dict(rank=1,T_map_camera=pose(raw['T_map_camera_optical']).tolist(),
                                 quality=dict(pnp_inliers=raw['inliers']))]
            else:
                if not 1<=len(raw['candidates'])<=3: raise ValueError('SPVLoc must provide actual Top1--3 hypotheses')
                for rank,candidate in enumerate(raw['candidates'],1):
                    if candidate['rank']!=rank: raise ValueError('SPVLoc candidate order changed')
                    T=raw['refined_top1_T_map_camera_optical'] if rank==1 else candidate['T_map_camera_optical']
                    candidates.append(dict(rank=rank,T_map_camera=pose(T).tolist(),
                        quality=dict(viewport_score=candidate['viewport_score'],
                                     score_scope='initial_viewport_score_not_calibrated_probability'),
                        refinement='official_top1' if rank==1 else 'initial_hypothesis'))
                row['provenance']['camera_preprocessing']=raw.get('camera_crop')
                row['provenance']['top_k_policy']='official_refined_top1_with_unmodified_initial_rank2_rank3'
        row.update(source=source,status='VALID' if candidates else 'FAILED',candidates=candidates,
                   measured_service_s=elapsed,completed_time_s=row['capture_time_s']+elapsed)
        validate_observation(row);rows.append(row)
    if not rows: raise ValueError('missing actual '+source+' inference outputs')
    return rows


def schedule(motions,observations,enabled,mode):
    """No GT; one frozen serial-service schedule for each declared variant."""
    m=copy.deepcopy(motions);o=copy.deepcopy([r for r in observations if r['source'] in enabled])
    if mode=='offline_accuracy':
        for row in m+o: row['completed_time_s']=row['capture_time_s']
    elif mode=='measured_service_fifo':
        priority={'macvo':0,'ace':1,'spvloc':2}
        tasks=sorted(m+o,key=lambda r:(r['capture_time_s'],priority[r['source']]))
        free=-float('inf')
        for row in tasks:
            start=max(free,row['capture_time_s'])
            row['completed_time_s']=start+finite_seconds(row['measured_service_s'])
            row['queued_service_wait_s']=start-row['capture_time_s'];free=row['completed_time_s']
    else: raise ValueError('unknown timing mode')
    return m,o


def load_frontends(dataset,macvo,ace,spvloc):
    dataset=Path(dataset).resolve(strict=True)
    manifest=json.loads((dataset/'map_manifest.json').read_text())
    frames=read_jsonl(dataset/'query/frames.jsonl');inputs=index_rows(frames)
    if len(frames)<3: raise ValueError('not a trajectory')
    last=-1
    for frame in frames:
        no_truth(frame)
        if frame['timestamp_ns']<=last: raise ValueError('nonmonotonic query capture times')
        last=frame['timestamp_ns']
        if frame['map_id']!=manifest['map_id']: raise ValueError('mixed source maps')
        if (frame['camera']['width'],frame['camera']['height']) != (manifest['camera']['width'],manifest['camera']['height']) or not np.allclose(frame['camera']['K'],manifest['camera']['K'],atol=1e-6):
            raise ValueError('query calibration differs from map calibration')
        for path_key,sha_key in [('image_path','image_sha256'),('right_image_path','right_image_sha256')]:
            path=(dataset/frame[path_key]).resolve(strict=True)
            if not path.is_relative_to(dataset/'query'): raise ValueError('online image outside query split')
            if sha256_file(path)!=frame[sha_key]: raise ValueError('captured image modified')
        if frame['right_timestamp_ns']!=frame['timestamp_ns']: raise ValueError('unsynchronized stereo')
    macvo,ace,spvloc=map(Path,(macvo,ace,spvloc))
    protocol=json.loads((macvo/'protocol.json').read_text())
    if protocol['online_manifest_sha256']!=sha256_file(dataset/'query/frames.jsonl'):
        raise ValueError('MAC-VO input manifest differs from frozen query inventory')
    if protocol['map_id']!=manifest['map_id'] or protocol['map_sha256']!=manifest['map_sha256']:
        raise ValueError('MAC-VO protocol/map mismatch')
    motions=adapt_motion(read_jsonl(macvo/'predictions.jsonl'),inputs,manifest,protocol,
                         json.loads((macvo/'per_frame.json').read_text()))
    a=adapt_absolute(read_jsonl(ace/'predictions.jsonl'),inputs,manifest,'ace')
    s=adapt_absolute(read_jsonl(spvloc/'predictions.jsonl'),inputs,manifest,'spvloc')
    if {r['query_id'] for r in a}!={r['query_id'] for r in s}:
        raise ValueError('ACE/SPVLoc must use the same predeclared sampled images')
    return manifest,inputs,motions,a+s


def offline_truth(path,inputs,manifest):
    original=read_jsonl(path);index=index_rows(original)
    if set(index)!=set(inputs): raise ValueError('offline GT must cover the entire independent query inventory')
    result=[]
    for qid,frame in inputs.items():
        row=index[qid]
        if row['timestamp_ns']!=frame['timestamp_ns']: raise ValueError('offline GT/image timestamp mismatch')
        result.append(dict(**identity(manifest),query_id=qid,timestamp_s=frame['timestamp_ns']/1e9,
                           T_map_camera=pose(row['T_map_camera']).tolist(),image_sha256=frame['image_sha256']))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','macvo','ace','spvloc','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/fusion/experiment.json'))
    p.add_argument('--ground-truth',type=Path,help='explicit separate offline evaluation input')
    p.add_argument('--max-output-age-s',type=float,default=.5)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('refuse to overwrite a replay result')
    manifest,inputs,motions,observations=load_frontends(args.dataset,args.macvo,args.ace,args.spvloc)
    protocol=validate_config(json.loads(args.config.read_text()))
    write_jsonl(args.output/'motion.jsonl',motions);write_jsonl(args.output/'observations.jsonl',observations)
    versions=['A_macvo_local','B_macvo_ace','C_macvo_spvloc','D_three_fixed','E_three_recovery']
    modes=['offline_accuracy','measured_service_fifo'];runs=[]
    for mode in modes:
        for variant in versions:
            cfg={**protocol['replay_defaults'],**protocol['variants'][variant]}
            m,o=schedule(motions,observations,cfg['absolute_sources'],mode)
            outputs,audit=run_replay(m,o,cfg)
            target=args.output/mode/variant
            write_jsonl(target/'outputs.jsonl',outputs)
            write_jsonl(target/'scheduled_observations.jsonl',o)
            audit.update(mode=mode,variant=variant,config=cfg,real_time_claim=False,
                         protocol_scope='replay defaults/variants only; actual Gazebo rig and model training come from input provenance',
                         timing_scope='ideal availability accuracy only' if mode=='offline_accuracy' else
                         'single-resource FIFO from measured service; warm models, excludes model swap, unmeasured I/O and OS overhead; lower bound, not actual live execution')
            write_json(target/'audit.json',audit)
            runs.append((mode,variant,target,outputs,o))
    # Only now, after all online replay outputs are immutable, read query GT.
    if args.ground_truth:
        from research3d.fusion.diagnostics import complementarity,correction_metrics
        from research3d.fusion.metrics import system_metrics
        truth=offline_truth(args.ground_truth,inputs,manifest)
        sampled={r['query_id'] for r in observations}
        comp=complementarity(observations,[t for t in truth if t['query_id'] in sampled],
                             translation_threshold_m=1.,rotation_threshold_deg=30.)
        comp['sampling_scope']='same fixed ACE/SPVLoc inference queries only; system denominator is separately all recorded frames'
        write_json(args.output/'complementarity.json',comp)
        for mode,variant,target,outputs,obs in runs:
            score=system_metrics(outputs,truth,translation_threshold_m=1.,rotation_threshold_deg=30.,
                                 max_output_age_s=args.max_output_age_s if mode=='measured_service_fifo' else None)
            corrections=correction_metrics(outputs,obs,truth,translation_threshold_m=1.,rotation_threshold_deg=30.)
            write_json(target/'metrics.json',dict(system=score,corrections=corrections,
                real_time_claim=False,recovery_events='NOT_MEASURED: no independent failure events supplied',
                local_baseline_note='A has no map gauge and correctly emits no global pose; use separate MAC-VO aligned ATE/RPE'))
    files=[args.dataset/'query/frames.jsonl',args.dataset/'map_manifest.json',args.config]
    for folder in (args.macvo,args.ace,args.spvloc): files.append(folder/'predictions.jsonl')
    files.extend([args.macvo/'protocol.json',args.macvo/'per_frame.json'])
    if args.ground_truth: files.append(args.ground_truth)
    write_json(args.output/'receipt.json',dict(status='REPLAY_AND_OFFLINE_EVALUATION_COMPLETE' if args.ground_truth else 'REPLAY_COMPLETE_NOT_EVALUATED',
        queries=len(inputs),sampled_absolute_queries=len(observations)//2,variants=versions,modes=modes,
        input_sha256={str(f.resolve()):sha256_file(f) for f in files},map_id=manifest['map_id'],
        source_scope='one known Gazebo development building; not original benchmark nor physical robot validation',
        predictions_modified=False,GT_in_replay=False,real_time_claim=False))
    print(json.dumps(dict(output=str(args.output),runs=len(runs),GT_evaluation=bool(args.ground_truth))))


if __name__=='__main__': main()
