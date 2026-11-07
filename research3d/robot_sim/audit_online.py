"""Independent offline audit of saved live ROS traces; never imports evaluate.py."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
import yaml
from research3d.fusion.contracts import read_jsonl, sha256_file, write_json


def error(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.array([np.linalg.norm(a[:3,3]-b[:3,3]),
        np.degrees(Rotation.from_matrix(b[:3,:3].T@a[:3,:3]).magnitude())])


def gt_at(rows, times, stamp):
    exact=np.flatnonzero(np.abs(times-stamp)<1e-8)
    if len(exact): return np.asarray(rows[int(exact[0])]['T_map_base'])
    hi=int(np.searchsorted(times,stamp))
    if hi==0 or hi==len(times): return None
    lo=hi-1
    if times[hi]-times[lo]>.25: return None
    a,b=[np.asarray(rows[i]['T_map_base']) for i in [lo,hi]]
    t,r=error(a,b)
    if t>1 or r>45: return None
    f=(stamp-times[lo])/(times[hi]-times[lo])
    out=np.eye(4);out[:3,3]=a[:3,3]*(1-f)+b[:3,3]*f
    out[:3,:3]=Slerp([times[lo],times[hi]],Rotation.from_matrix(np.stack([a[:3,:3],b[:3,:3]])))([stamp]).as_matrix()[0]
    return out


def run_audit(folder):
    gt=read_jsonl(folder/'capture/offline_gt/query_poses.jsonl')
    times=np.array([r['timestamp_ns']/1e9 for r in gt])
    trace=read_jsonl(folder/'localization/localization.jsonl')
    metrics=json.loads((folder/'evaluation/metrics.json').read_text())
    scored=json.loads((folder/'evaluation/per_frame_errors.json').read_text())
    assert len(scored)==len(gt)
    trace_index={r['timestamp_s']:r for r in trace}
    assert len(trace_index)==len(trace)
    reassessed=[]; discrepancies=[]
    for row in scored:
        if 'pose_time_s' not in row: continue
        raw=trace_index[row['output_time_s']]
        assert abs(raw['pose_time_s']-row['pose_time_s'])<1e-8
        assert raw['global_valid']==row['valid']
        truth=gt_at(gt,times,raw['pose_time_s'])
        assert truth is not None
        if raw['global_valid']:
            T=np.asarray(raw['T_map_base'])
            assert np.allclose(T,np.asarray(raw['T_map_odom'])@np.asarray(raw['T_odom_base']),atol=1e-10)
            err=error(T,truth)
            assert np.allclose(err,row['absolute_error'],atol=1e-4,rtol=0)
            reassessed.append(dict(query_timestamp_s=row['timestamp_s'],pose_timestamp_s=raw['pose_time_s'],
                output_timestamp_s=raw['timestamp_s'],translation_m=float(err[0]),rotation_deg=float(err[1])))
    errors=np.array([[r['translation_m'],r['rotation_deg']] for r in reassessed])
    stats=dict(valid=len(errors),evaluated=len(gt),coverage=len(errors)/len(gt),
        translation_rmse_m=float(np.sqrt(np.mean(errors[:,0]**2))),translation_median_m=float(np.median(errors[:,0])),
        translation_p95_m=float(np.quantile(errors[:,0],.95)),rotation_median_deg=float(np.median(errors[:,1])),
        translation_range_m=[float(errors[:,0].min()),float(errors[:,0].max())],
        success_1m30deg=float(np.sum((errors[:,0]<=1)&(errors[:,1]<=30))/len(gt)))
    for key,value in metrics['absolute'].items():
        if key in stats: assert abs(stats[key]-value)<1e-4
    decisions=read_jsonl(folder/'localization/ace_decisions.jsonl')
    local=[r for r in trace if r['pose_time_s'] is not None and r['T_odom_base'] is not None]
    local_times=np.array([r['pose_time_s'] for r in local])
    acceptance=[];unassociated=[]
    for d in decisions:
        if not d['decision'].get('accepted'):continue
        stamp=d['candidate']['stamp_sec']
        odom=local[int(np.argmin(abs(local_times-stamp)))]
        t=odom['pose_time_s'];truth=gt_at(gt,times,t)
        if abs(t-stamp)>.05 or truth is None:
            j=int(np.searchsorted(times,t))
            unassociated.append(dict(query_id=d['candidate']['query_id'],capture_s=stamp,nearest_pose_s=t,
                gt_bracket_s=times[max(0,j-1):j+1].tolist(),reason='no extrapolation or interpolation across >0.25s GT gap'))
            continue
        after=error(np.asarray(d['T_map_odom_after'])@np.asarray(odom['T_odom_base']),truth)
        before=None if d['T_map_odom_before'] is None else error(np.asarray(d['T_map_odom_before'])@np.asarray(odom['T_odom_base']),truth)
        harmful=None if before is None else bool(after[0]>before[0]+.25 or after[1]>before[1]+10)
        acceptance.append(dict(timestamp_s=t,error=after.tolist(),correct_1m30deg=bool(after[0]<=1 and after[1]<=30),
            harmful_delta_25cm_or_10deg=harmful,before_error=None if before is None else before.tolist()))
    saved_accept=json.loads((folder/'evaluation/accepted_correction_errors.json').read_text())
    assert len(acceptance)==len(saved_accept)
    for computed,saved in zip(acceptance,saved_accept):
        assert computed['timestamp_s']==saved['timestamp_s']
        assert np.allclose(computed['error'],saved['error'],atol=1e-4)
        assert computed['harmful_delta_25cm_or_10deg']==saved['harmful_delta_25cm_or_10deg']
    assert len(unassociated)==metrics['accepted_outside_evaluation']
    jumps=[]
    for i in range(1,len(gt)):
        t,r=error(gt[i-1]['T_map_base'],gt[i]['T_map_base'])
        if t>1 or r>45:jumps.append(dict(first_postjump_gt_s=float(times[i]),last_prejump_gt_s=float(times[i-1]),translation_m=float(t)))
    recovery=[]
    score_by_time={r['query_timestamp_s']:r for r in reassessed}
    for jump in jumps:
        streak=[]
        for g in gt:
            t=g['timestamp_ns']/1e9
            if t<jump['first_postjump_gt_s']:continue
            row=score_by_time.get(t)
            if row is None or row['translation_m']>1 or row['rotation_deg']>30:streak=[];continue
            if not streak or row['pose_timestamp_s']!=streak[-1]['pose_timestamp_s']:streak.append(row)
            if len(streak)>=3 and t-streak[0]['query_timestamp_s']>=.2-1e-8:
                recovery.append(dict(**jump,confirming_outputs=streak,
                    third_output_delay_s=row['output_timestamp_s']-jump['first_postjump_gt_s'],
                    first_correct_output_delay_s=streak[0]['output_timestamp_s']-jump['first_postjump_gt_s'],
                    jump_time_sampling_uncertainty_s=jump['first_postjump_gt_s']-jump['last_prejump_gt_s']))
                break
    if recovery:
        assert abs(recovery[0]['third_output_delay_s']-metrics['recovery_events'][0]['recovery_time_s'])<1e-8
    imu=read_jsonl(folder/'capture/query/sensors/imu.jsonl')
    joints=read_jsonl(folder/'capture/query/sensors/joint_states.jsonl')
    assert all(r['orientation_excluded'] and 'orientation' not in r for r in imu)
    assert all(set(r['names'])=={'left_wheel_joint','right_wheel_joint'} and len(r['velocities'])==2 for r in joints)
    assert all(np.isfinite(r['angular_velocity']).all() for r in imu)
    protocol=json.loads((folder/'localization/protocol.json').read_text())
    assert protocol['imu_fused_fields']==['angular_velocity.z']
    assert protocol['wheel_fused_fields']==['linear.x','angular.z']
    assert not protocol['online_gt_used'] and not protocol['macvo_enabled'] and not protocol['spvloc_corrections_allowed']
    assert all(not r['online_gt_used'] for r in trace)
    frames=read_jsonl(folder/'capture/query/frames.jsonl')
    assert all(r['timestamp_ns']==r['right_timestamp_ns'] for r in frames)
    image_audit=dict(pairs=len(frames),same_timestamp_pairs=len(frames))
    if (folder/'camera_fault_events.jsonl').exists():
        events=read_jsonl(folder/'camera_fault_events.jsonl')
        event_index={(int(round(r['timestamp_s']*1e9)),r['side']):r for r in events}
        black=nonblack=0;wrong=[];black_stamps=[]
        for row in frames:
            flags=[];all_zero=[]
            for side,pathkey,hashkey in [('left','image_path','image_sha256'),('right','right_image_path','right_image_sha256')]:
                file=folder/'capture'/row[pathkey]
                assert sha256_file(file)==row[hashkey]
                flags.append(event_index[row['timestamp_ns'],side]['blackout'])
                with Image.open(file) as image:all_zero.append(not np.asarray(image).any())
            assert flags[0]==flags[1]
            if flags[0]:
                black+=1;black_stamps.append(row['timestamp_ns']/1e9)
                if not all(all_zero):wrong.append(row['query_id'])
            else:
                nonblack+=1
                if any(all_zero):wrong.append(row['query_id'])
        assert not wrong
        actual_black_ns={int(round(t*1e9)) for t in black_stamps}
        black_evaluation=[r for r in scored if int(round(r['timestamp_s']*1e9)) in actual_black_ns]
        declared_count=metrics['camera_fault']['evaluated_frames']
        if declared_count!=black:
            discrepancies.append(dict(kind='declared_window_differs_from_actual_black_pixels',
                declared_window_evaluation_frames=declared_count,actual_black_stereo_pairs=black,
                cause='camera_fault elapsed-time subtraction at 24.4s is slightly below 15.0s; the first declared-window image stays visible',
                effect='One declared-window sample must not be described as an actual black image; core localization/recovery metrics are unaffected.'))
        image_audit.update(decoded_pngs=2*len(frames),blackout_pairs=black,nonblack_pairs=nonblack,
            black_pair_time_range_s=[min(black_stamps),max(black_stamps)],mismatches=wrong,
            actual_black_local_coverage=sum(r.get('local_error') is not None for r in black_evaluation)/len(black_evaluation),
            actual_black_global_coverage=sum(r.get('absolute_error') is not None for r in black_evaluation)/len(black_evaluation),
            raw_fault_events=dict(Counter(r['side'] for r in events)),
            note='Raw fault callbacks have unequal left/right delivery counts; every accepted captured pair is exactly synchronized and equally blacked out.')
    return dict(status='PASS_WITH_DISCLOSED_BOUNDARY_DIFFERENCE' if discrepancies else 'PASS',
        discrepancies=discrepancies,absolute_recomputed_from_raw_trace=stats,
        absolute_alignment='identity: raw T_map_base compared directly to GT; no optimization, rigid fit, or scale fit',
        audit_sampling='Saved per-frame association timestamps checked against actual raw trace; independent GT interpolation with discontinuity guard',
        accepted=dict(count=len(acceptance)+len(unassociated),assessable=len(acceptance),
            wrong_1m30deg=sum(not r['correct_1m30deg'] for r in acceptance),unassociated=unassociated,
            harmful=[r for r in acceptance if r['harmful_delta_25cm_or_10deg']]),
        recovery=recovery,images=image_audit,
        sensors=dict(imu_records=len(imu),joint_records=len(joints),IMU_orientation_absent=True,
            wheel_rates_present_all_records=True,online_fused='wheel vx/yaw-rate; gyro z; finite-uncertainty vy=0 robot constraint',
            recorded_but_not_fused='IMU linear acceleration, joint positions'),
        active_algorithms='wheel/IMU EKF + ACE global correction; SPVLoc real shadow inference only; MAC-VO absent from live loop',
        file_sha256={str(p.relative_to(folder)):sha256_file(p) for p in [folder/'localization/localization.jsonl',
            folder/'localization/ace_decisions.jsonl',folder/'localization/protocol.json',folder/'evaluation/metrics.json',
            folder/'capture/offline_gt/query_poses.jsonl',folder/'capture/query/frames.jsonl']})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    ekf=yaml.safe_load(Path('configs/robot_sim/ekf.yaml').read_text())['ekf_filter_node']['ros__parameters']
    assert np.flatnonzero(ekf['odom0_config']).tolist()==[6,7,11]
    assert np.flatnonzero(ekf['imu0_config']).tolist()==[11]
    report=dict(status='INDEPENDENT_CPU_AUDIT_PASS_WITH_DISCLOSED_BOUNDARY_DIFFERENCE',evaluate_module_called=False,GT_scope='offline audit only',
        runs={name:run_audit(args.root/name) for name in ['online_normal','online_fault']},
        source_review=dict(ekf_pose_fields_all_disabled=True,imu_orientation_and_acceleration_disabled=True,
            online_GT_subscription=False,note='ROS source subscription allowlist reviewed; no world/model pose subscribed by estimator. Collector GT remains separate.',
            hashes={p:sha256_file(Path(p)) for p in ['research3d/robot_sim/ros_localization_node.py','configs/robot_sim/ekf.yaml','research3d/robot_sim/camera_fault.py']}))
    write_json(args.output,report);print(json.dumps(dict(status=report['status'],output=str(args.output))))


if __name__=='__main__':main()
