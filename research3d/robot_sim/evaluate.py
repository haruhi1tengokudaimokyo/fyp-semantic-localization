"""Independent offline simulator scoring. Never import this module online.

Absolute poses are scored in the original fixed map, with no fit to truth.
Local odometry gets one explicitly reported initial SE(2) origin transform;
this diagnoses dead reckoning and is not a global-localization result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from research3d.fusion.contracts import discrepancy, read_jsonl, write_json


def interpolate(times, poses, t, max_gap=.25):
    i = int(np.searchsorted(times, t))
    if i < len(times) and abs(times[i] - t) < 1e-6:
        return np.array(poses[i], float)
    if i == 0 or i == len(times) or times[i] - times[i-1] > max_gap:
        return None
    dt = times[i] - times[i-1]
    a, b = np.asarray(poses[i-1]), np.asarray(poses[i])
    if np.linalg.norm(b[:3,3]-a[:3,3])>1. or Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude()>np.deg2rad(45):
        return None  # Never invent intermediate truth through a kidnapping jump.
    result = np.eye(4)
    result[:3, 3] = a[:3, 3] + (b[:3, 3] - a[:3, 3]) * (t-times[i-1])/dt
    result[:3, :3] = Slerp(times[i-1:i+1], Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]])))(t).as_matrix()
    return result


def statistics(errors, total):
    if not errors:
        return dict(valid=0, evaluated=total, coverage=0., translation_rmse_m=None,
                    translation_median_m=None, rotation_median_deg=None, success_1m30deg=0.)
    e = np.array(errors)
    return dict(valid=len(e), evaluated=total, coverage=len(e)/total,
                translation_rmse_m=float(np.sqrt(np.mean(e[:,0]**2))),
                translation_median_m=float(np.median(e[:,0])),
                translation_p95_m=float(np.quantile(e[:,0], .95)),
                rotation_median_deg=float(np.median(e[:,1])),
                success_5cm5deg=float(np.sum((e[:,0]<=.05)&(e[:,1]<=5))/total),
                success_25cm10deg=float(np.sum((e[:,0]<=.25)&(e[:,1]<=10))/total),
                success_1m30deg=float(np.sum((e[:,0]<=1)&(e[:,1]<=30))/total))


def runtime_statistics(decisions):
    """Separate measured wall service from simulation capture-to-use age."""
    result={}
    for source in sorted({r['candidate'].get('source','unknown') for r in decisions}):
        rows=[r for r in decisions if r['candidate'].get('source','unknown')==source]
        def summary(values):
            return None if not values else dict(count=len(values),p50=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=float(max(values)))
        result[source]=dict(calls_returned=len(rows),accepted=sum(r['decision']['accepted'] for r in rows),
            outcomes=dict(Counter(r['decision']['reason'] for r in rows)),
            rpc_roundtrip_wall_s=summary([r['candidate']['runtime_timing']['rpc_roundtrip_wall_s'] for r in rows if 'runtime_timing' in r['candidate']]),
            pending_wait_wall_s=summary([r['candidate']['runtime_timing']['pending_wait_wall_s'] for r in rows if 'runtime_timing' in r['candidate']]),
            capture_to_decision_simulation_s=summary([r['received_time_s']-r['candidate']['stamp_sec'] for r in rows]))
    return result


def path_statistics(times,poses):
    """Keep simulator relocation jumps separate from physical travelled path."""
    continuous,jumps=[],[]
    for i,(a,b) in enumerate(zip(poses,poses[1:]),1):
        a,b=np.asarray(a),np.asarray(b)
        distance=float(np.linalg.norm(b[:3,3]-a[:3,3]))
        angle=float(np.rad2deg(Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude()))
        if distance>1. or angle>45.:
            jumps.append(dict(timestamp_s=float(times[i]),translation_m=distance,rotation_deg=angle))
        else:continuous.append(distance)
    return dict(sampled_continuous_path_length_m=float(sum(continuous)),
                relocation_jump_distance_m=float(sum(x['translation_m'] for x in jumps)),
                total_sampled_displacement_including_relocations_m=float(sum(continuous)+sum(x['translation_m'] for x in jumps)),
                detected_jumps=jumps,
                discontinuity_rule='adjacent GT displacement >1m or rotation >45deg; jumps excluded from travelled path')


def recovery_statistics(events,rows,gt_times,gt_poses):
    """Recovery uses the confirming output time, with separate event horizons."""
    events=sorted([e for e in events if e['type']=='kidnap'],key=lambda e:e['requested_at_timestamp_ns'])
    jumps=path_statistics(gt_times,gt_poses)['detected_jumps']
    result=[]
    for index,event in enumerate(events):
        requested=event['requested_at_timestamp_ns']/1e9
        next_request=events[index+1]['requested_at_timestamp_ns']/1e9 if index+1<len(events) else float('inf')
        horizon=min(next_request,float(gt_times[-1]))
        candidates=[j['timestamp_s'] for j in jumps if requested<j['timestamp_s']<next_request and j['timestamp_s']<=horizon]
        actual=candidates[0] if candidates else None
        confirmed=None;first_correct=None;streak=[];seen_outputs=set()
        if actual is not None:
            for frame in rows:
                if frame['timestamp_s']<actual:continue
                if frame['timestamp_s']>=next_request or frame['timestamp_s']>horizon:break
                error=frame.get('absolute_error')
                output_time=frame.get('output_time_s')
                pose_time=frame.get('pose_time_s')
                correct=(error is not None and error[0]<=1 and error[1]<=30
                         and output_time is not None and actual<=output_time<=horizon and output_time<next_request)
                if not correct:
                    streak=[];seen_outputs.clear();continue
                key=pose_time  # Re-publishing one stale pose is not independent confirmation.
                if key in seen_outputs:continue
                seen_outputs.add(key);streak.append(frame)
                if len(streak)>=3 and streak[-1]['timestamp_s']-streak[0]['timestamp_s']>=.2-1e-6:
                    first_correct=streak[0]['output_time_s'];confirmed=streak[-1]['output_time_s'];break
        result.append(dict(event='kidnap',requested_timestamp_s=requested,actual_jump_timestamp_s=actual,
            horizon_timestamp_s=horizon,event_observed_in_gt=actual is not None,
            success=confirmed is not None,confirmed_output_timestamp_s=confirmed,
            first_correct_output_timestamp_s=first_correct,
            recovery_time_s=None if confirmed is None else confirmed-actual,
            first_correct_delay_s=None if first_correct is None else first_correct-actual,
            criterion='three distinct correct outputs on consecutive evaluated frames spanning >=0.2s, <=1m/30deg; detection time is third confirming output availability',
            censored=confirmed is None))
    return result


def evaluate(dataset, run, output):
    manifest = json.loads((dataset/"map_manifest.json").read_text())
    truth = read_jsonl(dataset/"offline_gt/query_poses.jsonl")
    trace = read_jsonl(run/"localization.jsonl")
    if not truth or not trace:
        raise ValueError("empty actual trajectory")
    if any(t["map_id"] != manifest["map_id"] for t in trace):
        raise ValueError("map mismatch")
    gt_times = np.array([t["timestamp_ns"]/1e9 for t in truth])
    gt_poses = [t["T_map_base"] for t in truth]
    available = [t for t in trace if t["pose_time_s"] is not None]
    if not available:
        raise ValueError("no local odometry was emitted")
    times = np.array([t["pose_time_s"] for t in available])
    absolute, local, rows = [], [], []
    initial = None
    for gt, t in zip(truth, gt_times):
        # Pick the closest recorded actual pose, never invent interpolated
        # estimates through a missing interval or a localization-state change.
        index = int(np.argmin(abs(times-t)))
        row = available[index]
        st = row["pose_time_s"]
        exact_gt = interpolate(gt_times, gt_poses, st)
        if abs(st-t) > .06 or exact_gt is None:
            rows.append(dict(timestamp_s=t, valid=False, reason="missing_time_association"))
            continue
        local_pose = np.asarray(row["T_odom_base"])
        if initial is None:
            initial = exact_gt @ np.linalg.inv(local_pose)
        local_error = discrepancy(initial @ local_pose, exact_gt) if row["local_valid"] else None
        if local_error is not None:
            local.append(local_error)
        absolute_error = discrepancy(row["T_map_base"], exact_gt) if row["global_valid"] else None
        if absolute_error is not None:
            absolute.append(absolute_error)
        rows.append(dict(timestamp_s=t, valid=absolute_error is not None,
                         pose_time_s=st,output_time_s=float(row['timestamp_s']),
                         output_age_s=float(row['timestamp_s']-st),
                         local_error=local_error, absolute_error=absolute_error,
                         local_state=row["local_state"], global_state=row["global_state"],
                         truth_xy=exact_gt[:2,3].tolist(),
                         local_xy=(initial@local_pose)[:2,3].tolist(),
                         global_xy=None if absolute_error is None else np.asarray(row["T_map_base"])[:2,3].tolist()))
    decisions_path = run/"ace_decisions.jsonl"
    decisions = read_jsonl(decisions_path) if decisions_path.exists() else []
    good, wrong, unscorable, harmful = 0, 0, 0, 0
    latencies, decision_rows = [], []
    for row in decisions:
        candidate, decision = row["candidate"], row["decision"]
        if "latency_ms" in candidate:
            latencies.append(candidate["latency_ms"])
        if not decision["accepted"]:
            continue
        t = decision.get("capture_time_s",candidate["stamp_sec"])
        gt = interpolate(gt_times, gt_poses, t)
        if gt is None:
            unscorable += 1
            continue
        after=row.get("T_map_odom_after")
        i=int(np.argmin(abs(times-t)))
        if after is None or abs(times[i]-t)>.06:
            unscorable+=1
            continue
        # Score the actual accepted correction, including selected SPVLoc modes,
        # not whichever Top-1 candidate happens to be closest to GT.
        t=times[i]
        gt=interpolate(gt_times,gt_poses,t)
        if gt is None:
            unscorable+=1
            continue
        estimate=np.asarray(after)@np.asarray(available[i]["T_odom_base"])
        error = discrepancy(estimate, gt)
        ok = error[0] <= 1. and error[1] <= 30.
        good += int(ok)
        wrong += int(not ok)
        before, after = row.get("T_map_odom_before"), row.get("T_map_odom_after")
        worsened = None
        if before is not None and after is not None:
            # Both origins applied to one and the same real local pose.
            i = int(np.argmin(abs(times-t)))
            if abs(times[i]-t) <= .06:
                odom = np.asarray(available[i]["T_odom_base"])
                paired_gt = interpolate(gt_times, gt_poses, times[i])
                if paired_gt is not None:
                    be = discrepancy(np.asarray(before)@odom, paired_gt)
                    ae = discrepancy(np.asarray(after)@odom, paired_gt)
                    worsened = ae[0] > be[0]+.25 or ae[1] > be[1]+10
                    harmful += int(worsened)
        decision_rows.append(dict(timestamp_s=t, error=error, correct_1m30deg=ok,
                                  harmful_delta_25cm_or_10deg=worsened))
    path=path_statistics(gt_times,gt_poses)
    valid_times = [r["output_time_s"] for r in rows if r["valid"]]
    events_file=dataset/"offline_gt/events.json"
    recovery=recovery_statistics(json.loads(events_file.read_text()) if events_file.exists() else [],rows,gt_times,gt_poses)
    report = dict(status="EVALUATED_GAZEBO_DEVELOPMENT_PILOT_NOT_REAL_ROBOT_VALIDATION",
                  map_id=manifest["map_id"], samples=len(truth), duration_s=float(gt_times[-1]-gt_times[0]),
                  path_length_m=path['sampled_continuous_path_length_m'],path_statistics=path,absolute=statistics(absolute,len(truth)),
                  local_initial_origin_only=statistics(local,len(truth)),
                  absolute_alignment="none", local_alignment="one first-pose origin; offline diagnostic only",
                  first_global_output_after_route_start_s=None if not valid_times else valid_times[0]-gt_times[0],
                  accepted_correct=good, accepted_wrong_1m30deg=wrong, accepted_outside_evaluation=unscorable,
                  harmful_corrections=harmful,
                  decision_reasons=dict(Counter(r["decision"]["reason"] for r in decisions)),
                  inference_latency_ms=None if not latencies else dict(p50=float(np.median(latencies)),p95=float(np.quantile(latencies,.95))),
                  measured_runtime_by_method=runtime_statistics(decisions),
                  gt_usage="offline evaluation only; reference GT restricted to map training",
                  output_age_s=None if not rows else dict(
                      p50=float(np.median([r['output_age_s'] for r in rows if 'output_age_s' in r])) if any('output_age_s' in r for r in rows) else None,
                      p95=float(np.percentile([r['output_age_s'] for r in rows if 'output_age_s' in r],95)) if any('output_age_s' in r for r in rows) else None),
                  recovery_events=recovery,
                  recovery_experiment=("measured simulator kidnapping" if any(r['event_observed_in_gt'] for r in recovery) else
                                       "requested kidnapping not observed in captured GT" if recovery else "not measured: no recorded failure event"),
                  test_tuning=False)
    fault_file=dataset/"offline_gt/camera_fault.json"
    if fault_file.exists():
        fault=json.loads(fault_file.read_text())
        start,end=[fault["origin_timestamp_s"]+x for x in fault["interval_s"]]
        affected=[r for r in rows if start<=r["timestamp_s"]<end]
        report["camera_fault"]=dict(protocol=fault,evaluated_frames=len(affected),
            local_coverage_during_blackout=None if not affected else sum(r.get("local_error") is not None for r in affected)/len(affected),
            global_coverage_during_blackout=None if not affected else sum(r["valid"] for r in affected)/len(affected),
            scope="synthetic image blackout, not natural occlusion")
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"metrics.json", report)
    write_json(output/"per_frame_errors.json", rows)
    write_json(output/"accepted_correction_errors.json", decision_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    for key, label, color in [("truth_xy","Gazebo GT (offline)","black"), ("local_xy","Wheel + IMU (initial origin)","tab:blue"), ("global_xy","Gated absolute + EKF (map frame)","tab:orange")]:
        pts=np.array([r.get(key) if r.get(key) is not None else [np.nan,np.nan] for r in rows])
        ax[0].plot(pts[:,0],pts[:,1],label=label,color=color)
    ax[0].set(xlabel="map x (m)",ylabel="map y (m)",title="Actual robot trajectory")
    ax[0].axis("equal")
    ax[0].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.20),ncol=1)
    for key,label in [("local_error","Wheel + IMU"),("absolute_error","Gated absolute + EKF")]:
        ax[1].plot([r["timestamp_s"]-gt_times[0] for r in rows],
                   [r[key][0] if r.get(key) is not None else np.nan for r in rows], label=label)
    ax[1].set(xlabel="simulation time (s)",ylabel="translation error (m)",title="Missing predictions remain gaps")
    for index,jump in enumerate(path['detected_jumps']):
        ax[1].axvline(jump['timestamp_s']-gt_times[0],color='crimson',linestyle='--',alpha=.8,
                      label='Actual GT relocation jump' if index==0 else None)
    if 'camera_fault' in report:
        protocol=report['camera_fault']['protocol']
        start,end=[protocol['origin_timestamp_s']+x-gt_times[0] for x in protocol['interval_s']]
        ax[1].axvspan(start,end,color='gray',alpha=.2,label='Declared stereo image blackout')
    ax[1].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2)
    fig.tight_layout();fig.subplots_adjust(bottom=.30);fig.savefig(output/"trajectory.png",dpi=160);plt.close(fig)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(evaluate(a.dataset,a.run,a.output),indent=2))


if __name__ == "__main__":
    main()
