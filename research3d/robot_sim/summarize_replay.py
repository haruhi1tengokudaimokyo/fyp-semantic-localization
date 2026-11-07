"""Independently verify saved replay outputs against offline GT, then summarize.

This read-only evaluator never invokes a frontend or updates fusion gates.  Its
rotation check uses scipy Rotation rather than the replay metrics implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from research3d.fusion.contracts import read_jsonl, sha256_file, write_json


def errors(estimate, truth):
    estimate, truth = np.asarray(estimate), np.asarray(truth)
    return (float(np.linalg.norm(estimate[:3, 3] - truth[:3, 3])),
            float(np.degrees(Rotation.from_matrix(truth[:3, :3].T @ estimate[:3, :3]).magnitude())))


def correct(estimate, truth):
    t, r = errors(estimate, truth)
    return t <= 1.0 and r <= 30.0


def summarize(root: Path, output: Path):
    replay = root / 'fusion_replay'
    receipt = json.loads((replay / 'receipt.json').read_text())
    for path, expected in receipt['input_sha256'].items():
        if sha256_file(Path(path)) != expected:
            raise ValueError('Frozen replay input changed: ' + path)
    dataset = root / 'stereo_dataset'
    manifest = json.loads((dataset / 'map_manifest.json').read_text())
    frames = read_jsonl(dataset / 'query/frames.jsonl')
    gt_rows = read_jsonl(dataset / 'offline_gt/query_poses.jsonl')
    truth = {r['query_id']: r['T_map_camera'] for r in gt_rows}
    assert len(truth) == len(gt_rows) == len(frames)
    assert all(r['timestamp_ns'] == r['right_timestamp_ns'] for r in frames)
    observations = {(r['source'], r['query_id']): r for r in read_jsonl(replay / 'observations.jsonl')}
    rows = []
    checked = 0
    for mode in receipt['modes']:
        for variant in receipt['variants']:
            folder = replay / mode / variant
            predictions = read_jsonl(folder / 'outputs.jsonl')
            assert len(predictions) == len(truth)
            assert set(r['query_id'] for r in predictions) == set(truth)
            saved = json.loads((folder / 'metrics.json').read_text())
            per_query = {r['query_id']: r for r in saved['system']['per_query']}
            age_limit = saved['system']['thresholds']['max_output_age_s']
            ts, rs, ages = [], [], []
            successes = wrong_accepts = accepted = harmful = assessable = 0
            for p in predictions:
                q = p['query_id']; target = truth[q]
                if p['valid']:
                    t, r = errors(p['T_map_camera'], target)
                    age = p['output_time_s'] - p['pose_time_s']
                    success = t <= 1 and r <= 30 and (age_limit is None or age <= age_limit)
                    assert abs(t - per_query[q]['translation_m']) < 1e-9
                    assert abs(r - per_query[q]['rotation_deg']) < 1e-3
                    assert success == per_query[q]['success']
                    ts.append(t); rs.append(r); ages.append(age); successes += success
                else:
                    assert per_query[q]['success'] is False
                for ref in p['accepted_observations']:
                    original = observations[ref['source'], ref['query_id']]
                    candidate = next(c for c in original['candidates'] if c['rank'] == ref['rank'])
                    accepted += 1
                    wrong_accepts += not correct(candidate['T_map_camera'], truth[ref['query_id']])
                if p['accepted_observations'] and p['T_before_correction'] is not None:
                    assessable += 1
                    harmful += correct(p['T_before_correction'], target) and not correct(p['T_map_camera'], target)
                checked += 1
            aggregate = saved['system']['aggregate']
            corrections = saved['corrections']
            assert successes == aggregate['successes']
            assert len(ts) == aggregate['valid_outputs']
            assert wrong_accepts == corrections['wrong_accepted_candidate_count']
            assert accepted == corrections['accepted_candidate_count']
            assert assessable == corrections['bad_corrections_assessable']
            assert (harmful if assessable else None) == corrections['correct_to_wrong_count']
            rows.append(dict(mode=mode, variant=variant, queries=len(predictions), valid_outputs=len(ts),
                successes=int(successes), global_success_percent=100 * successes / len(predictions),
                translation_median_m=float(np.median(ts)) if ts else None,
                translation_rmse_m=float(np.sqrt(np.mean(np.square(ts)))) if ts else None,
                rotation_median_deg=float(np.median(rs)) if rs else None,
                output_age_p95_s=float(np.quantile(ages, .95)) if ages else None,
                accepted_candidates=accepted, wrong_accepted_candidates=int(wrong_accepts),
                harmful_correct_to_wrong_events=int(harmful) if assessable else None,
                correction_events_assessable=assessable, states=dict(Counter(r['status'] for r in predictions)),
                outputs_sha256=sha256_file(folder / 'outputs.jsonl')))
    cells = Counter(both_correct=0, ace_only_correct=0, spvloc_only_correct=0, both_wrong=0)
    top3_correct = 0
    for q, target in truth.items():
        a = observations['ace', q]; s = observations['spvloc', q]
        a_ok = bool(a['candidates']) and correct(a['candidates'][0]['T_map_camera'], target)
        s_ok = bool(s['candidates']) and correct(s['candidates'][0]['T_map_camera'], target)
        cells['both_correct' if a_ok and s_ok else 'ace_only_correct' if a_ok else 'spvloc_only_correct' if s_ok else 'both_wrong'] += 1
        top3_correct += any(correct(c['T_map_camera'], target) for c in s['candidates'][:3])
    complementary = json.loads((replay / 'complementarity.json').read_text())['aggregate']
    assert dict(cells) == complementary['four_cells']
    assert top3_correct == complementary['spvloc_top3_oracle_successes']
    visual_dir = root / 'three_model_visualization'
    artifacts = {name: dict(path=str((visual_dir / name).resolve()), sha256=sha256_file(visual_dir / name),
                           bytes=(visual_dir / name).stat().st_size)
                 for name in ['localization_replay.gif', 'trajectory.png', 'final_frame.png', 'visualization_receipt.json']}
    macvo = json.loads((root / 'macvo_queries/metrics.json').read_text())
    report = dict(status='ACTUAL_FRONTENDS_REPLAYED_AND_INDEPENDENTLY_VERIFIED',
        scope='Single known-map Gazebo development building; not an original-paper benchmark or physical robot validation',
        queries=len(frames), duration_s=(frames[-1]['timestamp_ns'] - frames[0]['timestamp_ns']) / 1e9,
        map_id=manifest['map_id'], map_sha256=manifest['map_sha256'], source=manifest['source'],
        thresholds=dict(translation_m=1, rotation_deg=30, service_fifo_output_age_s=.5),
        independent_verification=dict(status='PASS', output_rows_checked=checked,
            method='Direct numpy translation and scipy SO(3) angle from output poses plus original GT; no replay metric function reused',
            checks=['every evaluation output', 'success counts', 'raw accepted-candidate correctness',
                    'harmful corrections', 'ACE/SPVLoc complementarity', 'frozen source checksums', 'exact stereo timestamps']),
        complementarity=dict(complementary, per_query='See fusion_replay/complementarity.json'),
        results=rows,
        macvo_local_diagnostic={k: macvo[k] for k in ['n', 'ate_se3_aligned_rmse_m', 'ate_first_pose_anchored_rmse_m',
            'path_length_m', 'lost_frames', 'input_sampling', 'world_absolute_localization', 'alignment_warning']},
        interpretation=[
            'B MAC-VO+ACE improves complete-trajectory absolute-pose coverage in ideal offline replay; this is not real-time performance.',
            'SPVLoc has zero successful top1/top3 candidates in this Gazebo appearance domain; semantic complementarity is not established.',
            'D/E preserve conflicting modes and abstain; zero valid global outputs must not be described as successful three-model fusion.',
            'A has arbitrary odometry gauge, so global coverage is undefined operationally; consult separate local MAC-VO diagnostics.',
            'Measured serial-service FIFO is a warm-model scheduling lower bound retaining inference backlog; excludes model swaps and unmeasured I/O.',
            'All measured-service outputs fail the 0.5s freshness requirement. Offline accuracy must not imply real-time readiness.',
            'No independent kidnapping events occur in this shared 561-image capture; recovery performance is not measured here.',
            'Scene03000 is an existing validation building reused for system development; this run is not held-out cross-building generalization evidence.'
        ],
        GT_online=False, frontend_predictions_modified=False, thresholds_tuned_on_query_GT=False,
        interpolation=False, real_time_claim=False, input_sha256=receipt['input_sha256'], artifacts=artifacts)
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root.resolve(strict=True), args.output)
    print(json.dumps(dict(status=report['status'], rows=report['independent_verification']['output_rows_checked'], output=str(args.output))))


if __name__ == '__main__':
    main()
