"""Package accepted ACE/MAC-VO evidence without recomputing or changing it."""
import hashlib, json, statistics, zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = Path('/data/fyp_3d/runs/reproduction')
DELIVERY = REPO.parents[1] / 'outputs/FYP_Delivery_20261102'
COMPACT = REPO / 'results/reproduction_20261102'

def make_zip(name, files, note):
    path = DELIVERY / name
    temp = path.with_suffix('.zip.tmp')
    checksums = []
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr('README.md', note)
        for source, relative in sorted(files, key=lambda item: item[1]):
            assert source.is_file() and not source.is_symlink()
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            checksums.append(f'{digest}  {relative}\n')
            archive.write(source, relative)
        archive.writestr('SHA256SUMS.txt', ''.join(checksums))
    assert temp.stat().st_size <= 500_000_000, 'User archive size ceiling'
    with zipfile.ZipFile(temp) as archive:
        assert archive.testzip() is None
        for line in archive.read('SHA256SUMS.txt').decode().splitlines():
            digest, relative = line.split('  ', 1)
            assert hashlib.sha256(archive.read(relative)).hexdigest() == digest
    temp.replace(path)
    return dict(file=name, bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), entries=len(files))

def main():
    DELIVERY.mkdir(parents=True,exist_ok=True);COMPACT.mkdir(parents=True,exist_ok=True)
    ace_files, ace_rows = [], []
    for base in sorted((RUNS/'ACE/7scenes').glob('*/buffer8M_mmap')):
        summary = base/'official_evaluation_v2/summary.json'
        if not summary.exists(): continue
        row=json.loads(summary.read_text()); row['scene']=base.parent.name;ace_rows.append(row)
        for source in base.rglob('*'):
            if source.is_file() and not source.is_symlink() and 'official_evaluation' not in source.relative_to(base).parts:
                ace_files.append((source, str(source.relative_to(RUNS))))
    assert len(ace_rows)==7 and sum(row['n'] for row in ace_rows)==17000
    ace=dict(scenes=ace_rows,frames=17000,macro_recall_5cm5deg=statistics.mean(row['recall_5cm_5deg'] for row in ace_rows),
             pose_condition='Original D-SLAM, not Pseudo-GT/SfM',paper_comparable_macro_percent=80.8,
             official_8M_buffer_training=True,unmodified_official_evaluation=True)
    (COMPACT/'ACE_summary.json').write_text(json.dumps(ace,indent=2))
    mac_files=[];mac_rows=[]
    for source in sorted((RUNS/'MACVO').rglob('*')):
        if not source.is_file() or source.is_symlink():continue
        # Keep complete trajectories, real full-pipeline ablations, independent
        # metrics, exact replay acceptance and timings. Dense correspondence
        # arrays stay on disk; a SHA index documents the omitted bulky evidence.
        if 'solver_inputs' in source.parts or source.name=='observations.npz':continue
        mac_files.append((source,str(source.relative_to(RUNS))))
    for source in sorted((RUNS/'MACVO/diagnostics_20261102').glob('*/*/official_metrics.json')):
        row=json.loads(source.read_text()); runtime=json.loads((source.parent/'summary.json').read_text())
        mac_rows.append(dict(sequence=source.parent.parent.name,condition=source.parent.name,
                             frames=row['frames'],ATE_m=row['independent_ATE_m'],lost_frames=row['lost_frames'],
                             relative_metrics=row['metrics'],mean_runtime_ms=runtime['runtime_mean_ms'],
                             full_source_sequence=row['full_source_sequence'],GT_online=row['GT_online']))
    assert len(mac_rows)==4
    demo=json.loads((RUNS/'MACVO/official_metrics_existing_demo_v2.json').read_text())
    for row in demo['results'].values():
        directory=Path(row['source'])
        for source in directory.rglob('*'):
            if source.is_file() and not source.is_symlink() and source.name!='observations.npz':
                mac_files.append((source,str(Path('MACVO/demo')/directory.name/source.relative_to(directory))))
    mac=dict(full_pipeline_controls=mac_rows,demo=demo,
             static_initialization_replay='Separate controlled backend experiment; not full VO ablation',
             speed_note='FP32 GTX1660SUPER about 1.3sec/frame; no accepted FP16 speed claim; not author hardware')
    (COMPACT/'MACVO_summary.json').write_text(json.dumps(mac,indent=2))
    note='# ACE and MAC-VO accepted results\n\nACE: 7 Scenes / 17000 queries; original D-SLAM poses, 8M buffer training. Compare with 80.8%, not 97.1% Pseudo-GT.\n\nMAC-VO: author demo + full Hospital and AbandonedFactory v1 Easy P000, and actual full-pipeline covariance/identity controls. GT is offline evaluation only; no scale fit or dropped-frame interpolation. Static-init backend replay is separately named. Full original paper benchmark coverage is incomplete.\n\nDense correspondence caches, raw datasets and environments remain on the host and are omitted. SHA256SUMS verifies every packaged source.\n'
    (COMPACT/'README.md').write_text(note)
    archives=[make_zip('ACE_7Scenes_results.zip',ace_files,note),make_zip('MACVO_full_sequence_results.zip',mac_files,note)]
    (DELIVERY/'archives.json').write_text(json.dumps(archives,indent=2))
    (COMPACT/'archives.json').write_text(json.dumps(archives,indent=2))
    print(json.dumps(archives,indent=2))

if __name__=='__main__':main()
