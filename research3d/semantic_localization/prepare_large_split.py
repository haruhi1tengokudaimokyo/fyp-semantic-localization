"""Freeze publisher-ordered candidates, including bounded same-split reserves.

Existing captured houses and the active 16-house manifest are never overwritten.
Only fixed source geometry criteria are used; no localization results are read.
"""
import gzip, hashlib, json, subprocess
from pathlib import Path
from procthor_geometry import convert

ROOT = Path('/data/fyp_3d/datasets/procthor_spvloc')
SOURCE = Path('/data/fyp_3d/sources/procthor-10k')
MANIFEST = ROOT / 'building_candidates_100_20_20.json'
TARGET = {'train': 100, 'val': 20, 'test': 20}

def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def main():
    if MANIFEST.exists():
        value = json.loads(MANIFEST.read_text())
        assert value['target'] == TARGET
        return value
    records, seen = [], set()
    for split, base in [('train', 0), ('val', 3000), ('test', 3250)]:
        selected = 0
        for index, line in enumerate(gzip.open(SOURCE / (split + '.jsonl.gz'), 'rt')):
            house = json.loads(line)
            if len(house['rooms']) < 2 or not house['doors'] or not house['windows']:
                continue
            identity = {k: house[k] for k in ['rooms','walls','doors','windows','objects']}
            sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if sha in seen:
                raise ValueError('Duplicated house geometry in publisher splits')
            seen.add(sha)
            sid = base + selected
            scene = ROOT / f'scene_{sid:05d}'
            scene.mkdir(exist_ok=True)
            path = scene / 'procthor_house.json'
            if path.exists():
                assert json.loads(path.read_text()) == house, 'Frozen identity mismatch'
            else:
                atomic(path, house)
            annotation = scene / 'annotation_3d.json'
            if not annotation.exists():
                atomic(annotation, convert(house))
            records.append(dict(scene_id=sid, split=split, official_house_index=index,
                                source_geometry_sha256=sha, candidate_rank=selected,
                                reserve=selected >= TARGET[split]))
            selected += 1
            if selected == TARGET[split] * 3 // 2:
                break
        assert selected == TARGET[split] * 3 // 2
    value = dict(dataset='ProcTHOR-10K', target=TARGET, houses=records,
                 source='https://github.com/allenai/procthor-10k', license='Apache-2.0',
                 source_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
                 selection='Publisher order, >=2 rooms, >=1 door and window; 50% bounded reserves. Replace only on predeclared acquisition/geometry/image failures within the same official split.',
                 source_split_authoritative=True, no_test_localization_scores_used=True)
    atomic(MANIFEST, value)
    return value

if __name__ == '__main__':
    value = main()
    print('FROZEN_CANDIDATES', len(value['houses']), 'TARGET', value['target'])
