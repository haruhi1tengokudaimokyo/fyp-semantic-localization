"""Freeze conservative HSSD synthetic-house groups before rendering/training.
Merge shared house-prefix and byte-identical architectural meshes across groups.
These are synthetic layout groups, never asserted to be real physical buildings.
"""

import argparse, hashlib, json, random
from pathlib import Path


def main(a):
    root = Path(a.dataset)
    rows = []
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(y)] = find(x)

    seen = {}
    for p in sorted((root / "scenes").glob("*.scene_instance.json")):
        scene = p.name.split(".")[0]
        group = scene.split("_")[0]
        find(group)
        cfg = json.loads(p.read_text())
        mesh = root / (cfg["stage_instance"]["template_name"] + ".glb")
        h = hashlib.sha256(mesh.read_bytes()).hexdigest()
        if h in seen:
            union(group, seen[h])
        seen[h] = group
        rows.append(dict(scene=scene, house_prefix=group, stage_sha256=h))
    groups = sorted({find(r["house_prefix"]) for r in rows})
    assert len(groups) >= 12, "Need at least 12 audited layout groups"
    random.Random(a.seed).shuffle(groups)
    nv = max(2, len(groups) // 6)
    nt = max(2, len(groups) // 6)
    splits = {
        g: ("test" if i < nt else "val" if i < nt + nv else "train")
        for i, g in enumerate(groups)
    }
    for r in rows:
        r.update(
            building_group=find(r["house_prefix"]),
            split=splits[find(r["house_prefix"])],
        )
    result = dict(
        seed=a.seed,
        independent_layout_groups=len(groups),
        simulated=True,
        real_physical_building_identity_claim=False,
        grouping="shared original house prefix OR identical architectural GLB SHA256; splits frozen before rendering/model fitting",
        scenes=rows,
        split_groups={
            s: sorted(g for g in groups if splits[g] == s)
            for s in ["train", "val", "test"]
        },
        limitations=[
            "Synthetic house groups; near-duplicate nonidentical layouts remain possible; no real-building claim",
            "Native object lexicon lacks wall/window/column coverage; ignore unknown labels",
        ],
    )
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert (
            json.loads(target.read_text()) == result
        ), "Refusing changed split protocol"
    else:
        target.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "scenes"}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("output")
    p.add_argument("--seed", type=int, default=1305)
    main(p.parse_args())
