"""Check scene dependency closure, beyond hashes of files that happened to download."""

import argparse, json
from pathlib import Path


def main(a):
    root = Path(a.dataset)
    templates = {
        p.name.split(".")[0]: p
        for p in (root / "objects").rglob("*.object_config.json")
    }
    missing = []
    needed = set()
    scenes = list((root / "scenes").glob("*.scene_instance.json"))
    for scene in scenes:
        cfg = json.loads(scene.read_text())
        stage = cfg["stage_instance"]["template_name"]
        for suffix in [".glb", ".stage_config.json"]:
            if not (root / (stage + suffix)).is_file():
                missing.append(stage + suffix)
        for o in cfg.get("object_instances", []):
            needed.add(o["template_name"])
    for name in sorted(needed):
        if name not in templates:
            missing.append("object template " + name)
            continue
        p = templates[name]
        cfg = json.loads(p.read_text())
        for key in ["render_asset", "collision_asset"]:
            if cfg.get(key) and not (p.parent / cfg[key]).is_file():
                missing.append(str(p.parent / cfg[key]))
    result = dict(
        scenes=len(scenes),
        required_templates=len(needed),
        missing=missing,
        complete=not missing,
    )
    (root / "closure_integrity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    assert not missing


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    main(p.parse_args())
