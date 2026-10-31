"""Fetch minimal scene dependency closures from official HF repositories.
Every fetched file is revision-pinned and logged with local SHA256.
Use audit_assets.py to compare sizes and LFS hashes against the official tree.
"""

import argparse, concurrent.futures, hashlib, json, time
from pathlib import Path
import requests


class Fetcher:
    def __init__(self, repo, root):
        self.repo = repo
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sha = requests.get(f"https://huggingface.co/api/datasets/{repo}").json()[
            "sha"
        ]
        self.records = {}

    def tree(self, folder=""):
        url = f"https://huggingface.co/api/datasets/{self.repo}/tree/{self.sha}/{folder}?limit=1000"
        out = []
        while url:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            out += r.json()
            url = r.links.get("next", {}).get("url")
        return out

    def get(self, path):
        p = self.root / path
        url = f"https://huggingface.co/datasets/{self.repo}/resolve/{self.sha}/{path}"
        if not p.exists():
            for n in range(5):
                try:
                    r = requests.get(url, timeout=120)
                    r.raise_for_status()
                    p.parent.mkdir(parents=True, exist_ok=True)
                    temporary = p.with_suffix(p.suffix + ".download.tmp")
                    temporary.write_bytes(r.content)
                    temporary.replace(p)
                    break
                except Exception:
                    if n == 4:
                        raise
                    time.sleep(2**n)
        data = p.read_bytes()
        self.records[path] = dict(
            path=path, url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
        )
        return p

    def many(self, paths):
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(self.get, sorted(set(paths))))

    def save(self):
        (self.root / "acquisition.json").write_text(
            json.dumps(
                dict(
                    repo=self.repo, revision=self.sha, files=list(self.records.values())
                ),
                indent=2,
            )
        )


def hssd(root, n):
    f = Fetcher("hssd/hssd-hab", root)
    f.many(
        [
            "README.md",
            "hssd-hab.scene_dataset_config.json",
            "scene_splits.yaml",
            "semantics/hssd-hab_semantic_lexicon.json",
            "metadata/hssd_obj_semantics_condensed.csv",
            "metadata/objects.json",
        ]
    )
    scenes = sorted(f.tree("scenes"), key=lambda x: x["size"])[:n]
    templates = set()
    for s in scenes:
        p = f.get(s["path"])
        d = json.loads(p.read_text())
        sid = p.name.split(".")[0]
        stage = d["stage_instance"]["template_name"]
        f.many(
            [
                stage + ".stage_config.json",
                stage + ".glb",
                f"semantics/scenes/{sid}.semantic_config.json",
            ]
        )
        templates.update(o["template_name"] for o in d.get("object_instances", []))
    paths = []
    for char in sorted({s[0] for s in templates} | {"openings"}):
        for item in f.tree("objects/" + char):
            if (
                item["type"] == "file"
                and item["path"].split("/")[-1].split(".")[0] in templates
            ):
                paths.append(item["path"])
    f.many(paths)
    # Asset paths can include additional meshes. Resolve from configs, not guessed IDs.
    extra = []
    for path in paths:
        if path.endswith(".object_config.json"):
            p = f.root / path
            d = json.loads(p.read_text())
            for k in ["render_asset", "collision_asset"]:
                if d.get(k):
                    extra.append(str((Path(path).parent / d[k])))
    f.many(extra)
    f.get("metadata/fpmodels-with-decomposed.csv")
    f.save()
    print(
        "HSSD complete",
        len(f.records),
        "files",
        len(scenes),
        "scenes",
        len(templates),
        "object templates",
        flush=True,
    )


def replica(root):
    f = Fetcher("ai-habitat/ReplicaCAD_dataset", root)
    todo = [""]
    files = []
    while todo:
        folder = todo.pop()
        for x in f.tree(folder):
            if x["type"] == "directory":
                todo.append(x["path"])
            else:
                files.append(x["path"])
    f.many(files)
    f.save()
    print("ReplicaCAD interactive assets", len(files), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--dataset", choices=["hssd", "replica"], default="hssd")
    p.add_argument("--scenes", type=int, default=3)
    a = p.parse_args()
    hssd(a.root, a.scenes) if a.dataset == "hssd" else replica(a.root)
