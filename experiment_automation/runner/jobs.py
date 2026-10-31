"""Concrete CPU/download jobs used by the durable DAG; no placeholder success paths."""

import argparse, hashlib, json, os, sys, tarfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research3d/scripts"))
BASE = Path("/data/fyp_3d")


def models():
    import requests, zipfile

    url = "https://storage.googleapis.com/niantic-lon-static/research/ace/ace_models.tar.gz"
    out = BASE / "models/ace_official"
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
            for m in archive:
                if (
                    not m.isfile()
                    or not m.name.startswith("ace_models/7Scenes/")
                    or not m.name.endswith(".pt")
                ):
                    continue
                name = Path(m.name).name
                p = out / name
                data = archive.extractfile(m).read()
                assert len(data) == m.size
                tmp = p.with_suffix(".tmp")
                tmp.write_bytes(data)
                with zipfile.ZipFile(tmp) as z:
                    assert z.testzip() is None
                tmp.replace(p)
                records[name] = {
                    "member": m.name,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                print(name, len(data), flush=True)
    assert all(
        "7scenes_" + n + ".pt" in records
        for n in ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
    )
    from core import atomic

    atomic(
        out / "acceptance.json",
        {"source": url, "complete_7scenes_heads": True, "models": records},
    )


def selftest(path, fail_once=False):
    from core import atomic

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    cp = p / "checkpoint.json"
    i = json.loads(cp.read_text())["step"] if cp.exists() else 0
    if fail_once and not (p / "failed_once").exists():
        (p / "failed_once").write_text(
            "intentional infrastructure recovery test; not experiment data"
        )
        raise RuntimeError("Intentional one-time self-test fault")
    while i < 12:
        i += 1
        atomic(cp, {"step": i, "payload_pid": os.getpid(), "actual_time": time.time()})
        print("CHECKPOINT", i, flush=True)
        time.sleep(0.5)
    atomic(
        p / "complete.json",
        {"status": "passed", "steps": i, "infrastructure_test_only": True},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("job", choices=["ace-models", "selftest"])
    p.add_argument("--path")
    p.add_argument("--fail-once", action="store_true")
    a = p.parse_args()
    models() if a.job == "ace-models" else selftest(a.path, a.fail_once)
